// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef REDUCE_POLYTOPE_HPP
#define REDUCE_POLYTOPE_HPP

#include <cmath>
#include <vector>
#include <limits>
#include "convex_bodies/constrained_polytope.hpp"
#include "preprocess/constrained_polytope/highs_polytope.hpp"

// Configuration for reducing a polytope.
struct ReduceConfig {
    // Variables whose observed range is below this are fixed.
    double reduce_tol = 1e-7;

    // Observed variation below this is treated as zero.
    double observe_tol = 1e-6;
};

// Result of reducing a polytope
template <typename Point>
struct ReduceResult {
    // The reduced polytope.
    ConstrainedPolytope<Point> polytope;

    // The number of variables fixed.
    unsigned fixed_vars = 0;

    // The number of LPs solved.
    unsigned solved_lps = 0;

    // Tracks if the reduction algorithm ran.
    bool valid = false;
};

// Substitues each fixed variable x_k = val out of every row except from its singleton row:
// 
//      .. + a*x_k + a_k+1 + .. <= b_i -> .. + a_k+1 + .. <= (b_i-x_k)
// @param highs the HiGHS model
// @param fixed the vector holding the fixed variables
// @param first_fix_row the index where fixed rows start
inline void substitute_fixed(Highs& highs,
                             std::vector<std::pair<HighsInt, double>> const& fixed,
                             HighsInt first_fix_row)
{
    if (first_fix_row == 0) return;

    HighsLp lp = highs.getLp();
    lp.a_matrix_.ensureColwise();
    auto const& A = lp.a_matrix_;

    std::vector<double> lower = lp.row_lower_;
    std::vector<double> upper = lp.row_upper_;

    for (auto const& [k, val] : fixed) {
        for (HighsInt p = A.start_[k]; p < A.start_[k+1]; ++p) {
            HighsInt const r = A.index_[p];
            if (r >= first_fix_row) continue;

            double const shift = A.value_[p]*val;
            lower[r] -= shift;
            upper[r] -= shift;
            highs.changeCoeff(r, k, 0.0);
        }
    }

    highs.changeRowsBounds(0, first_fix_row, lower.data(), upper.data());
}

// Removes rows with no nonzero coefficients left. Such as row reads ai: 0 <= bi.
// @return the number of rows removed
inline unsigned remove_empty_rows(Highs& highs, double tol=1e-9)
{
    HighsLp lp = highs.getLp();
    lp.a_matrix_.ensureColwise();
    auto const& A = lp.a_matrix_;

    // Counter for the nonzeros per row
    std::vector<HighsInt> nonzero(lp.num_row_, 0);

    for (HighsInt p = 0; p < A.start_[lp.num_col_]; ++p)
        if (std::abs(A.value_[p]) > tol) ++nonzero[A.index_[p]];

    // Rows to delete (with zero norm)
    std::vector<HighsInt> to_remove;

    for (HighsInt i = 0; i < lp.num_row_; ++i) {
        if (nonzero[i] > 0) continue;
        to_remove.push_back(i);
    }

    if (!to_remove.empty())
        highs.deleteRows((HighsInt)to_remove.size(), to_remove.data());

    return (unsigned)to_remove.size();
}

// Fixes degenerate dimensions of a ConstrainedPolytope by identifying variables that take
// a single value over the feasible set and moving them into A_eq.
//
// The method builds a HiGHS LP from the polytope, runs one LP to collect observed
// variable values, and for each variable that shows no variation across the solutions
// seen so far it solves a max and min LP to confirm the range.
//
// The reduced polytope is returned with the fixed variables expressed as rows of A_eq.
// @tparam Point the point type
// @param P the input polytope
// @param config the file
// @return the result
template <typename Point>
ReduceResult<Point> reduce_polytope(ConstrainedPolytope<Point> const& P,
                                    ReduceConfig const& config = ReduceConfig{})
{
    unsigned const d = P.getDimension();

    ReduceResult<Point> res;

    Highs highs;
    configure_highs(highs);
    build_lp_model(P, highs);

    auto run_lp = [&]() -> bool {
        highs.run();
        ++res.solved_lps;

        if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
            highs.clearSolver();
            highs.run();
            ++res.solved_lps;
        }

        return highs.getModelStatus() == HighsModelStatus::kOptimal;
    };

    // Checks that this region describes a non empty and bounded polytope
    if (!run_lp()) return res;

    // Observed min/max values for each variables across all LP solutions seen
    std::vector<double> obs_min(d, std::numeric_limits<double>::infinity());
    std::vector<double> obs_max(d, -std::numeric_limits<double>::infinity());

    auto observe = [&]() {
        auto const& sol = highs.getSolution().col_value;
        for (unsigned j = 0; j < d; ++j) {
            if (sol[j] < obs_min[j]) obs_min[j] = sol[j];
            if (sol[j] > obs_max[j]) obs_max[j] = sol[j];
        }
    };

    observe();

    auto observe_variation = [&](unsigned k) {
        return std::abs(obs_max[k]-obs_min[k]) > config.observe_tol;
    };

    HighsInt const first_fix_row = highs.getLp().num_row_;

    // Holds the fixed vars.
    std::vector<std::pair<HighsInt, double>> fixed;

    for (unsigned k = 0; k < d; ++k) {
        // Prunes based on past observation
        if (observe_variation(k)) continue;

        highs.changeColCost((HighsInt)k, 1.0);
        highs.changeObjectiveSense(ObjSense::kMaximize);

        if (!run_lp()) {
            highs.changeColCost((HighsInt)k, 0.0);
            continue;
        }

        double max_val = highs.getObjectiveValue();
        observe();

        if (observe_variation(k)) {
            highs.changeColCost((HighsInt)k, 0.0);
            continue;
        }

        highs.changeObjectiveSense(ObjSense::kMinimize);

        if (!run_lp()) {
            highs.changeColCost((HighsInt)k, 0.0);
            continue;
        }

        double min_val = highs.getObjectiveValue();
        observe();

        highs.changeColCost((HighsInt)k, 0.0);

        if (std::abs(max_val-min_val) < config.reduce_tol) {
            double val = (max_val+min_val)/2.0;
            HighsInt id = (HighsInt)k;
            double coeff = 1.0;
            highs.addRow(val, val, 1, &id, &coeff);
            fixed.emplace_back(id, val);
            ++res.fixed_vars;
        }
    }

    // Removes the fixed variables from the columns of the LP.
    if (!fixed.empty()) {
        substitute_fixed(highs, fixed, first_fix_row);
        res.fixed_vars = fixed.size();

        remove_empty_rows(highs);
    }
    
    res.polytope = extract_polytope<Point>(highs);
    res.valid = true;
    return res;
}
#endif