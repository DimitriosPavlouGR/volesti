// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef HIGHS_POLYTOPE_HPP
#define HIGHS_POLYTOPE_HPP

#include <cmath>
#include <vector>
#include "Highs.h"
#include "convex_bodies/constrained_polytope.hpp"

// Configures a HiGHS instance for use in simplification.
// @param highs the HiGHS instance to configure
inline void configure_highs(Highs& highs) {
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("presolve", "on");
    highs.setOptionValue("solver", "simplex");
    highs.setOptionValue("simplex_strategy", 4);
}

// Loads the polytope A_eq x = b_eq, A_in x <= b_in into a HiGHS instance.
//
// All variables are added as free (no column bounds)
// @tparam Point the point type
// @param P the polytope
// @param highs the HiGHS instance to load into, must be empty
template <typename Point>
void build_lp_model(ConstrainedPolytope<Point> const& P,
                    Highs& highs)
{   
    using MT = typename ConstrainedPolytope<Point>::MT;

    auto const& A_eq = P.getEqualities();
    auto const& b_eq = P.getEqualityRHS();
    auto const& A_in = P.getInequalities();
    auto const& b_in = P.getInequalityRHS();
    unsigned const d = P.getDimension();
    unsigned const m_eq = P.getNumEqualities();
    unsigned const m_in = P.getNumInequalities();

    // Adds the variables
    for (unsigned j = 0; j < d; ++j)
        highs.addVar(-kHighsInf, kHighsInf);

    // Adds the equality rows
    for (unsigned i = 0; i < m_eq; ++i) {
        std::vector<HighsInt> ids;
        std::vector<double> vals;

        for (typename MT::InnerIterator it(A_eq, i); it; ++it) {
            ids.push_back((HighsInt)it.col());
            vals.push_back((double)it.value());
        }

        highs.addRow((double)b_eq(i), (double)b_eq(i), (HighsInt)ids.size(), 
                     ids.data(), vals.data());
    }

    // Adds the inequality rows
    for (unsigned i = 0; i < m_in; ++i) {
        std::vector<HighsInt> ids;
        std::vector<double> vals;

        for (typename MT::InnerIterator it(A_in, i); it; ++it) {
            ids.push_back((HighsInt)it.col());
            vals.push_back((double)it.value());
        }

        highs.addRow(-kHighsInf, (double)b_in(i), (HighsInt)ids.size(), 
                     ids.data(), vals.data());
    }
}

// Extracts the current equality and inequality systems from a HiGHS model
// and returns them as a ConstrainedPolytope.
//
// Equality rows are identified by equal lower and upper row bounds. Rows whose
// lower bound is -inf are treated as inequalities. Any column bounds of HiGHS
// are ignored.
// @tparam Point the point type
// @param highs the HiGHS instance to read from
// @param tol the numerical tolerance
// @return the polytope extracted from the model
template <typename Point>
ConstrainedPolytope<Point> extract_polytope(Highs const& highs,
                                            double tol = 1e-7)
{
    using NT = typename ConstrainedPolytope<Point>::NT;
    using VT = typename ConstrainedPolytope<Point>::VT;
    using MT = typename ConstrainedPolytope<Point>::MT;
    using Triplet = typename ConstrainedPolytope<Point>::Triplet;

    HighsLp const& lp = highs.getLp();
    unsigned const d = (unsigned)lp.num_col_;
    unsigned const num_rows = (unsigned)lp.num_row_;

    HighsSparseMatrix A = lp.a_matrix_;
    A.ensureRowwise();
    
    std::vector<Triplet> eq_triplets, in_triplets;
    std::vector<double> eq_rhs, in_rhs;
    unsigned m_eq = 0;
    unsigned m_in = 0;

    std::vector<int> ids(d);
    std::vector<double> val(d);
    for (unsigned i = 0; i < num_rows; ++i) {
        double lo = lp.row_lower_[i];
        double hi = lp.row_upper_[i];
        bool is_eq = (!std::isinf(lo) && !std::isinf(hi) && std::abs(hi-lo) < tol);

        HighsInt start = A.start_[i];
        HighsInt end = A.start_[i+1];

        if (is_eq) {
            for (unsigned j = start; j < end; ++j) 
                eq_triplets.emplace_back(m_eq, A.index_[j], NT(A.value_[j]));

            eq_rhs.push_back(hi);
            m_eq++;
        } else {
            for (unsigned j = start; j < end; ++j) 
                in_triplets.emplace_back(m_in, A.index_[j], NT(A.value_[j]));

            in_rhs.push_back(hi);
            m_in++;
        }
    }

    MT A_eq(m_eq, d), A_in(m_in, d);
    A_eq.setFromTriplets(eq_triplets.begin(), eq_triplets.end());
    A_in.setFromTriplets(in_triplets.begin(), in_triplets.end());

    VT b_eq(m_eq), b_in(m_in);
    for (unsigned i = 0; i < m_eq; ++i) b_eq(i) = (NT)eq_rhs[i];
    for (unsigned i = 0; i < m_in; ++i) b_in(i) = (NT)in_rhs[i];

    return ConstrainedPolytope<Point>(d, A_eq, b_eq, A_in, b_in);
}
#endif