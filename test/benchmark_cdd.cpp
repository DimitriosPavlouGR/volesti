// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cmath>
#include <ctime>
#include <limits>
#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "preprocess/constrained_polytope/bigg_parser.hpp"
#include "preprocess/constrained_polytope/simplifier.hpp"
#include "preprocess/constrained_polytope/transformation.hpp"
#include "preprocess/constrained_polytope/simplification_exporter.hpp"

extern "C" {
#include <cddlib/setoper.h>
#include <cddlib/cdd.h>
}

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef ConstrainedPolytope<Point> Polytope;

// The result of a cdd run.
struct CddResult {
    // The simplified polytope.
    Polytope polytope;

    // True if cdd finished without error.
    bool valid = false;

    // The CPU time of the canonicalization in seconds.
    double time = 0.0;

    // The number of implicit equalities found.
    unsigned implicit_rows = 0;

    // The number of redundant rows removed.
    unsigned redundant_rows = 0;
};

// Writes row i of the cdd matrix as b - a x >= 0, stored as [b | -a].
// @param M the cdd matrix
// @param i the row of M
// @param A the sparse matrix to copy from
// @param r the row of A
// @param b the right hand side
template <typename MT>
void set_sparse_row(dd_MatrixPtr M, long i, MT const& A, Eigen::Index r, double b)
{
    dd_set_d(M->matrix[i][0], b);
    for (typename MT::InnerIterator it(A, r); it; ++it)
        dd_set_d(M->matrix[i][it.col()+1], -(double)it.value());
}

// Finds the implicit equalities of P and removes its redundant rows with cdd.
// Requires dd_set_global_constants() to have been called.
// @param P the polytope
// @return the result
CddResult cdd_canonicalize(Polytope const& P)
{
    using MT = typename Polytope::MT;
    using VT = typename Polytope::VT;
    using Triplet = typename Polytope::Triplet;

    CddResult res;

    unsigned const d = P.getDimension();
    MT const& A_eq = P.getEqualities();
    MT const& A_in = P.getInequalities();
    VT const& b_eq = P.getEqualityRHS();
    VT const& b_in = P.getInequalityRHS();
    VT const& lb = P.getLowerBounds();
    VT const& ub = P.getUpperBounds();

    long const m_eq = A_eq.rows();
    long const m_in = A_in.rows();
    long const m = m_eq+m_in+P.getNumFiniteBounds();

    dd_MatrixPtr M = dd_CreateMatrix(m, d+1);
    M->representation = dd_Inequality;
    M->numbtype = dd_Real;

    long i = 0;

    // Equalities, marked as linearity rows (1-based)
    for (Eigen::Index r = 0; r < m_eq; ++r, ++i) {
        set_sparse_row(M, i, A_eq, r, (double)b_eq(r));
        set_addelem(M->linset, i+1);
    }

    // Inequalities a x <= b
    for (Eigen::Index r = 0; r < m_in; ++r, ++i)
        set_sparse_row(M, i, A_in, r, (double)b_in(r));

    // Bounds: x_j <= u_j -> u_j - x_j >= 0, x_j >= l_j -> -l_j + x_j >= 0
    long const first_bound = i;
    std::vector<std::pair<unsigned, bool>> bounds;

    for (unsigned j = 0; j < d; ++j) {
        if (std::isfinite((double)ub(j))) {
            dd_set_d(M->matrix[i][0], (double)ub(j));
            dd_set_d(M->matrix[i][j+1], -1.0);
            bounds.emplace_back(j, true);
            ++i;
        }
        if (std::isfinite((double)lb(j))) {
            dd_set_d(M->matrix[i][0], -(double)lb(j));
            dd_set_d(M->matrix[i][j+1], 1.0);
            bounds.emplace_back(j, false);
            ++i;
        }
    }

    dd_rowset impl_lin = nullptr;
    dd_rowset redset = nullptr;
    dd_rowindex newpos = nullptr;
    dd_ErrorType err = dd_NoError;

    std::clock_t c0 = std::clock();
    dd_boolean ok = dd_MatrixCanonicalize(&M, &impl_lin, &redset, &newpos, &err);
    res.time = double(std::clock()-c0)/CLOCKS_PER_SEC;

    if (ok && err == dd_NoError) {
        res.valid = true;
        res.implicit_rows = (unsigned)set_card(impl_lin);
        res.redundant_rows = (unsigned)set_card(redset);

        // origin[p] is the original row at the new position p (1-based), -1 if none
        long const rows = M->rowsize;
        std::vector<long> origin(rows+1, -1);
        for (long k = 0; k < m; ++k) {
            long const p = newpos[k+1];
            if (p > 0) origin[p] = k;
        }

        std::vector<Triplet> eq_triplets, in_triplets;
        std::vector<NT> eq_rhs, in_rhs;

        NT const INF = std::numeric_limits<NT>::infinity();
        VT lb_new = VT::Constant(d, -INF);
        VT ub_new = VT::Constant(d, INF);

        for (long p = 1; p <= rows; ++p) {
            // Equalities come from cdd's output, stored as b - a x = 0
            if (set_member(p, M->linset)) {
                long const r = (long)eq_rhs.size();
                for (unsigned j = 0; j < d; ++j) {
                    double const v = -dd_get_d(M->matrix[p-1][j+1]);
                    if (v != 0.0) eq_triplets.emplace_back((int)r, (int)j, v);
                }
                eq_rhs.push_back(dd_get_d(M->matrix[p-1][0]));
                continue;
            }

            // Inequalities are copied from the original rows they come from
            long const k = origin[p];

            if (k >= first_bound) {
                auto [j, is_upper] = bounds[k-first_bound];
                if (is_upper) ub_new(j) = ub(j);
                else lb_new(j) = lb(j);
            } else if (k >= m_eq) {
                long const r = (long)in_rhs.size();
                for (typename MT::InnerIterator it(A_in, k-m_eq); it; ++it)
                    in_triplets.emplace_back((int)r, (int)it.col(), it.value());
                in_rhs.push_back(b_in(k-m_eq));
            }
        }

        MT A_eq_new((Eigen::Index)eq_rhs.size(), d);
        A_eq_new.setFromTriplets(eq_triplets.begin(), eq_triplets.end());
        VT b_eq_new = Eigen::Map<VT const>(eq_rhs.data(), (Eigen::Index)eq_rhs.size());

        MT A_in_new((Eigen::Index)in_rhs.size(), d);
        A_in_new.setFromTriplets(in_triplets.begin(), in_triplets.end());
        VT b_in_new = Eigen::Map<VT const>(in_rhs.data(), (Eigen::Index)in_rhs.size());

        res.polytope = Polytope(d, A_eq_new, b_eq_new, A_in_new, b_in_new, lb_new, ub_new);
    }

    if (impl_lin) set_free(impl_lin);
    if (redset) set_free(redset);
    if (newpos) free(newpos);
    dd_FreeMatrix(M);
    return res;
}

void benchmark(std::string const& model_path, std::filesystem::path const& export_dir) {
    Polytope P = parse_from_json<Point>(model_path);

    std::string name = std::filesystem::path(model_path).stem().string();

    std::cout << "[" << name << "]"
              << "  n=" << P.getDimension()
              << "  m_eq=" << P.getNumEqualities()
              << "  m_in=" << P.getNumInequalities()
              << "  finite bounds=" << P.getNumFiniteBounds() << std::endl;

    dd_set_global_constants();
    CddResult cres = cdd_canonicalize(P);
    dd_free_global_constants();

    if (!cres.valid) {
        std::cout << "failed, " << cres.time << std::endl;
        return;
    }

    // Wraps the cdd run as a simplification result so the exporter can be reused
    SimplifierResult<Point> res;
    res.polytope = cres.polytope;
    res.status = SimplifierStatus::OK;
    res.times.clarkson = cres.time;
    res.reduce.implicit_rows = cres.implicit_rows;

    Polytope const& Ps = res.polytope;
    std::cout << "ok"
              << "  -> m_eq=" << Ps.getNumEqualities()
              << "  m_in=" << Ps.getNumInequalities()
              << "  finite bounds=" << Ps.getNumFiniteBounds()
              << "  removed=" << (P.getNumFiniteBounds()-Ps.getNumFiniteBounds())
              << "  implicit=" << cres.implicit_rows
              << "  redundant=" << cres.redundant_rows
              << "  " << res.times.total() << std::endl;

    // Grabs the full dimensional polytope
    std::clock_t c0 = std::clock();
    auto [HP, shift, N] = transform(Ps);
    std::clock_t c1 = std::clock();
    double trans_t = double(c1-c0)/CLOCKS_PER_SEC;

    // Same format as the Clarkson export, without an LP count since cdd does not report one
    nlohmann::json jsn = construct_json(name, P, res, HP, N, shift, trans_t);
    jsn["report"].erase("solved_lps");
    jsn["report"]["cdd"] = {
        {"implicit_rows", cres.implicit_rows},
        {"redundant_rows", cres.redundant_rows}
    };

    std::filesystem::path export_path = export_dir/(name+"_cdd.json");
    std::ofstream f(export_path);
    f << jsn.dump();

    std::cout << "  exported to " << export_path.string() << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: benchmark_cdd <model.json>" << std::endl;
        return 1;
    }

    std::filesystem::path const export_dir = "simplified_cdd";
    std::filesystem::create_directories(export_dir);

    benchmark(argv[1], export_dir);
    return 0;
}