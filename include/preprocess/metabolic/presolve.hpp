
// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef METABOLIC_PRESOLVE_HPP
#define METABOLIC_PRESOLVE_HPP

#include "cholmod.h"
#include "SuiteSparseQR.hpp"

#include <vector>
#include "Eigen/Eigen"
#include "Eigen/SparseCore"
#include "cholmod.h"
#include "SuiteSparseQR.hpp"
#include "convex_bodies/metabolic_polytope.hpp"

// Outcome of a nullspace presolve.
struct NullspacePresolveResult {
    std::vector<unsigned> pinned;
    Eigen::MatrixXd N;
    long rank = 0;
};

// Converts an Eigen sparse matrix into a CHOLMOD sparse matrix.
// @tparam SparseMT the Eigen sparse matrix type
// @param Ain the matrix to convert
// @param cc the CHOLMOD workspace
// @return a CHOLMOD sparse matrix
template <typename SparseMT>
cholmod_sparse* eigen_to_cholmod(SparseMT const& Ain, cholmod_common* cc) 
{
    Eigen::SparseMatrix<double, Eigen::ColMajor> Mc = Ain.template cast<double>();
    Mc.makeCompressed();

    cholmod_sparse* Aout = cholmod_l_allocate_sparse(
        (size_t)Mc.rows(), (size_t)Mc.cols(), (size_t)Mc.nonZeros(),
        /*sorted*/ true, /*packed*/ true, /*stype*/ 0, CHOLMOD_REAL, cc);
    
    if (!Aout) throw std::runtime_error("eigen_to_cholmod: cholmod_l_allocate_sparse failed");

    long* Ap = (long*)Aout->p;
    long* Ai = (long*)Aout->i;
    double* Ax = (double*)Aout->x;

    for (Eigen::Index k = 0; k <= Mc.outerSize(); ++k) {
        Ap[k] = Mc.outerIndexPtr()[k];
    }
    for (Eigen::Index k = 0; k < Mc.nonZeros(); ++k) {
        Ai[k] = Mc.innerIndexPtr()[k];
        Ax[k] = Mc.valuePtr()[k];
    } 

    return Aout;
}

// Presolves, removes pinned reactions by solving solving the homogeneous system.
// @tparam SparseMT the sparse matrix type of M
// @param M the equality matrix
// @param tol the rank / zero tolerance
// @return the pinned reactions, and they basis they were read from
template <typename SparseMT>
NullspacePresolveResult solve_homogeneous_presolve_spqr(SparseMT const& M,
                                                        double tol = 1e-7)
{
    cholmod_common cc;
    cholmod_l_start(&cc);

    Eigen::SparseMatrix<double, Eigen::ColMajor> MT = M.transpose().eval().template cast<double>();
    cholmod_sparse* A = eigen_to_cholmod(MT, &cc);

    SuiteSparseQR_factorization<double>* QR = SuiteSparseQR_factorize<double>(SPQR_ORDERING_DEFAULT, tol, A, &cc);

    if (!QR) {
        cholmod_l_free_sparse(&A, &cc);
        cholmod_l_finish(&cc);
        throw std::runtime_error("solve_homogeneous_presolve_spqr: factorization failed");
    }

    long n = (long)M.cols();
    long r = QR->rank;
    long k = n-r;

    NullspacePresolveResult out;
    out.rank = r;
    out.N = Eigen::MatrixXd::Zero(n, k > 0 ? k : 0);

    if (k > 0) {
        cholmod_dense* X = cholmod_l_allocate_dense((size_t)n, (size_t)k, (size_t)n, CHOLMOD_REAL, &cc);

        if (!X) {
            SuiteSparseQR_free<double>(&QR, &cc);
            cholmod_l_free_sparse(&A, &cc);
            cholmod_l_finish(&cc);
            throw std::runtime_error("solve_homogeneous_presolve_spqr: allocate_dense failed");
        }

        std::memset(X->x, 0, (size_t)n*(size_t)k*sizeof(double));
        double* Xx = (double*)X->x;
        for (long c = 0; c < k; ++c) {
            Xx[c*n+(r+c)] = 1.0;
        }

        cholmod_dense* Y = SuiteSparseQR_qmult<double>(SPQR_QX, QR, X, &cc);
        cholmod_l_free_dense(&X, &cc);

        if (!Y) {
            SuiteSparseQR_free<double>(&QR, &cc);
            cholmod_l_free_sparse(&A, &cc);
            cholmod_l_finish(&cc);
            throw std::runtime_error("solve_homogeneous_presolve_spqr: allocate_dense failed");
        }

        std::memcpy(out.N.data(), Y->x, (size_t)n*(size_t)k*sizeof(double));
        cholmod_l_free_dense(&Y, &cc);

        for (long j = 0; j < n; ++j) {
            if (out.N.row(j).cwiseAbs().maxCoeff() < tol)
                out.pinned.push_back((unsigned)j);
        }
    } else {
        for (long j = 0; j < n; ++j)
                out.pinned.push_back((unsigned)j);
    }

    SuiteSparseQR_free<double>(&QR, &cc);
    cholmod_l_free_sparse(&A, &cc);
    cholmod_l_finish(&cc);

    return out;
}

// Presolves, removes pinned reactions by solving solving the homogeneous system.
// @tparam Point the point type of the polytope
// @param P the metabolic polytope
// @param tol the rank / zero tolerance
// @return the pinned reactions, and they basis they were read from
template <typename Point>
NullspacePresolveResult solve_homogeneous_presolve_spqr(MetabolicPolytope<Point> const& P, double tol = 1e-7)
{
    return solve_homogeneous_presolve_spqr(P.getEqualities(), tol);
}

// Returns the indices of a maximal independent of rows of A sorted.
// @tparam MT the matrix type
// @param A the equality matrix
// @param tol the rank / zero tolerance
template <typename MT>
std::vector<unsigned> find_independent_rows_spqr(MT const& A, double tol = 1e-7) {
    cholmod_common cc;
    cholmod_l_start(&cc);

    Eigen::SparseMatrix<double, Eigen::ColMajor> At = A.transpose().eval().template cast<double>();
    cholmod_sparse* B = eigen_to_cholmod(At, &cc);

    cholmod_sparse* R = nullptr;
    int64_t* E = nullptr;
    int64_t r = SuiteSparseQR<double, int64_t>(SPQR_ORDERING_DEFAULT, tol, (int64_t)0, B, &R, &E, &cc);

    if (r < 0) {
        cholmod_l_free_sparse(&B, &cc);
        cholmod_l_finish(&cc);
        throw std::runtime_error("find_independent_rows_spqr: SPQR failed");
    }

    std::vector<unsigned> keep(r);
    for (int64_t i = 0; i < r; ++i) {
        keep[i] = E ? (unsigned)E[i] : (unsigned)i;
    }

    std::sort(keep.begin(), keep.end());

    if (E) {
        cholmod_l_free((size_t)At.cols(), sizeof(int64_t), E, &cc);
    }

    cholmod_l_free_sparse(&R, &cc);
    cholmod_l_free_sparse(&B, &cc);
    cholmod_l_finish(&cc);

    return keep;
}
#endif