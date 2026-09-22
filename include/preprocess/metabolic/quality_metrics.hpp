// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef METABOLIC_QUALITY_METRICS_HPP
#define METABOLIC_QUALITY_METRICS_HPP

#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/metabolic_polytope.hpp"
#include "preprocess/max_inscribed_ellipsoid.hpp"
#include "convex_bodies/hpolytope.h"
#include "preprocess/metabolic/transformation.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;

NT measure_skinniness(MetabolicPolytope<Point> const& P) {
    typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> DenseVT;
    typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> DenseMT;

    auto transformed = transform(P);
    HPolytope<Point> HP = std::get<0>(transformed);

    HP.normalize();
    DenseMT A_in = HP.get_mat();
    DenseVT b_in = HP.get_vec();

    auto ball = HP.ComputeInnerBall();
    DenseVT x0 = ball.first.getCoefficients();
    std::cerr << "inner_ball_radius=" << ball.second << '\n';
    DenseVT slack = b_in-A_in*x0;
    std::cerr << "rows=" << A_in.rows() << " cols=" << A_in.cols()
              << " min_slack=" << slack.minCoeff() << '\n';
    if (slack.minCoeff() <= NT(0)) {
        std::cerr << "x0 is not strictly interior\n";
        return NT(-1);
    }

    JohnEllipsoidParams<NT> params;
    params.maxiter = 2000;
    std::cerr << "maxiter=" << params.maxiter
              << " tol=" << params.tol << " reg=" << params.reg << '\n';

    DenseMT E;
    DenseVT center;
    bool converged = false;
    std::tie(E, center, converged) =
        max_inscribed_ellipsoid<DenseMT, DenseMT, DenseVT, NT>(A_in, b_in, x0, params);

    if (!converged) {
        std::cerr << "solver did not converge\n";
        return NT(-1);
    }

    E = (E+E.transpose())/NT(2);
    Eigen::SelfAdjointEigenSolver<DenseMT> es(E);
    if (es.info() != Eigen::Success) {
        std::cerr << "eigensolver failed\n";
        return NT(-1);
    }

    NT lam_min = es.eigenvalues()(0);
    NT lam_max = es.eigenvalues()(es.eigenvalues().size()-1);
    std::cerr << "lam_min=" << lam_min << " lam_max=" << lam_max << '\n';
    if (lam_min <= NT(0)) return NT(-1);

    return std::sqrt(lam_max/lam_min);
}

#endif
