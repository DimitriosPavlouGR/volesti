// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"
#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "preprocess/metabolic/metabolic_simplification.hpp"
#include "preprocess/metabolic/metabolic_transformation.hpp"
#include "random_walks/random_walks.hpp"
#include "volume/volume_cooling_balls.hpp"
#include "generators/boost_random_number_generator.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef MetabolicPolytope<Point> Polytope;
typedef typename Polytope::MT MT;
typedef typename Polytope::VT VT;
typedef BoostRandomNumberGenerator<boost::mt19937, double> RNG;

NT compute_median_volume(HPolytope<Point> & HP,
                         double e,
                         unsigned walk_len,
                         unsigned num_trials) 
{
    std::vector<double> volumes;
    for (unsigned i = 0; i < num_trials; ++i) {
        auto v = volume_cooling_balls<BallWalk, RNG, HPolytope<Point>>(HP, e, walk_len);
        volumes.push_back(v.second);
    }             

    std::stable_sort(volumes.begin(), volumes.end());
    return volumes[volumes.size()/2];
}

void test_cube_transformation(unsigned d) 
{
    unsigned m = 2*d;
    VT b_l(m);
    VT b_u(m);
    MT A_eq(0, m);
    VT b_eq(0);

    for (unsigned i = 0; i < d; ++i) {
        b_l(i) = 0.0;
        b_u(i) = 1.0;
    }

    for (unsigned i = d; i < m; ++i) {
        b_l(i) = 1.0;
        b_u(i) = 1.0+1e-12;
    }

    Polytope P1 = Polytope(m, A_eq, b_l, b_u, b_eq);
    simplification::Config config;
    config.fix_dimensions = true;
    auto res = simplification::simplify(P1, config);
    Polytope P2 = res.P;

    auto trs_res = transform(P2);
    HPolytope<Point> HP = std::get<0>(trs_res);

    unsigned walk_len = 10+d/10;
    double volume = compute_median_volume(HP, 0.05, walk_len, 10);

    CHECK(HP.dimension() == d);
    CHECK(volume > 0.90);
    CHECK(volume < 1.10);
}

void test_simplex_transformation(unsigned d) 
{
    Polytope P1 = Polytope::simplex(d);
    auto result = simplification::simplify(P1);
    Polytope P2 = result.P;

    auto trs_res = transform(P2);
    HPolytope<Point> HP = std::get<0>(trs_res);

    unsigned walk_len = 10+d/10;
    double actual_v = std::sqrt(d)/std::tgamma(d);
    double volume = compute_median_volume(HP, 0.05, walk_len, 10);

    CHECK(HP.dimension() == d-1);
    CHECK(volume > 0.90*actual_v);
    CHECK (volume < 1.10*actual_v);
}

TEST_CASE("test_cube_transformation") {
    test_cube_transformation(10);
}

TEST_CASE("test_simplex_transformation") {
    test_simplex_transformation(10);
}