// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"
#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "lp_oracles/vpolyoracles.hpp"
#include <vector>
#include <tuple>

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;

void test_membership() {
    MT V(5, 2);
    
    V << 0, 0,
         4, 0,
         5, 2,
         3, 4,
         0, 3;

    CHECK(std::get<0>(memLP_Vpoly(V, Point(2, {0.0, 0.0}))));
    CHECK(std::get<0>(memLP_Vpoly(V, Point(2, {1.0, 0.5}))));
    CHECK(!std::get<0>(memLP_Vpoly(V, Point(2, {0.0, 5.0}))));
    CHECK(!std::get<0>(memLP_Vpoly(V, Point(2, {-1.0, -1.0}))));
}

void test_line_intersection_vpoly() {
    MT V(5, 2);
    
    V << 0, 0,
         4, 0,
         5, 2,
         3, 4,
         0, 3;

    std::vector<NT> conv_comb(5);
    Point p(2, {2.0, 1.0});
    Point v(2, {1.0, 0.0});

    // Single point tests.
    auto [max_l, ok1] = intersect_line_Vpoly<NT>(V, p, v, conv_comb.data(), true, false);
    auto [min_l, ok2] = intersect_line_Vpoly<NT>(V, p, v, conv_comb.data(), false, false);
    
    CHECK(ok1);
    CHECK(ok2);
    CHECK(max_l == doctest::Approx(-2.0));
    CHECK(min_l == doctest::Approx(2.5));

    // Two points tests.
    auto [min_l2, max_l2, ok3] = intersect_double_line_Vpoly<NT>(V, p, v);

    CHECK(ok3);
    CHECK(min_l2 == doctest::Approx(2.5));
    CHECK(max_l2 == doctest::Approx(-2.0));
}

void test_line_intersection_zpoly() {
    MT Z(3, 2);
    
    Z << 1, 0,
         0, 1,
         1, 1;

    std::vector<NT> conv_comb(3);

    Point p(2, {0.0, 0.0});
    Point v(2, {1.0, 0.0});

    auto [max_l, ok1] = intersect_line_Vpoly<NT>(Z, p, v, conv_comb.data(), true, true);
    auto [min_l, ok2] = intersect_line_Vpoly<NT>(Z, p, v, conv_comb.data(), false, true);

    CHECK(ok1);
    CHECK(ok2);
    CHECK(max_l == doctest::Approx(-2.0));
    CHECK(min_l == doctest::Approx(2.0));
}

TEST_CASE("test_membership") {
    test_membership();
}

TEST_CASE("line_intersection") {
    test_line_intersection_vpoly();
    test_line_intersection_zpoly();
}
   