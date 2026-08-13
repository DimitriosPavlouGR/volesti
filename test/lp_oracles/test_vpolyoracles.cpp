// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#include "doctest.h"
#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "lp_oracles/vpolyoracles.h"
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
    
    std::vector<NT> row(3);
    std::vector<int> colno(3);

    CHECK(memLP_Vpoly(V, Point(2, {0.5, 0.5}), row.data(), colno.data()));    // inside
    CHECK(memLP_Vpoly(V, Point(2, {0.0, 0.0}), row.data(), colno.data()));    // on boundary
    CHECK(!memLP_Vpoly(V, Point(2, {20.0, 20.0}), row.data(), colno.data())); // outside
}

void test_line_intersection_vpoly() {
    MT V(5, 2);
    
    V << 0, 0,
         4, 0,
         5, 2,
         3, 4,
         0, 3;

    std::vector<NT> conv_comb(5);
    std::vector<NT> row(6);
    std::vector<int> colno(6);

    Point p(2, {2.0, 1.0});
    Point v(2, {1.0, 0.0});

    // Single point tests.
    NT max_l = intersect_line_Vpoly<NT>(V, p, v, conv_comb.data(),
                                        row.data(), colno.data(),
                                        true, false);
    
    NT min_l = intersect_line_Vpoly<NT>(V, p, v, conv_comb.data(),
                                    row.data(), colno.data(),
                                    false, false);

    CHECK(max_l == doctest::Approx(-2.0));
    CHECK(min_l == doctest::Approx(2.5));

    // Double point test.
    std::tie(min_l, max_l) = intersect_double_line_Vpoly<NT>(V, p, v,
                                        row.data(), colno.data());
    
    CHECK(max_l == doctest::Approx(-2.0));
    CHECK(min_l == doctest::Approx(2.5));
}

void test_line_intersection_zpoly() {
    MT Z(3, 2);
    
    Z << 1, 0,
         0, 1,
         1, 1;

    std::vector<NT> conv_comb(3);
    std::vector<NT> row(4);
    std::vector<int> colno(4);

    Point p(2, {0.0, 0.0});
    Point v(2, {1.0, 0.0});

    // Single point tests.
    NT max_l = intersect_line_Vpoly<NT>(Z, p, v, conv_comb.data(),
                                        row.data(), colno.data(),
                                        true, true);
    
    NT min_l = intersect_line_Vpoly<NT>(Z, p, v, conv_comb.data(),
                                    row.data(), colno.data(),
                                    false, true);

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
