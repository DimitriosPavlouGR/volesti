// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#include <limits>
#include <vector>
#include "doctest.h"
#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "preprocess/metabolic/exhaustive_simplification.hpp"
#include "preprocess/metabolic/clarkson_simplification.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef MetabolicPolytope<Point> Polytope;
typedef typename Polytope::MT MT;
typedef typename Polytope::VT VT;
static NT const INF = std::numeric_limits<NT>::infinity();

struct Exhaustive {
    static exhaustive_simplification::Result<Point> run(Polytope const& P, 
                                                        bool fix_dimensions)
    {
        exhaustive_simplification::Config config;
        config.fix_dimensions = fix_dimensions;
        return exhaustive_simplification::simplify(P, config);
    }

    static char const* name() {return "exhaustive";}
};

struct Clarkson {
    static clarkson_simplification::Result<Point> run(Polytope const& P, 
                                                      bool fix_dimensions)
    {
        clarkson_simplification::Config config;
        config.fix_dimensions = fix_dimensions;
        return clarkson_simplification::simplify(P, config);
    }

    static char const* name() {return "clarkson";}
};

template <typename Simplifier>
void test_cube_no_change(unsigned d) 
{   
    INFO("simplifier: " << Simplifier::name());

    Polytope P1 = Polytope::cube(d);
    auto result = Simplifier::run(P1, false);
    Polytope P2 = result.P;

    CHECK(result.bounds_relaxed == 0);
    CHECK(result.dims_fixed == 0);
    CHECK(P2.getEqualities().isApprox(P1.getEqualities()));
    CHECK(P2.getLowerBounds() == P1.getLowerBounds());
    CHECK(P2.getUpperBounds() == P1.getUpperBounds());
    CHECK(P2.getEqualityBounds() == P1.getEqualityBounds());
}

template <typename Simplifier>
void test_cube_relaxed_bounds(unsigned d) 
{
    unsigned m = 2*d;
    VT b_l(m);
    VT b_u(m);
    MT A_eq(d, m);
    VT b_eq = VT::Ones(d);

    std::vector<Polytope::Triplet> triplets;
    for (unsigned i = 0; i < d; ++i) {
        triplets.emplace_back(i, i, NT(1));
        triplets.emplace_back(i, d+i, NT(1));
    }
    A_eq.setFromTriplets(triplets.begin(), triplets.end());
    A_eq.makeCompressed();

    for (unsigned i = 0; i < d; ++i) {
        b_l(i) = -10.0;
        b_u(i) = 10.0;
    }

    for (unsigned i = d; i < m; ++i) {
        b_l(i) = 0.0;
        b_u(i) = 1.0;
    }

    INFO("simplifier: " << Simplifier::name());

    Polytope P1 = Polytope(m, A_eq, b_l, b_u, b_eq);
    auto result = Simplifier::run(P1, false);
    Polytope P2 = result.P;

    // Checks the simplification statistics.
    CHECK(result.bounds_relaxed == 2*d);
    CHECK(result.dims_fixed == 0);

    // Checks that equalities were untouched.
    CHECK(P2.getEqualities().isApprox(P1.getEqualities()));
    CHECK(P2.getEqualityBounds() == P1.getEqualityBounds());

    // Checks that the first d reactions were relaxed.
    CHECK((P2.getLowerBounds().head(d).array() == -INF).all());
    CHECK((P2.getUpperBounds().head(d).array() == INF).all());

    // The other m-d bounds must remain untouched.
    CHECK(P2.getLowerBounds().tail(d) == P1.getLowerBounds().tail(d));
    CHECK(P2.getUpperBounds().tail(d) == P1.getUpperBounds().tail(d));
}

template <typename Simplifier>
void test_simplex_relaxed_bounds(unsigned d) 
{
    INFO("simplifier: " << Simplifier::name());

    Polytope P1 = Polytope::simplex(d);
    auto result = Simplifier::run(P1, false);
    Polytope P2 = result.P;

    // Checks statistics.
    CHECK(result.bounds_relaxed == d);
    CHECK(result.dims_fixed == 0);

    // Checks every upper bound is infinity.
    CHECK((P2.getUpperBounds().array() == INF).all());

    // Checks that the rest are untouched.
    CHECK(P1.getDimension() == P2.getDimension());
    CHECK(P1.getLowerBounds() == P2.getLowerBounds());
    CHECK(P1.getEqualities().isApprox(P2.getEqualities()));
    CHECK(P1.getEqualityBounds() == P2.getEqualityBounds());
}

template <typename Simplifier>
void test_cube_degenerate_dimensions(unsigned d) 
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

    INFO("simplifier: " << Simplifier::name());

    Polytope P1 = Polytope(m, A_eq, b_l, b_u, b_eq);
    auto result = Simplifier::run(P1, true);
    Polytope P2 = result.P;

    // Checks statistics.
    CHECK(result.bounds_relaxed == 2*d);
    CHECK(result.dims_fixed == d);

    // Checks that the pinned constraints were added to A_eq.
    CHECK(P2.getNumEqualities() == d);
    CHECK((unsigned)P2.getEqualityBounds().size() == d);

    // Checks that the last d reactions were relaxed.
    CHECK((P2.getLowerBounds().tail(d).array() == -INF).all());
    CHECK((P2.getUpperBounds().tail(d).array() == INF).all());

    // Checks that the rest of the bounds remain untouched.
    CHECK(P2.getLowerBounds().head(d) == P1.getLowerBounds().head(d));
    CHECK(P2.getUpperBounds().head(d) == P1.getUpperBounds().head(d));
}

TEST_CASE_TEMPLATE("test_no_change", Simplifier, Exhaustive, Clarkson) {
    test_cube_no_change<Simplifier>(10);
}

TEST_CASE_TEMPLATE("test_no_dimension_fixing", Simplifier, Exhaustive, Clarkson) {
    test_simplex_relaxed_bounds<Simplifier>(10);
    test_cube_relaxed_bounds<Simplifier>(10);
}

TEST_CASE_TEMPLATE("test_dimension_fixing", Simplifier, Exhaustive, Clarkson) {
    test_cube_degenerate_dimensions<Simplifier>(10);
}
