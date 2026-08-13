// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef SOLVE_LP_HIGHS_HPP
#define SOLVE_LP_HIGHS_HPP

#include <cmath>
#include <vector>
#include "Highs.h"
#include "configure_highs.hpp"

// Computes the Chebychev ball of an H-Polytope A x <= b, i.e.
// the largest ball that fits it.
//
// The LP solved is: max r s.t. a_i x + || a_i || r <= b_i, r >= 0
// @tparam NT the number type
// @tparam Point the point type
// @tparam MT the matrix type of A
// @tparam VT the vector type of b
// @param A the constraint matrix
// @param b the right hand side
// @return the center and radius of the Chebychev ball, or a radius of -1
// on failiures
template<typename NT, typename Point, typename MT, typename VT>
std::pair<Point, NT> ComputeChebychevBall(const MT& A, const VT& b)
{
    unsigned d = A.cols();
    unsigned m = A.rows();

    Highs highs;
    lp_oracles_configure_highs(highs);

    // Adds the variables, center variables are free and r >= 0.
    for (unsigned i = 0; i < d; ++i) {
        highs.addVar(-kHighsInf, kHighsInf);
    }
    highs.addVar(0.0, kHighsInf);

    // Applies a_i x + ||a_i|| r <= b_i.
    std::vector<HighsInt> indices(d+1);
    std::vector<double> values(d+1);

    for (unsigned i = 0; i <= d; ++i) 
        indices[i] = (HighsInt)i;

    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < d; ++j) {
            values[j] = (double)A.coeff(i,j);
        }

        values[d] = (double)A.row(i).norm();
        highs.addRow(-kHighsInf, (double)b(i), d+1, indices.data(), 
                     values.data());
    }

    // Maximizes the radius in the LP
    highs.changeColCost((HighsInt)d, 1.0);
    highs.changeObjectiveSense(ObjSense::kMaximize);
    highs.run();

    std::pair<Point, NT> exception_pair(Point(1), -1.0);

    // Checks the model status
    if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
        #ifdef VOLESTI_DEBUG
        std::cout << "Could not solve the Linear Program for Chebychev center "
                  << ", highs returned status code " 
                  << (int)highs.getModelStatus()
                  << std::endl;
        #endif

        return exception_pair;
    }

    const auto& sol = highs.getSolution().col_value;
    std::vector<NT> center(d);
    for (unsigned i = 0; i < d; ++i) {
        center[i] = (NT)sol[i];
    }

    Point x(d, center.begin(), center.end());
    NT r = (NT)highs.getObjectiveValue();

    return std::pair<Point, NT>(x, r);
}

// Finds a point in the intersection of the two V-Polytopes, given by the 
// vertex matrices V1, V2.
//
// The point is written as a convex combination of the vertices of each
// polytope, so the LP variables are the n+m combination weights, s.t. 
// V1^T lambda - V2^T mu = 0, with the vectors having a sum of one.
// @tparam VT the vector type
// @tparam MT the matrix type of V1, V2
// @tparam Point the point type
// @param V1 the vertex matrix of the first polytope
// @param V2 the vertex matrix of the second polytope
// @param direction the direction of the point
// @param empty set to true if the interscetion is empty
// @return a point in the intersection, meaningful only when empty is false
template<typename VT, typename MT, typename Point>
Point PointInIntersection(MT V1, MT V2, Point direction, bool &empty) 
{
    typedef typename Point::FT NT;

    unsigned d = V1.cols();
    unsigned n = V1.rows();
    unsigned m = V2.rows();
    unsigned k = n+m;

    Point p(d);

    Highs highs;
    lp_oracles_configure_highs(highs);

    // Forces the combinations to be non-negative.
    for (unsigned i = 0; i < k; ++i)
        highs.addVar(0.0, kHighsInf);

    std::vector<HighsInt> indices(k);
    std::vector<double> values(k);

    for (unsigned i = 0; i < k; ++ i)
        indices[i] = (HighsInt)i;

    // Forces the point to be in the intersection (V1^T lambda - V2^T mu = 0).
    for (unsigned i = 0; i < d; ++i) {
        for (unsigned j = 0; j < n; ++j)
            values[j] = (double)V1(j, i);

        for (unsigned j = 0; j < m; ++j)
            values[n+j] = -(double)V2(j, i);

        highs.addRow(0.0, 0.0, k, indices.data(), values.data());
    }

    // Makes the combinations convex.
    for (unsigned i = 0; i < n; ++i) values[i] = 1.0;
    for (unsigned i = 0; i < m; ++i) values[n+i] = 0.0;
    highs.addRow(1.0, 1.0, k, indices.data(), values.data());

    for (unsigned i = 0; i < n; ++i) values[i] = 0.0;
    for (unsigned i = 0; i < m; ++i) values[n+i] = 1.0;
    highs.addRow(1.0, 1.0, k, indices.data(), values.data());

    // Maximizes the given direction.
    VT const& dir = direction.getCoefficients();
    for (unsigned i = 0; i < k; ++i)
        highs.changeColCost((HighsInt)i, (double)dir(i));

    highs.changeObjectiveSense(ObjSense::kMaximize);
    highs.run();

    if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
        #ifdef VOLESTI_DEBUG
        std::cout << "Could not solve the Linear Program for VPolytope intersection "
                  << ", highs returned status code " 
                  << (int)highs.getModelStatus()
                  << std::endl;
        #endif

        empty = true;
        return p;
    }

    // For the solution we use lambda, mu also works.
    const auto& sol = highs.getSolution().col_value;
    VT lambda(n);
    for (unsigned i = 0; i < n; ++i) 
        lambda(i) = (NT)sol[i];
    
    p = V1.transpose()*lambda;
    empty = false;
    return p;
}
#endif