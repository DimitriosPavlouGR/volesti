// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef VPOLYORACLES_HIGHS_HPP
#define VPOLYORACLES_HIGHS_HPP

#include <vector>
#include <cmath>
#include "Highs.h"
#include "configure_highs.hpp"

// Decides whether q belongs in the V-Polytope given by the vertex matrix V.
//
// The LP looks for a hyperplane seperating q from the vertices, so it
// maximizes q k - l s.t. V k <= l. Such a hyperplane exists when the optimum
// is positive, and q is inside the hull when it is not.
// @tparam MT the matrix type of V
// @tparam Point the point type
// @tparam NT the number type
// @param V the vertex matrix, one vertex per row
// @param q the point to test
// @param row unused, kept for legacy support
// @param colno unused, kept for legacy support
// @return true when q belongs to V
template <typename MT, typename Point, typename NT>
bool memLP_Vpoly(const MT& V, const Point& q, NT* row, int* colno)
{
    unsigned d = q.dimension();
    unsigned m = V.rows();

    Highs highs;
    lp_oracles_configure_highs(highs);

    for (unsigned i = 0; i <= d; ++i)
        highs.addVar(-kHighsInf, kHighsInf);

    // Adds the variables for k and l. 
    std::vector<HighsInt> indices(d+1);
    std::vector<double> values(d+1);
    for (unsigned i = 0; i <= d; ++i)
        indices[i] = (HighsInt)i;

    // Forces V k <= l.
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < d; ++j) {
            values[j] = (double)V(i, j);
        }
        values[d] = -1.0;
        highs.addRow(-kHighsInf, 0.0, d+1, indices.data(), values.data());
    }

    // Bounds the objective so the LP stays bounded.
    for (unsigned i = 0; i < d; ++i) {
        values[i] = (double)q[i];
    }
    values[d] = -1.0;
    highs.addRow(-kHighsInf, 1.0, d+1, indices.data(), values.data());

    // Maximizes q k - l
    for (unsigned i = 0; i < d; ++i) 
        highs.changeColCost((HighsInt)i, (double)q[i]);

    highs.changeColCost((HighsInt)d, -1.0);
    highs.changeObjectiveSense(ObjSense::kMaximize);
    highs.run();
    
    if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
        #ifdef VOLESTI_DEBUG
            std::cout << "Could not solve the Linear Program for membership"
                      << ", highs returned code "
                      << (int)highs.getModelStatus()
                      << std::endl;
        #endif
        return false;
    }

    return highs.getObjectiveValue() <= 0.0;
}

// Computes the intersection of the ray p + l v with a V-Polytope, or with
// a zonotope when zonotope is true.
//
// The point on the ray is written as a combination of the vertices, so the LP
// has the m combination weights and l as variables, with the d rows V^T x + l v = p. 
// Additionally Z-Polytope and V-Polytope conditions are applied appropriately.
// @tparam NT the number type
// @tparam MT the matrix type of V
// @tparam Point the point type
// @param V the vertex matrix, one vertex per row
// @param p the line origin
// @param v the line direction
// @param conv_comb set to the combination of weight of the intersection point
// @param row unused, kept for legacy support
// @param colno unused, kept for legacy support
// @param maxi if true the largest lambda is computed, otherwise the smallest
// @param zonotope true when V describes a zonotope
// @return the value of l, or -1 if the lp failed
template <typename NT, typename MT, typename Point>
NT intersect_line_Vpoly(MT const& V, Point const& p, Point const& v,
                        NT *conv_comb, NT *row, int *colno,
                        bool maxi, bool zonotope)
{
    unsigned d = v.dimension();
    unsigned m = V.rows();
    unsigned k = m+1;

    Highs highs;
    lp_oracles_configure_highs(highs);

    // Adds the variables of the lp
    for (unsigned i = 0; i < m; ++i) {
        highs.addVar(zonotope ? -1.0 : 0.0, 1.0);
    }
    highs.addVar(-kHighsInf, kHighsInf); // Lambda can be unbounded

    std::vector<HighsInt> indices(k);
    std::vector<double> values(k);
    for (unsigned i = 0; i < k; ++i)
        indices[i] = (HighsInt)i;

    // Ray membership rules (V^T x +l v = p).
    for (unsigned i = 0; i < d; ++i) {
        for (unsigned j = 0; j < m; ++j) {
            values[j] = (double)V(j, i);
        }
        values[m] = (double)v[i];
        highs.addRow((double)p[i], (double)p[i], k, indices.data(), values.data());
    }

    // V-Polytope rules.
    if (!zonotope) {
        for (unsigned i = 0; i < m; ++i) {
            values[i] = 1.0;
        }
        values[m] = 0.0;
        highs.addRow(1.0, 1.0, k, indices.data(), values.data());
    }

    highs.changeColCost((HighsInt)m, 1.0);
    highs.changeObjectiveSense(maxi ? ObjSense::kMaximize : ObjSense::kMinimize);
    highs.run();

    if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
        #ifdef VOLESTI_DEBUG
            std::cout << "Could not solve the Linear Program for V-polytope line intersection"
                      << ", highs returned status code "
                      << (int)highs.getModelStatus()
                      << std::endl;
        #endif
        return NT(-1.0);
    }

    const auto& sol = highs.getSolution().col_value;
    for (unsigned i = 0; i < m; ++i) {
        conv_comb[i] = (NT)sol[i];
    }

    return NT(-highs.getObjectiveValue());
}

// Computes both intersections of the line p + l v with a V-Polytope.
//
// The function makes two calls to intersect_line_Vpoly to cmpute the two points.
// @tparam NT the number type
// @tparam MT the matrix type of V
// @tparam Point the point type
// @param V the vertex matrix, one vertex per row
// @param p the line origin
// @param v the line direction
// @param row unused, kept for legacy support
// @param colno unused, kept for legacy support
// @return the two values of l
template <typename NT, typename MT, typename Point>
std::pair<NT, NT> intersect_double_line_Vpoly(MT const& V, Point const& p,
                                              Point const& v, NT* row, int* colno)
{
    std::vector<NT> conv_comb(V.rows());
    NT l1 = intersect_line_Vpoly<NT>(V, p, v, conv_comb.data(), row, colno, false, false);
    NT l2 = intersect_line_Vpoly<NT>(V, p, v, conv_comb.data(), row, colno, true, false);
    return {l1, l2};
}
#endif