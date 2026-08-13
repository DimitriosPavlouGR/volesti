// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef ZPOLYORACLES_HIGHS_HPP
#define ZPOLYORACLES_HIGHS_HPP

#include <vector>
#include <cmath>
#include "Highs.h"
#include "configure_highs.hpp"
#include "lp_oracles/vpolyoracles_highs.hpp"

// Decides whether q belongs in the Z-Polytope given by the generator matrix V.
//
// A point of the zonotope is a signed combination of the rows of V,
// so the LP asks whether V^T l = q has a solution for every weight in
// [-1, 1]. Any feasible solution satisfies the inequalities, so the objective
// is set to zero.
// @tparam MT the matrix type of V 
// @tparam Point the point type
// @tparam NT the number type
// @param V the generator matrix
// @param q the point to test
// @param row unused, kept for legacy support
// @param colno unused, kept for legacy support
// @return true when q belongs to V
template <typename MT, typename Point, typename NT>
bool memLP_Zonotope(const MT& V, const Point& q, NT* row, int* colno)
{
    unsigned d = q.dimension();
    unsigned m = V.rows();

    Highs highs;
    lp_oracles_configure_highs(highs);

    for (unsigned i = 0; i < m; ++i)
        highs.addVar(-1.0, 1.0);

    // Adds the variable weights.
    std::vector<HighsInt> indices(m);
    std::vector<double> values(m);

    for (unsigned i = 0; i < m; ++i)
        indices[i] = (HighsInt)i;

    // Forces the combination to give q.
    for (unsigned i = 0; i < d; ++i) {
        for (unsigned j = 0; j < m; ++j) {
            values[j] = (double)V(j, i);
        }
        highs.addRow((double)q[i], (double)q[i], m, indices.data(), values.data());
    }

    highs.run();
    
    if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
        #ifdef VOLESTI_DEBUG
            std::cout << "Could not solve the Linear Program for zonotope membership"
                      << ", highs returned code "
                      << (int)highs.getModelStatus()
                      << std::endl;
        #endif
        return false;
    }

    return true;
}

// Computes the intersection of the ray p + l v with a Z-Polytope.
//
// The function makes two calls to intersect_line_Vpoly in its zonotope mode,
// which builds the same model.
// @tparam NT the number type
// @tparam MT the matrix type of V
// @tparam Point the point type
// @param V the generator matrix
// @param p the line origin
// @param v the line direction
// @param row unused, kept for legacy support
// @param colno unused, kept for legacy support
// @return the two values of l
template <typename NT, typename MT, typename Point>
std::pair<NT, NT> intersect_line_zono(MT const& V, Point const& p, Point const& v,
                                      NT *row, int *colno)
{
    std::vector<NT> conv_comb(V.rows());
    NT l1 = intersect_line_Vpoly<NT>(V, p, v, conv_comb.data(), row, colno, false, true);
    NT l2 = intersect_line_Vpoly<NT>(V, p, v, conv_comb.data(), row, colno, true, true);
    return {l1, l2};
}
#endif