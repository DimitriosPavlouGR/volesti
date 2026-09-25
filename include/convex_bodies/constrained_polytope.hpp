// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef CONSTRAINED_POLYTOPE_HPP
#define CONSTRAINED_POLYTOPE_HPP

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>

// This class describes a (generally not full dimensional) polytope defined by equality 
// constraints and general inequality constraints:
//
//      A_eq x = b_eq
//      A_in x <= b_in
//
// @tparam Point Point type
template <typename Point>
class ConstrainedPolytope {
    public:
        typedef Point PointType;
        typedef typename Point::FT NT;
        typedef Eigen::SparseMatrix<NT, Eigen::RowMajor> MT;
        typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;
        typedef Eigen::Triplet<NT> Triplet;

    private:
        unsigned d;  // the ambient dimension
        MT A_eq;     // equality constraint matrix
        VT b_eq;     // equality rhs
        MT A_in;     // inequality matrix
        VT b_in;     // inequality rhs

    public:
        // Default constructor.
        ConstrainedPolytope() = default;

        // Builds a polytope with equalities, general inequalities and box bounds.
        // @param d_ the ambient dimension
        // @param A_eq_ the equality matrix, s.t. A_eq x = b_eq
        // @param b_eq_ the equality rhs
        // @param A_in_ the inequality matrix
        // @param b_in_ the inequality rhs
        ConstrainedPolytope(unsigned d_, 
                          MT const& A_eq_, VT const& b_eq_, 
                          MT const& A_in_,  VT const& b_in_
        ) : 
            d{d_}, A_eq{A_eq_}, b_eq{b_eq_}, A_in{A_in_}, b_in{b_in_}
        {}

        // Default copy constructor, copies all members.
        ConstrainedPolytope(ConstrainedPolytope const&) = default;

        // @return the dimension d
        unsigned getDimension() const { return this->d; }

        // @return the equality matrix A_eq
        MT const& getEqualities() const { return this->A_eq; }

        // @return the equality rhs
        VT const& getEqualityRHS() const { return this->b_eq; }

        // @return the inequality matrix A_in
        MT const& getInequalities() const { return this->A_in; }

        // @return the inequality rhs
        VT const& getInequalityRHS() const { return this->b_in; }

        // @return true if the polytope has equality constraints
        bool hasEqualities() const { return A_eq.rows() > 0; }

        // @return true if the polytope has inequality constraints
        bool hasInequalities() const { return A_in.rows() > 0; }

        // @return the number of equality constraints (rows of A_eq)
        unsigned getNumEqualities() const { return (unsigned)A_eq.rows(); }

        // @return the number of inequality constraints (rows of A_in)
        unsigned getNumInequalities() const { return (unsigned)A_in.rows(); }
};
#endif