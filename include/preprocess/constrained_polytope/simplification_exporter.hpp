// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef CONSTRAINED_POLYTOPE_SIMPLIFICATION_EXPORTER_HPP
#define CONSTRAINED_POLYTOPE_SIMPLIFICATION_EXPORTER_HPP

#include <nlohmann/json.hpp>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <Eigen/Eigen>
#include <cmath>
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/constrained_polytope.hpp"
#include "preprocess/constrained_polytope/simplifier.hpp"

namespace constrained_polytope_simplification_exporter_utils {
    // Converts a dense matrix into an array for row arrays, so that it can be represented in .json
    // @tparam MT the eigen expression type
    // @param M the matrix
    // @return the JSON array of rows
    template <typename ET>
    nlohmann::json matrix_to_json(Eigen::MatrixBase<ET> const& M)
    {
        nlohmann::json rows = nlohmann::json::array();
        for (Eigen::Index i = 0; i < M.rows(); ++i) {
            nlohmann::json row = nlohmann::json::array();
            for (Eigen::Index j = 0; j < M.cols(); ++j) {
                row.push_back((double)M(i, j));
            }
            rows.push_back(std::move(row));
        }
        return rows;
    }

    // Converts a vector into a json array.
    // @tparam ET the eigen expression type
    // @param v the vector
    // @return the JSON array
    template <typename ET>
    nlohmann::json vector_to_json(Eigen::MatrixBase<ET> const& v)
    {
        nlohmann::json vec = nlohmann::json::array();
        for (Eigen::Index i = 0; i < v.size(); ++i)
            vec.push_back((double)v(i));
        return vec;
    }

    // Converts a bound vector into a json array (place null instead of inf where inf exists)
    // @tparam ET the eigen expression type
    // @param v the bound vector
    // @return the JSON array
    template <typename ET>
    nlohmann::json bounds_to_json(Eigen::MatrixBase<ET> const& v)
    {
        nlohmann::json vec = nlohmann::json::array();
        for (Eigen::Index i = 0; i < v.size(); ++i) {
            double x = (double)v(i);
            if (std::isfinite(x)) {
                vec.push_back(x);
            } else {
                vec.push_back(nullptr);
            }
        }
        return vec;
    }

    // Converts a sparse matrix into its triplet presentation.
    // @tparam ET the eigen expression type
    // @tparam M the sparse matrix
    // @return the JSON object holding the matrix
    template <typename ET>
    nlohmann::json sparse_matrix_to_json(Eigen::SparseMatrixBase<ET> const& M)
    {
        ET const& Md = M.derived();

        nlohmann::json triplets = nlohmann::json::array();
        for (Eigen::Index k = 0; k < Md.outerSize(); ++k) {
            for (typename ET::InnerIterator it(Md, k); it; ++it) {
                nlohmann::json triplet = nlohmann::json::array();
                triplet.push_back((long)it.row());
                triplet.push_back((long)it.col());
                triplet.push_back((double)it.value());
                triplets.push_back(std::move(triplet));
            }
        }

        nlohmann::json jsn;
        jsn["row_count"] = (long)Md.rows();
        jsn["col_count"] = (long)Md.cols();
        jsn["triplets"]  = std::move(triplets);

        return jsn;
    }

    // Returns the simplification status as a string.
    // @param the status
    // @return the name
    inline std::string status_to_string(SimplifierStatus s) {
        switch (s) {
            case SimplifierStatus::OK: return "OK";
            case SimplifierStatus::PRESOLVE_FAILED: return "PRESOLVE_FAILED";
            case SimplifierStatus::REMOVE_DEPENDENT_ROWS_FAILED: return "REMOVE_DEPENDENT_ROWS_FAILED";
            case SimplifierStatus::INFEASIBLE: return "INFEASIBLE";
            case SimplifierStatus::NO_INTERIOR_POINT: return "NO_INTERIOR_POINT";
            case SimplifierStatus::CLARKSON_FAILED: return "CLARKSON_FAILED";
        }
        return "UNKNOWN";
    }
};

// Constructs the JSON of a ConstrainedPolytope:
//
//      A_eq x = b_eq, A_in x <= b_in, lb <= x <= ub
//
// @tparam Point the point type
// @param P the polytope
// @return the JSON object holding the polytope
template <typename Point>
nlohmann::json construct_polytope_json(ConstrainedPolytope<Point> const& P)
{
    using namespace constrained_polytope_simplification_exporter_utils;

    nlohmann::json jsn;
    jsn["dimension"] = (long)P.getDimension();
    jsn["equality_count"] = (long)P.getNumEqualities();
    jsn["inequality_count"] = (long)P.getNumInequalities();
    jsn["A_eq"] = sparse_matrix_to_json(P.getEqualities());
    jsn["b_eq"] = vector_to_json(P.getEqualityRHS());
    jsn["A_in"] = sparse_matrix_to_json(P.getInequalities());
    jsn["b_in"] = vector_to_json(P.getInequalityRHS());
    jsn["lb"] = bounds_to_json(P.getLowerBounds());
    jsn["ub"] = bounds_to_json(P.getUpperBounds());
    return jsn;
}

// Builds the JSON representation of a full-dimensional polytope together with the data needed
// to map it back to the original coordinates.
// A     - the inequality matrix
// b     - the RHS of Ax <= b
// N     - a basis of the nullspace of A_eq
// shift - a particular solution of the equality system
// @tparam Point the point type
// @tparam DenseMT the dense matrix type holding the nullspace basis
// @tparam VT the vector type of the shift vector
// @param HP the full dimensional polytope
// @param shift a particular solution of the equality system
// @param N a basis of the nullspace of the equality system
// @return the JSON object holding the polytope
template <typename Point, typename DenseMT, typename VT>
nlohmann::json construct_transformed_json(HPolytope<Point> const& HP,
                                          DenseMT const& N,
                                          VT const& shift)
{   
    using namespace constrained_polytope_simplification_exporter_utils;
    
    auto const& A = HP.get_mat();
    auto const& b = HP.get_vec();

    nlohmann::json transformed_jsn;
    transformed_jsn["A"] = matrix_to_json(A);
    transformed_jsn["b"] = vector_to_json(b);
    transformed_jsn["N"] = matrix_to_json(N);
    transformed_jsn["shift"] = vector_to_json(shift);

    return transformed_jsn;
}

// Constructs the .json of the simplification run
// @tparam Point the point type
// @tparam DenseMT the dense matrix type of the nullspace basis
// @tparam VT the vector type of the shift vector
// @param name the name of the model
// @param P the original polytope
// @param res the simplification result
// @param HP the full dimensional polytope
// @param trans_t the transformation time in seconds
// @return the JSON object
template <typename Point, typename DenseMT, typename VT>
nlohmann::json construct_json(std::string const& name,
                              ConstrainedPolytope<Point> const& P,
                              SimplifierResult<Point> const& res,
                              HPolytope<Point> const& HP,
                              DenseMT const& N,
                              VT const& shift,
                              double trans_t)
{   
    using namespace constrained_polytope_simplification_exporter_utils;

    nlohmann::json jsn;
    jsn["name"] = name;
    jsn["original"] = construct_polytope_json(P);
    jsn["simplified"] = construct_polytope_json(res.polytope);
    jsn["transformed"] = construct_transformed_json(HP, N, shift);
    jsn["report"] = {
        {"status", status_to_string(res.status)},
        {"solved_lps", res.solved_lps},
        {"finite_bounds_before", P.getNumFiniteBounds()},
        {"finite_bounds_after", res.polytope.getNumFiniteBounds()},
        {"presolve", {
            {"pinned_vars", res.presolve.pinned_vars},
            {"rank", res.presolve.rank}
        }},
        {"reduce", {
            {"passes", res.reduce.passes},
            {"fixed_vars", res.reduce.fixed_vars},
            {"solved_lps", res.reduce.solved_lps}
        }},
        {"dependent_rows", {
            {"rank", res.dependent_rows.rank},
            {"removed_rows", res.dependent_rows.removed_rows},
            {"max_residual", res.dependent_rows.max_residual}
        }},
        {"stage_times", {
            {"presolve", res.times.presolve},
            {"reduce", res.times.reduce},
            {"dependent_rows", res.times.dependent_rows},
            {"interior_point", res.times.interior_point},
            {"clarkson", res.times.clarkson},
            {"simplification_total", res.times.total()},
            {"transformation_total", trans_t}
        }}
    };
    return jsn;
}

// Writes a simplification run into the given path as .json.
// @tparam Point the point type
// @tparam DenseMT the dense matrix type of the nullspace basis
// @tparam VT the vector type of the shift vector
// @param model_path path to the .json file
// @param name the name of the model
// @param P the original polytope
// @param res the simplification result
// @param HP the full dimensional polytope
// @param N a basis of the nullspace of the equality system
// @param shift a particular solution of the equality system
// @param trans_t the transformation time in seconds
template <typename Point, typename DenseMT, typename VT>
void export_to_json(std::string model_path,
                    std::string const& name,
                    ConstrainedPolytope<Point> const& P,
                    SimplifierResult<Point> const& res,
                    HPolytope<Point> const& HP,
                    DenseMT const& N,
                    VT const& shift,
                    double trans_t)
{
    nlohmann::json jsn = construct_json(name, P, res, HP, N, shift, trans_t);

    std::ofstream f(model_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open "+model_path+" for writing");

    f << jsn.dump();

    if (!f)
        throw std::runtime_error("Failed while writing "+model_path);
}
#endif