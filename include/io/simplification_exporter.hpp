// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef SIMPLIFICATION_EXPORTER_HPP
#define SIMPLIFICATION_EXPORTER_HPP

#include <nlohmann/json.hpp>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <Eigen/Eigen>
#include <cmath>
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/metabolic_polytope.hpp"
#include "preprocess/metabolic/simplification_clarkson.hpp"

// Converts the LP counters to json.
// @param j the JSON to fill
// @param s the counters
inline void to_json(nlohmann::json& j, LpStats const& s) {
    j = nlohmann::json{
        {"solves", s.solves},
        {"retries", s.retries},
        {"failures", s.failures},
        {"iterations", s.iterations}
    };
}

// Converts a phase into JSON.
// @param j the JSON to fill
// @param p the phase
inline void to_json(nlohmann::json& j, PhaseReport const& p) {
    j = nlohmann::json{
        {"name", p.name},
        {"seconds", p.seconds},
        {"lps", p.lps}
    };
}

// Converts the report into a JSON, grouped by stage in the order they run.
// @param j the JSON to fill
// @param r the report
inline void to_json(nlohmann::json& j, SimplificationReport const& r) {
    j["status"] = r.status;

    j["input"] = {
        {"variables", r.dimension},
        {"equalities", r.input_equalities},
        {"finite_bounds", r.input_finite_bounds}
    };

    if (r.rank >= 0) {
        j["presolve"] = {
            {"rank", r.rank},
            {"pinned_by_equalities", r.pinned_by_equalities}
        };
    } else {
        j["presolve"] = nullptr;
    }

    if (r.fixing_ran) {
        j["dimension_fixing"] = {
            {"fixed_by_bounds", r.fixed_by_bounds},
            {"equalities_after_fixing", r.equalities_after_fixing}
        };
    } else {
        j["dimension_fixing"] = nullptr;
    }

    j["row_reduction"] = {
        {"rows_before", r.rows_before_reduction},
        {"rows_after", r.rows_after_reduction},
        {"applied", r.row_reduction_applied},
        {"max_dropped_residual", r.dropped_row_residual}
    };

    j["interior_point"] = {
        {"found", r.interior_found},
        {"slack", r.interior_slack}
    };

    j["clarkson"] = {
        {"untested_bounds", r.untested_bounds},
    };

    j["output"] = {
        {"equalities", r.output_equalities},
        {"finite_bounds", r.output_finite_bounds},
        {"bounds_relaxed", r.input_finite_bounds-r.output_finite_bounds}
    };

    j["phases"] = r.phases;

    j["total"] = {
        {"seconds", r.total_seconds},
        {"lps", r.total}
    };
}

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
    auto const& A = HP.get_mat();
    auto const& b = HP.get_vec();

    nlohmann::json transformed_jsn;
    transformed_jsn["A"] = matrix_to_json(A);
    transformed_jsn["b"] = vector_to_json(b);
    transformed_jsn["N"] = matrix_to_json(N);
    transformed_jsn["shift"] = vector_to_json(shift);

    return transformed_jsn;
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

// Constructs the .json of the metabolic network.
// @tparam Point the point type
// @param P the metabolic polytope
// @return the JSON object holding the network
template <typename Point>
nlohmann::json construct_network_json(MetabolicPolytope<Point> const& P)
{
    nlohmann::json jsn;
    jsn["reaction_count"] = (long)P.getDimension();
    jsn["metabolite_count"] = (long)P.getEqualities().rows();
    jsn["A_eq"] = sparse_matrix_to_json(P.getEqualities());
    jsn["b_eq"] = vector_to_json(P.getEqualityBounds());
    jsn["b_l"]  = bounds_to_json(P.getLowerBounds());
    jsn["b_u"]  = bounds_to_json(P.getUpperBounds());

    return jsn;
}

// Constructs the .json of the simplification run
// @tparam Point the point type
// @tparam DenseMT the dense matrix type of the nullspace basis
// @tparam VT the vector type of the shift vector
// @param name the name of the model
// @param P the original network
// @param Ps the simplified network
// @param HP the full dimensional polytope
// @param N a basis of the nullspace of the equality system
// @param shift a particular solution of the equality system
// @param report the simplification report
template <typename Point, typename DenseMT, typename VT>
nlohmann::json construct_json(std::string const& name,
                              MetabolicPolytope<Point> const& P,
                              MetabolicPolytope<Point> const& Ps,
                              HPolytope<Point> const& HP,
                              DenseMT const& N,
                              VT const& shift,
                              SimplificationReport const* report = nullptr)
{
    nlohmann::json jsn;
    jsn["name"] = name;
    jsn["original"] = construct_network_json(P);
    jsn["simplified"] = construct_network_json(Ps);
    jsn["transformed"] = construct_transformed_json(HP, N, shift);

    if (report) jsn["report"] = *report;

    return jsn;
}

// Writes a simplification run into the given path as .json.
// @tparam Point the point type
// @tparam DenseMT the dense matrix type of the nullspace basis
// @tparam VT the vector type of the shift vector
// @param model_path path to the .json file
// @param name the name of the model
// @param P the original network
// @param Ps the simplified network
// @param HP the full dimensional polytope
// @param N a basis of the nullspace of the equality system
// @param shift a particular solution of the equality system
// @param report the simplification report
template <typename Point, typename DenseMT, typename VT>
void export_to_json(std::string model_path,
                    std::string const& name,
                    MetabolicPolytope<Point> const& P,
                    MetabolicPolytope<Point> const& Ps,
                    HPolytope<Point> const& HP,
                    DenseMT const& N,
                    VT const& shift,
                    SimplificationReport const* report = nullptr)
{
    nlohmann::json jsn = construct_json(name, P, Ps, HP, N, shift, report);

    std::ofstream f(model_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open "+model_path+" for writing");

    f << jsn.dump();

    if (!f)
        throw std::runtime_error("Failed while writing "+model_path);
}
#endif