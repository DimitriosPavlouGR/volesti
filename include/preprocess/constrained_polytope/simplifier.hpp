// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef SIMPLIFIER_HPP
#define SIMPLIFIER_HPP

#include "convex_bodies/constrained_polytope.hpp"
#include "preprocess/constrained_polytope/reduce_polytope.hpp"
#include "preprocess/constrained_polytope/interior_point.hpp"
#include "preprocess/constrained_polytope/clarkson.hpp"

// Configuration for the simplification pipeline.
struct SimplifierConfig {
    // Configuration for dimension fixing.
    ReduceConfig reduce;

    // Configuration for Clarkson's redundancy removal.
    ClarksonConfig clarkson;

    // The minimum slack for the interior point to count.
    double interior_tol = 1e-9;

    // Tracks if the reducer should be executed.
    bool fix_dimensions = true;
};

// The outcome of a simplification run.
enum class SimplifierStatus {
    OK,
    INFEASIBLE,
    NO_INTERIOR_POINT,
    CLARKSON_FAILED
};

// Result of the simplification pipeline.
template <typename Point> 
struct SimplifierResult {
   // The simplified polytope.
   ConstrainedPolytope<Point> polytope; 
   
   // Number of LPs solved.
   unsigned solved_lps = 0;

   // The outcome of simplification
   SimplifierStatus status = SimplifierStatus::OK;
};

// Simplifies a ConstrainedPolytope by running reducing it, finding an interior point
// and removing redundant inequality rows using Clarkson's algorithm.
//
// Pipeline:
//      1. reduce_polytope, fixes degenerate dimensions
//      2. find_interior_point, finds a strictly interior point
//      3. redundancy_removal_clarkson, removes redundant rows from A_in
// @tparam Point the point type
// @param P the input polytope
// @param config the simplification configuration
// @return the result
template <typename Point>
SimplifierResult<Point> simplify(ConstrainedPolytope<Point> const& P,
                                 SimplifierConfig const& config = SimplifierConfig{})
{
    SimplifierResult<Point> res;

    // Stage 1: fix degenerate dimensions
    ConstrainedPolytope<Point> Ps = P;
    if (config.fix_dimensions) {
        auto rres = reduce_polytope(P, config.reduce);
        if (!rres.valid) {
            res.status = SimplifierStatus::INFEASIBLE;
            return res;
        }
        Ps = rres.polytope;
        res.solved_lps += rres.solved_lps;
    }

    // Stage 2: Interior point
    auto ip = find_interior_point(Ps, config.interior_tol);
    
    if (!ip.valid) {
        res.status = SimplifierStatus::NO_INTERIOR_POINT;
        return res;
    }

    // Stage 3: Clarkson redundancy removal
    auto cres = redundancy_removal_clarkson(Ps, ip.point, config.clarkson);
    if (!cres.valid) {
        res.status = SimplifierStatus::CLARKSON_FAILED;
        return res;
    }

    res.polytope = cres.polytope;
    return res;
}

#endif