// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef CONSTRAINED_POLYTOPE_SIMPLIFIER_HPP
#define CONSTRAINED_POLYTOPE_SIMPLIFIER_HPP

#include "convex_bodies/constrained_polytope.hpp"
#include "preprocess/constrained_polytope/reduce_polytope.hpp"
#include "preprocess/constrained_polytope/interior_point.hpp"
#include "preprocess/constrained_polytope/clarkson.hpp"
#include "preprocess/constrained_polytope/presolve.hpp"

// Configuration for the simplification pipeline.
struct SimplifierConfig {
    // Configuration for dimension fixing.
    ReduceConfig reduce;

    // Configuration for dependent row removal.
    DependentRowsRemovalConfig dependent_rows;

    // Configuration for Clarkson's redundancy removal.
    ClarksonConfig clarkson;

    // The minimum slack for the interior point to count.
    double interior_tol = 1e-9;

    // Tracks if the reducer should be executed.
    bool fix_dimensions = true;

    // Tracks if presolve should happen.
    bool presolve = true;

    // Tracks if dependent rows should be removed.
    bool remove_dependent_rows = true;
};

// The outcome of a simplification run.
enum class SimplifierStatus {
    OK,
    PRESOLVE_FAILED,
    REMOVE_DEPENDENT_ROWS_FAILED,
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
    ConstrainedPolytope<Point> Ps = P;

    auto fail = [&](SimplifierStatus s) {
        res.status = s;
        res.polytope = Ps;
        return res;
    };

    // Stage 1: relax pinned variables by A_eq = b_eq (presolve)
    if (config.presolve) {
        auto pres = presolve_pinned_vars(Ps);
        if (!pres.valid) return fail(SimplifierStatus::PRESOLVE_FAILED);
        Ps = pres.polytope;
    }

    // Stage 2: fix degenerate dimensions
    if (config.fix_dimensions) {
        auto rres = reduce_polytope(Ps, config.reduce);
        res.solved_lps += rres.solved_lps;
        if (!rres.valid) return fail(SimplifierStatus::INFEASIBLE);
        Ps = rres.polytope;
    }

    // Stage 3: dependent equality rows
    if (config.remove_dependent_rows) {
        auto dres = remove_dependent_rows(Ps, config.dependent_rows);
        if (!dres.valid) return fail(SimplifierStatus::REMOVE_DEPENDENT_ROWS_FAILED);
        Ps = dres.polytope;
    }

    // Stage 4: Interior point
    auto ip = find_interior_point(Ps, config.interior_tol);
    res.solved_lps += 1;
    if (!ip.valid) return fail(SimplifierStatus::NO_INTERIOR_POINT);

    // Stage 5: Clarkson redundancy removal
    auto cres = redundancy_removal_clarkson(Ps, ip.point, config.clarkson);
    res.solved_lps += cres.solved_lps;
    if (!cres.valid) return fail(SimplifierStatus::CLARKSON_FAILED);

    res.polytope = cres.polytope;
    return res;
}

#endif