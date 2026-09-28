// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef CONSTRAINED_POLYTOPE_PRESOLVE_HPP
#define CONSTRAINED_POLYTOPE_PRESOLVE_HPP

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <Eigen/Sparse>
#include "cholmod.h"
#include "SuiteSparseQR.hpp"
#include "convex_bodies/constrained_polytope.hpp"

// Configuration for finding the variables pinned by the equalities.
struct PinnedPresolveConfig {
    // The rank tolerance of the QR factorization.
    double rank_tol = 1e-7;

    // A variable is pinned if its row of the nullspace basis is below this.
    double pinned_tol = 1e-7;
};

#endif