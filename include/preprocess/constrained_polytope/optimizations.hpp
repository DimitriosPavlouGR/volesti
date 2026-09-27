// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Sparse>
#include "cholmod.h"
#include "SuiteSparseQR.hpp"
#include "convex_bodies/constrained_polytope.hpp"

// Configuration for removing linearly dependent equality rows.
struct DependentRowsConfig {

};

#endif