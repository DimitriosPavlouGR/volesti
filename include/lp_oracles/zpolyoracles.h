// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef ZPOLYORACLES_H
#define ZPOLYORACLES_H

// Selects the LP library at compile time. lp_solve remains the default,
// HiGHS is used when VOLESTI_USE_HIGHS is defined.
#ifdef USE_HIGHS
    #include "lp_oracles/zpolyoracles_highs.hpp"
#else
    #include "lp_oracles/zpolyoracles_lpsolve.h"
#endif

#endif