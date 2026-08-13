// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef CONFIGURE_HIGHS_HPP
#define CONFIGURE_HIGHS_HPP

#include "Highs.h"

// Configures highs for the lp oracles.
inline void lp_oracles_configure_highs(Highs & highs) {
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("solver", "simplex");
}

#endif