// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "preprocess/constrained_polytope/bigg_parser.hpp"
#include "preprocess/constrained_polytope/simplifier.hpp"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef ConstrainedPolytope<Point> Polytope;

static std::filesystem::path const EXPORT_DIR  = "simplified_constrained";

void benchmark(std::string const& model_path) {
    Polytope P = parse_from_json<Point>(model_path);
    std::string name = std::filesystem::path(model_path).stem().string();

    std::cout << "[" << name << "]"
              << "  n=" << P.getDimension()
              << "  m_eq=" << P.getNumEqualities()
              << "  m_in=" << P.getNumInequalities()
              << "  bounds=" << P.getNumFiniteBounds() << "\n" << std::flush;

    SimplifierConfig config;
    config.clarkson.seed = 1;
    auto start = std::chrono::steady_clock::now();
    SimplifierResult res = simplify(P, config);
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
    
    if (res.status != SimplifierStatus::OK) {
        std::cout << "failed (status " << (int)res.status <<") " << ela
    }
    Polytope const& out = res.polytope;
    std::cout << "ok\n"
              << "  ->  m_eq=" << out.getNumEqualities()
              << "  m_in=" << out.getNumInequalities()
              << "  bounds=" << out.getNumFiniteBounds()
              << "  removed=" << (P.getNumFiniteBounds()-out.getNumFiniteBounds())
              << "  lps=" << (res.solved_lps)
              << "  " << elapsed << "s\n" << std::flush;
}

int main() {
    std::vector<std::filesystem::path> models;
    for (auto const& file : std::filesystem::directory_iterator(BIGG_DIR)) {
        if (file.path().extension() != ".json") continue;
        models.push_back(file.path());
    }
    std::sort(models.begin(), models.end());

    for (auto const& model : models)
        benchmark(model.string());

    return 0;
}