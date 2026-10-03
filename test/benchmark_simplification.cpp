// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#include <algorithm>
#include <iostream>
#include <filesystem>
#include <vector>
#include <ctime>
#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "preprocess/constrained_polytope/bigg_parser.hpp"
#include "preprocess/constrained_polytope/simplifier.hpp"
#include "preprocess/constrained_polytope/transformation.hpp"
#include "preprocess/constrained_polytope/simplification_exporter.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef ConstrainedPolytope<Point> Polytope;

static std::filesystem::path const EXPORT_DIR = "simplified_constrained";

void benchmark(std::string const& model_path, std::filesystem::path const& export_dir) {
    Polytope P = parse_from_json<Point>(model_path);

    std::string name = std::filesystem::path(model_path).stem().string();

    std::cout << "[" << name << "]"
              << "  n=" << P.getDimension()
              << "  m_eq=" << P.getNumEqualities()
              << "  m_in=" << P.getNumInequalities()
              << "  finite bounds=" << P.getNumFiniteBounds() << std::endl;

    SimplifierConfig config;

    auto res = simplify(P, config);

    if (res.status != SimplifierStatus::OK) {
        std::cout << "failed with status=" << (int)res.status << ", " << res.times.total() << std::endl;
        return;
    }

    Polytope const& Ps = res.polytope;
    std::cout << "ok"
              << "  -> m_eq=" << Ps.getNumEqualities()
              << "  m_in=" << Ps.getNumInequalities()
              << "  finite bounds=" << Ps.getNumFiniteBounds()
              << "  removed=" << (P.getNumFiniteBounds()-Ps.getNumFiniteBounds())
              << "  " << res.times.total() << std::endl; 

    // Grabs the full dimensional polytope
    std::clock_t c0 = std::clock();
    auto [HP, shift, N] = transform(Ps);
    std::clock_t c1 = std::clock();
    double trans_t = double(c1-c0)/CLOCKS_PER_SEC;

    std::filesystem::path export_path = export_dir/(name+"_clarkson.json");
    export_to_json(export_path.string(), name, P, res, HP, N, shift, trans_t);

    std::cout << "  exported to " << export_path.string() << std::endl;
}

int main() {
    std::filesystem::create_directories(EXPORT_DIR);

    std::vector<std::filesystem::path> models;
    for (auto const& file : std::filesystem::directory_iterator(BIGG_DIR)) {
        if (file.path().extension() != ".json") continue;
        models.push_back(file.path());
    }

    std::sort(models.begin(), models.end());

    for (auto const& model : models)
        benchmark(model.string(), EXPORT_DIR);

    return 0;
}