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
              << std::flush;

    SimplifierConfig config;
    config.fix_dimensions = true;

    std::cout << "  stage 1..." << std::flush;
    auto rres = reduce_polytope(P, config.reduce);
    std::cout << (rres.valid ? "ok" : "failed") << std::flush;

    std::cout << "  stage 2..." << std::flush;
    auto ip = find_interior_point(rres.polytope, config.interior_tol);
    std::cout << (ip.valid ? "ok" : "failed") << std::flush;

    std::cout << "  stage 3..." << std::flush;
    auto cres = redundancy_removal_clarkson(rres.polytope, ip.point, config.clarkson);
    std::cout << (cres.valid ? "ok" : "failed") << "\n" << std::flush;

    auto start = std::chrono::steady_clock::now();
    auto res   = simplify(P, config);
    double elapsed = std::chrono::duration<double>(
                     std::chrono::steady_clock::now()-start).count();

    switch (res.status) {
        case SimplifierStatus::OK:
            std::cout << "  ->  m_in=" << res.polytope.getNumInequalities()
                      << "  removed=" << (P.getNumInequalities()-res.polytope.getNumInequalities())
                      << "  lps=" << res.solved_lps
                      << "  " << elapsed << "s\n";
            break;
        case SimplifierStatus::INFEASIBLE:
            std::cout << "  INFEASIBLE\n"; break;
        case SimplifierStatus::NO_INTERIOR_POINT:
            std::cout << "  NO INTERIOR POINT\n"; break;
        case SimplifierStatus::CLARKSON_FAILED:
            std::cout << "  CLARKSON FAILED\n"; break;
    }
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