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
#include "preprocess/constrained_polytope/lp_parser.hpp"
#include "preprocess/constrained_polytope/simplifier.hpp"
#include "preprocess/constrained_polytope/transformation.hpp"
#include "preprocess/constrained_polytope/simplification_exporter.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef ConstrainedPolytope<Point> Polytope;

namespace fs = std::filesystem;

static fs::path const EXPORT_DIR = "simplified_netlib";

// Returns if the model follows a valid format.
// @param path the model's path
// @return true if the model is in valid format
bool is_valid_format(fs::path const& path) {
    std::string const ext = path.extension().string();
    if (ext == ".mps" || ext == ".lp") return true;
    return ext == ".gz" && path.stem().extension() == ".mps";
}

// Returns the model's name without the extension
// @param path the model's path
// @return the name as a string
std::string model_name(fs::path const& path) {
    fs::path p = path.filename();
    if (p.extension() == ".gz") p = p.stem();
    return p.stem().string();
}

void benchmark(std::string const& model_path, std::filesystem::path const& export_dir) {
    std::string const name = model_name(model_path);

    Polytope P;
    try {
        P = parse_from_lp_model<Point>(model_path);
    } catch (std::exception const& e) {
        std::cout << "[" << name << "] cannot load: " << e.what() << std::endl;
        return;
    }

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
              << "  removed: rows=" << (P.getNumInequalities()-Ps.getNumInequalities())
              << "  removed: bounds=" << (P.getNumFiniteBounds()-Ps.getNumFiniteBounds())
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
    std::vector<fs::path> models;
    for (auto const& file : fs::directory_iterator(NETLIB_DIR)) {
        if (!file.is_regular_file() || !is_valid_format(file.path())) continue;
        models.push_back(file.path());
    }

    std::sort(models.begin(), models.end());
    fs::create_directories(EXPORT_DIR);

    for (auto const& model : models) {
        std::string const name = model_name(model);

        // skips the ones that exist
        if (fs::exists(EXPORT_DIR/(name+"_clarkson.json"))) {
            std::cout << "[" << name << "] already exported, skipping" << std::endl;
            continue;
        }

        // skips ones that crash
        try {
            benchmark(model.string(), EXPORT_DIR);
        } catch (std::exception const& e) {
            std::cout << "[" << name << "] crashed: " << e.what() << ", skipping" << std::endl;
        }
    }
    return 0;
}