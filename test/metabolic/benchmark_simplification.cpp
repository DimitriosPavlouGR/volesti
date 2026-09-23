// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "io/bigg_parser.hpp"
#include "io/simplification_exporter.hpp"
#include "preprocess/metabolic/simplification_exhaustive.hpp"
#include "preprocess/metabolic/simplification_clarkson.hpp"
#include "preprocess/metabolic/transformation.hpp"
#include "preprocess/metabolic/quality_metrics.hpp"
#include "lp_oracles/metabolic_polyoracles.hpp"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <vector>

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef MetabolicPolytope<Point> Polytope;

typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> DenseVT;
typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> DenseMT;

static std::filesystem::path const EXPORT_DIR = "simplified_bigg";

template <typename Simplifier, typename Config>
void run_method(std::string method, std::string const& name, Polytope const& P, bool dimension_fixing) {
    Config config;
    config.verbose = true;
    config.fix_dimensions = dimension_fixing;
    
    auto start = std::chrono::high_resolution_clock::now();
    Simplifier simplifier(P, config);
    auto [Ps, success] = simplifier.simplify();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_s = std::chrono::duration<double>(end-start).count();

    std::string status = success ? "OK" : "FAILED";

    if (success) {
        auto trans_result = transform(Ps);
        HPolytope<Point> HP = std::get<0>(trans_result);
        DenseVT shift = std::get<1>(trans_result);
        DenseMT N = std::get<2>(trans_result);
        std::filesystem::create_directories(EXPORT_DIR);
        std::filesystem::path out = EXPORT_DIR/(name+"_"+method+".json");
        export_to_json(out.string(), name, P, Ps, HP, N, shift, &simplifier.getReport());
    }
}

void benchmark(std::string const& model, bool dimension_fixing) {
    Polytope P = parse_from_json<Point>(model);
    std::string name = std::filesystem::path(model).stem().string();

    run_method<ClarksonSimplifier<Point>, ClarksonConfig>(
        "clarkson", name, P, dimension_fixing);
}

int main() {
    // Collects and sorts the models to maintain a fixed order in testing.
    std::vector<std::filesystem::path> models;
    for (auto const& file : std::filesystem::directory_iterator(BIGG_DIR)) {
        if (file.path().extension() != ".json") continue;
        models.push_back(file.path());
    }
    std::sort(models.begin(), models.end());

    for (auto const& model : models) {
        benchmark(model.string(), true);  // with dimension fixing
    }

    return 0;
}