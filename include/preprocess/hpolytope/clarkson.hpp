#ifndef CLARKSON_HPP
#define CLARKSON_HPP

#include <vector>
#include <set>
#include <tuple>
#include <limits>
#include <iostream>
#include "convex_bodies/hpolytope.h"
#include "lp_oracles/lp_oracle_options.hpp"

template <typename MT, typename VT>
std::tuple<bool, typename VT::Scalar, bool> test_redundancy(Highs & highs, MT const& A, VT const& b,
                                                            unsigned k, double facet_tol)
{
    typedef typename VT::Scalar NT;

    unsigned d = (unsigned)A.cols();

    highs.changeRowBounds((HighsInt)k, -kHighsInf, (double)b(k)+1.0);

    for (unsigned i = 0; i < d; ++i) {
        highs.changeColCost((HighsInt)i, (double)A(k, i));
    }
    highs.changeObjectiveSense(ObjSense::kMaximize);
    highs.run();

    HighsModelStatus st = highs.getModelStatus();

    highs.changeRowBounds((HighsInt)k, -kHighsInf, kHighsInf);
    for (unsigned i = 0; i < d; ++i)
        highs.changeColCost((HighsInt)i, 0.0);

    std::cout << "row " << k
              << " cols " << highs.getNumCol()
              << " rows " << highs.getNumRow()
              << " cost0 " << highs.getLp().col_cost_[0]
              << " sense " << (int)highs.getLp().sense_
              << std::endl;
              
    if (st != HighsModelStatus::kOptimal) {
        #ifdef VOLESTI_DEBUG
        std::cout << "clarkson: lp status "
                  << (int)st 
                  << " on row "
                  << k 
                  << std::endl;
        #endif
        return {false, NT(0), false};
    }

    NT opt = (NT)highs.getObjectiveValue();
    return {opt <= b(k)+facet_tol, opt, true};

}

template <typename MT, typename VT>
std::pair<unsigned, bool> rayshoot_shoot(MT const& A, VT const& b,
                                         VT const& z, VT const& r,
                                         double ray_tol)
{
    typedef typename VT::Scalar NT;

    unsigned m = (unsigned)A.rows();

    VT Az = A*z;
    VT Ar = A*r;

    // We need the first point where the ray hits a constraint.
    NT best_t = std::numeric_limits<NT>::infinity();

    unsigned hit = 0;
    bool found = false;

    for (unsigned i = 0; i < m; ++i) {
        if (Ar(i) <= ray_tol) continue;

        NT t = (b(i)-Az(i))/Ar(i);
        if (t < NT(0)) continue;

        if (!found || t < best_t) {
            best_t = t;
            hit = i;
            found = true;
        }
    }

    return {hit, found};
}

template <typename MT, typename VT>
std::pair<bool, unsigned> clarkson(Highs & highs, MT const& A, VT const& b, 
                                   VT const& z, unsigned k, double facet_tol, 
                                   double ray_tol)
{
    typedef typename VT::Scalar NT;

    unsigned d = (unsigned)A.cols();

    auto [is_redundant, opt, ok] = test_redundancy(highs, A, b, k, facet_tol);

    if (!ok) return {true, k}; // Safeguard for arithmetic failiures

    if (is_redundant) return {false, k};

    const auto& sol = highs.getSolution().col_value;
    VT x(d);
    for (unsigned i = 0; i < d; ++i)
        x(i) = (NT)sol[i];

    auto [j, hit] = rayshoot_shoot(A, b, z, VT(x-z), ray_tol);
    
    if (!hit) return {true, k}; // Safeguard for arithmetic failiures

    return {true, j};
}

template <typename MT, typename VT> 
std::vector<unsigned> redundancy_removal_clarkson(MT const& A, VT const& b, VT const& z,
                                                  LPOracleOptions const& opts = nullptr,
                                                  double facet_tol = 1e-7, double ray_tol = 1e-7)

{
    unsigned d = (unsigned)A.cols();
    unsigned m = (unsigned)A.rows();

    Highs highs;
    lp_oracles_configure_highs(highs, opts);

    for (unsigned i = 0; i < d; ++i)
        highs.addVar(-kHighsInf, kHighsInf);

    std::vector<HighsInt> indices(d);
    std::vector<double> values(d);
    for (unsigned i = 0; i < d; ++i)
        indices[i] = (HighsInt)i;

    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < d; ++j) 
            values[j] = (double)A(i, j);
        
        highs.addRow(-kHighsInf, kHighsInf, d, indices.data(), values.data());
    }

    std::set<unsigned> I, J;
    for (unsigned i = 0; i < m; ++i) J.insert(i);

    while (!J.empty()) {
        unsigned k = *J.begin();

        auto [is_essential, j] = clarkson(highs, A, b, z, k, facet_tol, ray_tol);

        if (!is_essential) {
            J.erase(j);
            continue;
        }

        unsigned settled = I.empty() ? k : j;

        I.insert(settled);
        highs.changeRowBounds((HighsInt)settled, -kHighsInf, (double)b(settled));
        J.erase(settled);
    }

    return std::vector<unsigned>(I.begin(), I.end());
}

template <typename Point>
std::pair<HPolytope<Point>, bool> remove_redundant_facets(HPolytope<Point> const& P,
                                                          LPOracleOptions const& opt = nullptr,
                                                          double interior_tol = 1e-7, double facet_tol = 1e-7, 
                                                          double ray_tol = 1e-9)
{
    typedef typename HPolytope<Point>::MT MT;
    typedef typename HPolytope<Point>::VT VT;
    typedef typename HPolytope<Point>::NT NT;

    MT const& A = P.get_mat();
    VT const& b = P.get_vec();

    auto [x, r, ok] = ComputeChebychevBall<NT, Point>(A, b, opt);
    if (!ok || r <= facet_tol) {
        return {P, false};
    }

    std::vector<unsigned> kept = redundancy_removal_clarkson(
        A, b, x.getCoefficients(), opt, facet_tol, ray_tol);
    
    MT A_new(kept.size(), A.cols());
    VT b_new(kept.size());
    for (unsigned i = 0; i < kept.size(); ++i) {
        A_new.row(i) = A.row(kept[i]);
        b_new(i) = b(kept[i]);
    }

    return {HPolytope<Point>((unsigned)A.cols(), A_new, b_new), true};
}

#endif