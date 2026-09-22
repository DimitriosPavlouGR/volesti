// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2026 Vissarion Fisikopoulos
// Copyright (c) 2018-2026 Apostolos Chalkis
// Copyright (c) 2026      Dimitrios Pavlou

// Contributed and/or modified by Dimitrios Pavlou, as part of Google Summer of Code 2026 program

// Licensed under GNU LGPL.3, see LICENCE file

#ifndef METABOLIC_SIMPLIFICATION_CLARKSON_HPP
#define METABOLIC_SIMPLIFICATION_CLARKSON_HPP

#include <iostream>
#include <array>
#include <vector>
#include <queue>
#include <utility>
#include <random>
#include <cmath>
#include <limits>
#include <Eigen/Eigen>
#include "convex_bodies/metabolic_polytope.hpp"
#include "preprocess/metabolic/simplification_exhaustive.hpp"
#include "preprocess/metabolic/presolve.hpp"
#include "common.hpp"
#include "Highs.h"
#include <chrono>

// Configuration parameters controlling the simplification process.
struct ClarksonConfig : ExhaustiveConfig {
    // The error tolerance for the interior point.
    double interior_tolerance = 1e-9;
    
    // The error tolerance for the ray shooting stage of clarkson.
    double ray_tolerance = 1e-9;

    // The gap by which a bound is relaxed in the redundancy LP.
    double relaxation_gap = 1.0;

    // The bound on the number of failed iteration's in clarkson.
    unsigned failed_iter_count = 2;

    // Tracks if presolve should be used, turned on by default.
    bool presolve = true;

    // The seed based on which clarkson picks inequalities.
    unsigned clarkson_seed = 0;

    // This is the observation tolerance.
    double observe_tolerance = 1e-6;

    // This is the residual tolerance when removing linear independent rows.
    double residual_tolerance = 1e-7;
};

// Counters of the LP work done during a simplification run.
struct LpStats {
    // Number of calls to HiGHS run(), includes failed runs
    unsigned solves = 0;

    // Number of cold re-solves after a non-optimal status.
    unsigned retries = 0;

    // Solves that failed even after a retry.
    unsigned failures = 0;

    // Simplex iterations summed over all solves.
    long long iterations = 0;
};

// The time and LP work spent in one phase of simplification.
struct PhaseReport {
    // The name of the phase.
    std::string name;

    // The wall clock time spent in the phase.
    double seconds = 0.0;

    // The LP work done in the phase.
    LpStats lps;
};

// All useful information worth knowing about a simplification run, collected while it runs.
struct SimplificationReport {
    // The outcome of the run.
    std::string status;

    // The number of variables.
    unsigned dimension = 0;

    // The number of equality rows.
    unsigned input_equalities = 0;

    // The number of finite bounds of the input.
    unsigned input_finite_bounds = 0;

    // The rank of the input equality matrix, -1 if presolve did not run.
    long rank = -1;

    // Number of skipped bounds during clarkson due to HiGHS failures.
    unsigned untested_bounds = 0;

    // The number of variables pinned by the equalities of the homogeneoys system A_eq x = 0.
    unsigned pinned_by_equalities = 0;

    // Tracks if dimension fixing ran.
    bool fixing_ran = false;

    // The number of variables fixed by bounds.
    unsigned fixed_by_bounds = 0;

    // The variables fixed by dimension fixing.
    unsigned equalities_after_fixing = 0;

    // The number of equality rows before and after dropping the dependent rows.
    unsigned rows_before_reduction = 0;
    unsigned rows_after_reduction = 0;

    // Tracks if the dependent rows were removed.
    bool row_reduction_applied = false;

    // The largest violation of a dropped row at a feasible point.
    double dropped_row_residual = 0.0;

    // Finds if an interior point was found and the slack value.
    bool interior_found = false;
    double interior_slack = 0.0;

    // The number of equality rows and finite bounds after simplification.
    unsigned output_equalities = 0;
    unsigned output_finite_bounds = 0;

    // The time and LP work of every phase in order.
    std::vector<PhaseReport> phases;

    // The LP work and wall clock time of the whole run.
    LpStats total;
    double total_seconds = 0.0;
};

// Simplifies a MetabolicPolytope using Clarkson's algorithm. The model starts with
// every bound relaxed and iteratively finds and applies essential constraints to the model.
//
// Clarkson needs a point in the interior of the polytope, which only exists once
// degenerate dimensions have been fixed. If no interior point is found the method cannot
// run and the user has to manually change the configuration parameters.
// @tparam Point the point type of the polytope
template <typename Point>
class ClarksonSimplifier {
    public:
        // Types.
        typedef typename MetabolicPolytope<Point>::MT MT;
        typedef typename MetabolicPolytope<Point>::VT VT;
        typedef typename MetabolicPolytope<Point>::NT NT;

        // Builds the LP model of the given polytope.
        // @param P_in the polytope to simplify
        // @param config_in the simplification configuration
        ClarksonSimplifier(MetabolicPolytope<Point> const& P_in,
                        ClarksonConfig const& config_in = ClarksonConfig{})
            : config(config_in), P(P_in)
        {   
            configure_highs(highs, config);
            build_lp_model(P, highs);
        }

        // Runs the simplification.
        // @return the simplified polytope, and false if the polytope was empty
        std::pair<MetabolicPolytope<Point>, bool> simplify() {
            report = SimplificationReport{};
            start_time = std::chrono::steady_clock::now();

            report.dimension = P.getDimension();
            report.input_equalities = P.getNumEqualities();
            report.input_finite_bounds = P.getNumFiniteBounds();

            MetabolicPolytope<Point> Ps = P;

            Phase p;

            // Presolve phase.
            if (config.presolve) {
                p = begin_phase();
                this->presolve();
                end_phase("presolve", p);
            }

            // Dimension fixing phase.
            if (config.fix_dimensions) {
                p = begin_phase();
                Ps = fix_degenerate_dimensions_observe();
                P = Ps;
                report.fixing_ran = true;
                report.equalities_after_fixing = Ps.getNumEqualities();
                end_phase("dimension fixing", p);
            }
            
            p = begin_phase();
            if (!remove_dependent_rows_from_polytope(Ps)) 
                return finalize(P, false, "INFEASIBLE");
            P = Ps;

            highs.clear();
            configure_highs(highs, config);
            build_lp_model(P, highs);
            end_phase("row reduction", p);


            p = begin_phase();
            bool has_interior = find_interior_point();
            report.interior_found = has_interior;
            end_phase("interior point", p);

            if (has_interior) {
                p = begin_phase();
                Ps = redundancy_removal_clarkson();
                end_phase("clarkson", p);
            } else {
                return finalize(Ps, false, "NO INTERIOR POINT");
            }

            return finalize(Ps, true, "OK");
        }

        // Returns the interior point found before Clarkson runs. This point is undefined if
        // an interior point was not found before simplification.
        // @return an interior point
        VT const& getInteriorPoint() const {
            return z;
        }

        // Returns the LP work done so far.
        // @return the LP counters
        LpStats const& getLpStats() const {
            return lp_stats;
        }

        // Returns the simplification report produced by simplifier.
        // @return the simplification report
        SimplificationReport const& getReport() const {
            return report;
        }

    private:
        // A single side of the box bound, treated as a single row of the equivalent inequality
        // system A x <= b. An upper bound is of the form x_k <= b_u(k) and a lower bound is of
        // the form b_l(k) <= x_k.
        struct Ineq {
            // The index of the variable.
            unsigned k;

            // True for an upper bound, false for a lower bound.
            bool is_upper;

            // Maps this inequality to a unique index in [0, 2d-1], used to address
            // position in the I vector during Clarkson's algorithm.
            unsigned map() const {
                return 2u*k+(is_upper ? 1u : 0u);
            }
        };

        // The simplification configuration.
        ClarksonConfig config;

        // The input polytope.
        MetabolicPolytope<Point> P;

        // The LP model.
        Highs highs;

        // The interior point.
        VT z;

        // The LP work done logged.
        LpStats lp_stats;

        // The report of the simplification.
        SimplificationReport report;

        // The time the current run started.
        std::chrono::steady_clock::time_point start_time;

        // Substracts one set of counters from another.
        // @param a the earlier counters
        // @param b the later counters
        // @return the work done between the two
        static LpStats lp_diff(LpStats const& a, LpStats const& b) {
            LpStats c;
            c.solves = b.solves-a.solves;
            c.retries = b.retries-a.retries;
            c.failures = b.failures-a.failures;
            c.iterations = b.iterations-a.iterations;
            return c;
        }

        // Snapshot of the clock and the counters at the start of a phase.
        struct Phase {
            std::chrono::steady_clock::time_point t;
            LpStats s;
        };

        // Takes a snapshot of the clock and the counters at the start of the phase.
        // @return the snapshot
        Phase begin_phase() const {
            return {std::chrono::steady_clock::now(), lp_stats};
        }

        // Records the time and LP work spent since the snapshot was taken.
        // @param name the name of the phase
        // @param p the snapshot taken at the start of the phase
        void end_phase(std::string const& name, Phase const& p) {
            PhaseReport pr;
            pr.name = name;
            pr.seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now()-p.t).count();
            pr.lps = lp_diff(p.s, lp_stats);
            report.phases.push_back(pr);
        };

        // Prints the report of the last run.
        // @param os the stream to print to
        void print_report(std::ostream& os = std::cout) const {
            SimplificationReport const& r = report;

            std::ostringstream out;

            auto key = [&](std::string const& k) -> std::ostream& {
                return out << " " << std::left << std::setw(18) << k << std::right;
            };


            key("status") << r.status << "\n";
            key("input")  << r.dimension << " variables, " << r.input_equalities
                          << " equalities, " << r.input_finite_bounds << " finite bounds\n";

            if (r.rank >= 0) {
                key("presolve") << "rank " << r.rank << ", " << r.pinned_by_equalities
                                << " variables pinned by the equalities\n";
            } else {
                key("presolve") << "skipped\n";
            }

            if (r.fixing_ran) {
                key("dimension fixing") << r.fixed_by_bounds << " variables fixed, "
                                        << r.equalities_after_fixing << " equalities in total\n";
            } else {
                key("dimension fixing") << "skipped\n";
            }

            if (r.rows_before_reduction > 0) {
                key("row reduction") << r.rows_after_reduction << " of "
                                     << r.rows_before_reduction << " rows kept";
                if (r.row_reduction_applied) {
                    out << ", max residual of dropped rows "
                       << std::scientific << std::setprecision(2) << r.dropped_row_residual
                       << std::defaultfloat;
                } else {
                    out << " (not applied)";
                }
                out << "\n";
            }

            key("interior point") << (r.interior_found ? "found" : "not found");
            if (r.interior_slack > 0.0 || r.interior_found) {
                out << ", slack " << std::scientific << std::setprecision(3)
                   << r.interior_slack << std::defaultfloat;
            }
            out << "\n";

            if (r.untested_bounds) {
                key("clarkson") << r.untested_bounds
                                << " bounds kept untested after LP failures in Clarkson\n";
            }

            if (r.status == "OK") {
                unsigned relaxed = r.input_finite_bounds-r.output_finite_bounds;
                key("output") << r.output_equalities << " equalities, "
                              << r.output_finite_bounds << " finite bounds ("
                              << relaxed << " relaxed)\n";
            }

            out << " phases\n";

            auto phase_line = [&](std::string const& name, double secs, LpStats const& s) {
                out << "     " << std::left << std::setw(20) << name << std::right
                    << std::fixed << std::setprecision(3) << std::setw(10) << secs << "s"
                    << std::setw(9) << s.solves << " LPs"
                    << std::setw(11) << s.iterations << " iterations";
                if (s.retries > 0) out << ", " << s.retries << " retries";
                if (s.failures > 0) out << ", " << s.failures << " failed";
                out << "\n";

            };

            for (PhaseReport const& pr : r.phases) {
                phase_line(pr.name, pr.seconds, pr.lps);
            }
            phase_line("total", r.total_seconds, r.total);

            std::string body = out.str();
            std::istringstream lines(body);
            std::size_t width = 0;

            for (std::string l; std::getline(lines, l); width = std::max(width, l.size()));

            std::string title = " [VolEsti] - [ClarksonSimplifier] ";
            width = std::max(width, title.size()+2);
            std::size_t pad = width-title.size();
            std::size_t left = pad/2;

            os << std::string(left, '-') << title << std::string(pad-left, '-') << "\n"
               << body
               << std::string(width, '-') << "\n" << std::flush;
        }

        // Finalizes the report, prints it when config.verbose is on, and returns the result.
        // @param Pout the polytope to return
        // @param ok false if the polytope was found empty or the run failed
        // @param status the outcome to record
        // @return the result of simplify()
        std::pair<MetabolicPolytope<Point>, bool> finalize(MetabolicPolytope<Point> const& Pout,
                                                           bool ok, std::string const& status)
        {
            report.status = status;
            report.total = lp_stats;
            report.total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now()-start_time).count();

            if (ok) {
                report.output_equalities = Pout.getNumEqualities();
                report.output_finite_bounds = Pout.getNumFiniteBounds();
            }

            if (config.verbose) print_report();

            return {Pout, ok};
        }

        std::vector<double> max_observed, min_observed;

        // Runs the LP stored in highs.
        // @return true if solved to optimality
        bool run_lp() {
            highs.run();
            ++lp_stats.solves;
            lp_stats.iterations += highs.getInfo().simplex_iteration_count;

            if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
                ++lp_stats.retries;
                highs.clearSolver();
                highs.run();
                ++lp_stats.solves;
                lp_stats.iterations += highs.getInfo().simplex_iteration_count;
            }

            bool ok = highs.getModelStatus() == HighsModelStatus::kOptimal;
            
            if (!ok) ++lp_stats.failures;

            return ok;
        }

        // Evaluates the left hand side of the constraint a x <= b. Trivially
        // returns x_k or -x_k depending on the side of the inequality.
        // @param a the inequality
        // @param the point to evaluate at
        // @return the inner dot product <a,x>
        static double row_value(Ineq const& a, VT const& x) {
            double xk = (double)x(a.k);
            return a.is_upper ? xk : -xk;
        }

        // Returns the right hand side of the box bound c written
        // as a row a x <= b. Notice that the lower bound b_l(k) <= x_k becomes -x_k
        // <= -b_l(k).
        // @param a the inequality
        // @param b the bound vector
        // @return the right hand side of the row
        inline double row_rhs(Ineq const& a) {
            return a.is_upper ? (double)P.getUpperBounds()(a.k) : 
                               -(double)P.getLowerBounds()(a.k);
        }

        // Collapses the bounds of a variable, by adding x_k = val to A_eq.
        // @param k the variable index
        // @param val the value to fix it at
        void fix_dimension(unsigned k, NT val) {
            HighsInt id = k;
            double coeff = 1.0;
            highs.addRow((double)val, (double)val, 1, &id, &coeff);
            highs.changeColBounds(k, -kHighsInf, kHighsInf);
            ++report.fixed_by_bounds;
        }

        // Applies one bound of P to the highs model.
        // @param ineq the inequality to apply
        void enforce_ineq(Ineq const& ineq)
        {
            const VT& b_l = P.getLowerBounds();
            const VT& b_u = P.getUpperBounds();

            // Stores the old lp state.
            double lo = highs.getLp().col_lower_[ineq.k];
            double hi = highs.getLp().col_upper_[ineq.k];

            // Adds only a single side inequality.
            if (ineq.is_upper) {
                hi = std::isinf((double)b_u(ineq.k)) ? kHighsInf : (double)b_u(ineq.k);
            } else {
                lo = std::isinf((double)b_l(ineq.k)) ? -kHighsInf : (double)b_l(ineq.k);
            }
            highs.changeColBounds((HighsInt)ineq.k, lo, hi);
        }

        // Shoots the ray z+t*r, t >= 0, and returns the first box bound it crosses.
        // @param J the untested bounds
        // @param r the ray direction
        // @param hit receives the bound crossed first, meaningful only on success
        // @return the facet hit first, meaningful only when success is true
        bool ray_shoot(std::vector<Ineq> const& J, VT const& r, Ineq & hit)
        {
            double best = std::numeric_limits<double>::infinity();
            bool found = false;

            // Goes over all variables.
            for (Ineq const& c : J) {
                double rk = (double)r(c.k);
                if (std::abs(rk) < config.ray_tolerance) continue;

                double tr = c.is_upper ? rk : -rk;
                if (tr <= config.ray_tolerance) continue;

                double rhs = row_rhs(c);
                double tz = row_value(c, z);
                double t = (rhs-tz)/tr;

                if (t < 0.0) continue;

                if (!found || t < best) {
                    best = t;
                    hit = c;
                    found = true;
                }
            }
            
            return found;
        }

        // Tests whether the side ineq is redundant given the essential set I. The model
        // already carries I, so only the tested constraint is temporarily applied.
        //
        // The tested bound is relaxed by `config.relaxation` rather than removed, and
        // if the derived solution x* is feasible for the original LP, then the constraint
        // is marked as redundant.
        // @param ineq the constraint to be tested
        // @param solved a variable tracking if the LP failed
        // @return whether ineq is redundant, and the LP optimum
        std::pair<bool, VT> test_redundancy(Ineq const& ineq, bool & solved)
        {   
            const VT& b_u = P.getUpperBounds();
            const VT& b_l = P.getLowerBounds();
            unsigned d = P.getDimension();

            double old_u = highs.getLp().col_upper_[ineq.k];
            double old_l = highs.getLp().col_lower_[ineq.k];
            double u = ineq.is_upper ? (double)b_u(ineq.k)+config.relaxation_gap : old_u;
            double l = !ineq.is_upper ? (double)b_l(ineq.k)-config.relaxation_gap : old_l;

            highs.changeColBounds((HighsInt)ineq.k, l, u);
            highs.changeColCost((HighsInt)ineq.k, 1.0);
            highs.changeObjectiveSense(ineq.is_upper ? ObjSense::kMaximize : ObjSense::kMinimize);
            
            solved = run_lp();

            VT x_star(d);
            bool redundant = false;
            if (solved) {
                const auto& sol = highs.getSolution().col_value;
                for (unsigned j = 0; j < d; ++j)
                    x_star(j) = (typename VT::Scalar)sol[j];

                redundant = row_value(ineq, x_star) <= row_rhs(ineq)+config.facet_tolerance;
            }

            highs.changeColBounds((HighsInt)ineq.k, old_l, old_u);
            highs.changeColCost((HighsInt)ineq.k, 0.0);

            return {redundant, x_star};
        }

        // Converts every variable that the inequalities and bounds pin to
        // a single value, into an equality. Repeats full passes over all variables until one changes nothing.
        // @return the polytope with the degenerate dimensions moved into A_eq
        MetabolicPolytope<Point> fix_degenerate_dimensions_naive() {
            MetabolicPolytope<Point> tP = P;
            unsigned const& d = P.getDimension();

            if (!run_lp()) return tP;

            bool changed = true;

            while (changed) {
                changed = false;

                for (unsigned k = 0; k < d; ++k) {
                    double l = highs.getLp().col_lower_[k];
                    double u = highs.getLp().col_upper_[k];

                    if (l > -kHighsInf && u < kHighsInf && std::abs(u-l) < config.dim_tolerance) {
                        fix_dimension(k, (NT)((l+u)/2.0));
                        changed = true;
                        continue;
                    }

                    if (l <= -kHighsInf && u >= kHighsInf) continue;

                    highs.changeColCost((HighsInt)k, 1.0);
                    highs.changeObjectiveSense(ObjSense::kMaximize);

                    if (!run_lp()) {
                        highs.changeColCost((HighsInt)k, 0.0);
                        continue;
                    }

                    double max_val = highs.getObjectiveValue();
                    highs.changeObjectiveSense(ObjSense::kMinimize);
                    if (!run_lp()) {
                        highs.changeColCost((HighsInt)k, 0.0);
                        continue;
                    }
                    double min_val = highs.getObjectiveValue();
                    highs.changeColCost((HighsInt)k, 0.0);

                    if (std::abs(max_val-min_val) < config.dim_tolerance) {
                        fix_dimension(k, (NT)((max_val+min_val)/2.0));
                        changed = true;
                    }
                }
            }

            build_polytope_from_highs(highs, tP);
            return tP;
        }

        // Converts every variable that the inequalities and bounds pin to
        // a single value, into an equality, in a single pass. A variable whose
        // values seen across the LP solutions already differ is known to vary, 
        // so its LPs are skipped. This is the variant simplify uses.
        // @return the polytope with the degenerate dimensions moved into A_eq
        MetabolicPolytope<Point> fix_degenerate_dimensions_observe() 
        {   
            MetabolicPolytope<Point> tP = P;
            unsigned const& d = P.getDimension();

            min_observed.assign(d, std::numeric_limits<double>::infinity());
            max_observed.assign(d, -std::numeric_limits<double>::infinity());

            auto observe = [&]() {
                const auto& sol = highs.getSolution().col_value;
                for (unsigned j = 0; j < d; ++j) {
                    if (sol[j] > max_observed[j]) max_observed[j] = sol[j];
                    if (sol[j] < min_observed[j]) min_observed[j] = sol[j];
                }
            };

            auto observed_variation = [&](unsigned k) {
                return std::abs(max_observed[k]-min_observed[k]) > config.observe_tolerance;
            };

            if (!run_lp()) return tP;
            observe();

            for (unsigned k = 0; k < d; ++k) {
                double l = highs.getLp().col_lower_[k];
                double u = highs.getLp().col_upper_[k];

                if (l > -kHighsInf && u < kHighsInf && std::abs(u-l) < config.dim_tolerance) {
                    fix_dimension(k, (NT)((l+u)/2.0));
                    continue;
                }

                if (l <= -kHighsInf && u >= kHighsInf) continue;

                if (observed_variation(k)) continue;

                highs.changeColCost((HighsInt)k, 1.0);
                highs.changeObjectiveSense(ObjSense::kMaximize);

                if (!run_lp()) {
                    highs.changeColCost((HighsInt)k, 0.0);
                    continue;
                }

                double max_val = highs.getObjectiveValue();
                observe();

                if (observed_variation(k)) {
                    highs.changeColCost((HighsInt)k, 0.0);
                    continue;
                }

                highs.changeObjectiveSense(ObjSense::kMinimize);

                if (!run_lp()) {
                    highs.changeColCost((HighsInt)k, 0.0);
                    continue;
                }

                double min_val = highs.getObjectiveValue();
                observe();

                highs.changeColCost((HighsInt)k, 0.0);

                if (std::abs(max_val - min_val) < config.dim_tolerance) {
                    fix_dimension(k, (NT)((max_val+min_val)/2.0));
                }
            }

            build_polytope_from_highs(highs, tP);
            return tP;
        }

        // Frees both bounds of a variable in the LP model.
        // @param k the variable index
        void relax_variable(unsigned k) {
            highs.changeColBounds((HighsInt)k, -kHighsInf, kHighsInf);
        }

        // Finds the variables that the equalities alone pin to a single value, and relaxes
        // their bounds so that later passes skip them.
        void presolve() {
            auto pre = solve_homogeneous_presolve_spqr(P.getEqualities());
            
            report.rank = pre.rank;
            report.pinned_by_equalities = (unsigned)pre.pinned.size();

            for (unsigned k : pre.pinned)
                relax_variable(k);
        }

        // Minimizes the equality system by dropping linear dependent rows.
        // @param Ps the polytope to minimize
        // @return false if no feasible point was found or a dropped row was not implied
        bool remove_dependent_rows_from_polytope(MetabolicPolytope<Point> & Ps) {
            MT const& A = Ps.getEqualities();
            VT const& b = Ps.getEqualityBounds();
            unsigned const d = Ps.getDimension();
            unsigned const m = (unsigned)A.rows();

            report.rows_before_reduction = m;
            report.rows_after_reduction = m;

            if (!run_lp()) {
                std::cerr << "ClarksonSimplifier: the model is infeasible or HiGHS failed" 
                          << std::endl;
                
                return false;
            }

            const auto& sol = highs.getSolution().col_value;
            Eigen::VectorXd x0(d);

            for (unsigned j = 0; j < d; ++j)
                x0(j) = sol[j];

            std::vector<unsigned> independent_rows_id = find_independent_rows_spqr(A);

            std::vector<bool> kept_rows(m, false);
            
            for (unsigned i : independent_rows_id)
                kept_rows[i] = true;

            Eigen::SparseMatrix<double, Eigen::RowMajor> Ad = A.template cast<double>();
            Eigen::VectorXd res = Ad*x0;

            double worst = 0.0;
            for (unsigned i = 0; i < m; ++i) 
                if (!kept_rows[i]) 
                    worst = std::max(worst, std::abs(res(i)-(double)b(i)));

            report.dropped_row_residual = worst;

            if (worst > config.residual_tolerance) {
                std::cerr << "ClarksonSimplifier: a dropped row is violated by " << worst
                          << ", lower the rank tolerance" << std::endl;

                return false;
            }

            MT Amin;
            VT bmin;

            select_rows(A, b, independent_rows_id, Amin, bmin);

            report.rows_after_reduction = (unsigned)independent_rows_id.size();
            report.row_reduction_applied = true;

            Ps = MetabolicPolytope<Point>(d, Amin, Ps.getLowerBounds(), Ps.getUpperBounds(), bmin);

            return true;
        }


        // Keeps only the listed rows of the equality system.
        // Keeps only the rows of the equality matrix listed in kept_rows.
        // @param A the matrix to slice
        // @param b the right hand side vector of A
        // @param kept_rows the indices of the rows to keep
        // @param Amin is the sliced matrix
        // @param bmin is the sliced right hand side
        void select_rows(MT const& A, VT const& b, std::vector<unsigned> const& kept_rows,
                         MT& Amin, VT& bmin)
        {
            std::vector<int> maps_rows((std::size_t)A.rows(), -1);
            for (unsigned i = 0; (unsigned)i < kept_rows.size(); ++i)
                maps_rows[kept_rows[i]] = (int)i;

            std::vector<Eigen::Triplet<typename MT::Scalar>> triplets;
            bmin.resize((Eigen::Index)kept_rows.size());

            for (unsigned i = 0; (unsigned)i < A.rows(); ++i) {
                if (maps_rows[i] < 0) continue;

                bmin(maps_rows[i]) = b(i);
                for (typename MT::InnerIterator it(A, i); it; ++it)
                    triplets.emplace_back(maps_rows[i], it.col(), it.value());
            }

            Amin.resize((Eigen::Index)kept_rows.size(), A.cols());
            Amin.setFromTriplets(triplets.begin(), triplets.end());
        }



        // Finds a point in the interior of P by maximizing a uniform slack variable against all bounds.
        //
        // The LP solved is the following:
        //
        // max y s.t. A_eq x = b_eq, b_l+y <= x <= b_u-y, 0 <= y <= 1
        // Note: the point is stored in z
        // @return true if a point with significant slack was found
        bool find_interior_point()
        {
            const MT& A_eq = P.getEqualities();
            const VT& b_eq = P.getEqualityBounds();
            const VT& b_l = P.getLowerBounds();
            const VT& b_u = P.getUpperBounds();
            unsigned d = P.getDimension();

            Highs slack_highs;
            configure_highs(slack_highs, config);

            for (unsigned j = 0; j < d; ++j) {
                slack_highs.addVar(-kHighsInf, kHighsInf);
            }
            slack_highs.addVar(0.0, 1.0);

            for (unsigned i = 0; i < (unsigned)A_eq.rows(); ++i) {
                std::vector<HighsInt> indices;
                std::vector<double> values;
                for (typename MT::InnerIterator it(A_eq, i); it; ++it) {
                    indices.push_back((HighsInt)it.col());
                    values.push_back((double)it.value());
                }
                slack_highs.addRow((double)b_eq(i), (double)b_eq(i), indices.size(), indices.data(), values.data());
            }

            for (unsigned j = 0; j < d; ++j) {
                if (!std::isinf((double)b_l(j))) {
                    HighsInt idx[2] = {(HighsInt)j, (HighsInt)d};
                    double val[2] = {1.0, -1.0};
                    slack_highs.addRow((double)b_l(j), kHighsInf, 2, idx, val);
                }
                if (!std::isinf((double)b_u(j))) {
                    HighsInt idx[2] = {(HighsInt)j, (HighsInt)d};
                    double val[2] = {1.0, 1.0};
                    slack_highs.addRow(-kHighsInf, (double)b_u(j), 2, idx, val);
                }
            }

            slack_highs.changeColCost(d, 1.0);
            slack_highs.changeObjectiveSense(ObjSense::kMaximize);
            slack_highs.run();

            ++lp_stats.solves;
            lp_stats.iterations += slack_highs.getInfo().simplex_iteration_count;

            if (slack_highs.getModelStatus() != HighsModelStatus::kOptimal) {
                ++lp_stats.failures;
                return false;
            }

            report.interior_slack = slack_highs.getObjectiveValue();

            if (report.interior_slack < config.interior_tolerance) {
                return false;
            }

            const auto& sol = slack_highs.getSolution().col_value;
            z.resize(d);
            for (unsigned j = 0; j < d; ++j)
                z(j) = (typename VT::Scalar)sol[j];
            
            return true;
        }

        // Removes redundant inequalities from the representation using Clarkson's algorithm.
        //
        // The model starts with every inequality relaxed and gains them back one at a time
        // as they are proved essential, so every LP is solved against the essential set I 
        // found so far rather than the full set of inequalities, keeping the LP sizes at a minimum.
        // @return the simplified polytope
        MetabolicPolytope<Point> redundancy_removal_clarkson()
        {
            const NT INF = std::numeric_limits<NT>::infinity();
            unsigned d = P.getDimension();
            const VT& b_l = P.getLowerBounds();
            const VT& b_u = P.getUpperBounds();

            // Starts with all inequalities relaxed.
            for (unsigned j = 0; j < d; ++j)
                highs.changeColBounds((HighsInt)j, -kHighsInf, kHighsInf);

            // Holds the inequalities with unknown redundancy status.
            std::vector<Ineq> J;
            std::vector<int> pos(2*d, -1);

            auto J_insert_const = [&](Ineq c) {
                pos[c.map()] = (int)J.size();
                J.push_back(c);
            };

            auto J_erase_const = [&](Ineq c) {
                int p = pos[c.map()];
                if (p < 0) return false;

                Ineq last = J.back();
                J[p] = last;
                pos[last.map()] = p;
                J.pop_back();
                pos[c.map()] = -1;
                return true;
            };


            for (unsigned k = 0; k < d; ++k) {
                if (!std::isinf((double)b_l(k))) J_insert_const(Ineq{k, false});
                if (!std::isinf((double)b_u(k))) J_insert_const(Ineq{k, true});
            }

            std::vector<Ineq> I;

            std::mt19937 rng(config.clarkson_seed);
            std::vector<unsigned> fail_count(2*d, 0);


            while (!J.empty()) {
                // Picks constraints at random to make progress when LPs fail.
                std::uniform_int_distribution<std::size_t> pick(0, J.size()-1);
                Ineq k_ineq = J[pick(rng)];

                bool solved = false;
                auto [is_redundant, x_star] = test_redundancy(k_ineq, solved);

                if (!solved) {
                    if (++fail_count[k_ineq.map()] > config.failed_iter_count) {
                        report.untested_bounds = (unsigned)J.size();
                        I.insert(I.end(), J.begin(), J.end());
                        break;
                    }
                    continue;
                }

                if (is_redundant) {
                    J_erase_const(k_ineq);
                    continue;
                }

                Ineq hit;
                if (!ray_shoot(J, x_star-z, hit) || !J_erase_const(hit)) {
                    I.push_back(k_ineq);
                    enforce_ineq(k_ineq);
                    J_erase_const(k_ineq);
                } else {
                    I.push_back(hit);
                    enforce_ineq(hit);
                }
            }

            std::vector<bool> keep_lo(d, 0), keep_hi(d, 0);
            for (Ineq const& in : I) {
                if (in.is_upper) keep_hi[in.k] = true;
                else keep_lo[in.k] = true;
            }

            VT b_l_new(d), b_u_new(d);
            for (unsigned j = 0; j < d; ++j) {
                b_l_new(j) = keep_lo[j] ? b_l(j) : -INF;
                b_u_new(j) = keep_hi[j] ? b_u(j) : INF;
            }

            return MetabolicPolytope<Point>(d, P.getEqualities(), b_l_new, b_u_new, 
                                            P.getEqualityBounds());
        }
};
#endif