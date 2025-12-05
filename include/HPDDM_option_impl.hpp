/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2015-07-26

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
                 2016-     Centre National de la Recherche Scientifique

   HPDDM is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   HPDDM is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with HPDDM.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HPDDM_OPTION_IMPL_HPP_
#define HPDDM_OPTION_IMPL_HPP_

#ifndef HPDDM_NO_REGEX
  #define HPDDM_REGEX_LEVEL "level_([2-9]|[1-9]\\d+)_"
#else
  #define HPDDM_REGEX_LEVEL ""
#endif

#if !HPDDM_PETSC
namespace HPDDM
{
template <int N>
inline Option::Option(Singleton::construct_key<N>)
{
  app_ = nullptr;
}
template <bool recursive, bool exact, class Container>
inline int Option::parse(std::vector<std::string> &args, bool display, const Container &reg, std::string prefix)
{
  if (args.size() == 0 && reg.size() == 0) return 0;
  std::vector<std::tuple<std::string, std::string, std::function<bool(std::string &, const std::string &, bool)>>> option{std::forward_as_tuple("help", "Display available options", Arg::anything),
                                                                                                                          std::forward_as_tuple("version", "Display information about HPDDM", Arg::anything),
                                                                                                                          std::forward_as_tuple("config_file=<input_file>", "Load options from a file saved on disk", Arg::argument),
                                                                                                                          std::forward_as_tuple("tol=<1.0e-6>", "Relative decrease in residual norm", Arg::numeric),
                                                                                                                          std::forward_as_tuple("max_it=<100>", "Maximum number of iterations", Arg::positive),
                                                                                                                          std::forward_as_tuple("verbosity(=<integer>)", "Level of output (higher means more displayed information)", Arg::anything),
                                                                                                                          std::forward_as_tuple("compute_residual=(l2|l1|linfty)", "Print the residual after convergence", Arg::argument),
                                                                                                                          std::forward_as_tuple("push_prefix", "Prepend the according prefix for all following options (use -" + std::string(HPDDM_PREFIX) + "pop_prefix when done)", Arg::anything),
                                                                                                                          std::forward_as_tuple("reuse_preconditioner=(0|1)", "Do not factorize again the local matrices when solving subsequent systems", Arg::argument),
                                                                                                                          std::forward_as_tuple("operator_spd=(0|1)", "Assume the operator is symmetric positive definite", Arg::argument),
                                                                                                                          std::forward_as_tuple("orthogonalization=(cgs|mgs)", "Classical (faster) or Modified (more robust) Gram--Schmidt process", Arg::argument),
  #ifndef HPDDM_NO_REGEX
                                                                                                                          std::forward_as_tuple("dump_matri(ces|x_[[:digit:]]+)=<output_file>", "Save either one or all local matrices to disk", Arg::argument),
    #if defined(EIGENSOLVER) || HPDDM_FETI || HPDDM_BDD
                                                                                                                          std::forward_as_tuple("dump_eigenvectors(_[[:digit:]]+)?=<output_file>", "Save either one or all local eigenvectors to disk", Arg::argument),
    #endif
  #else
        std::forward_as_tuple("dump_matrices=<output_file>", "Save all local matrices to disk", Arg::argument),
    #if defined(EIGENSOLVER) || HPDDM_FETI || HPDDM_BDD
        std::forward_as_tuple("dump_eigenvectors=<output_file>", "Save all local eigenvectors to disk", Arg::argument),
    #endif
  #endif
                                                                                                                          std::forward_as_tuple("krylov_method=(gmres|bgmres|cg|bcg|gcrodr|bgcrodr|bfbcg|richardson|none)", "(Block) Generalized Minimal Residual Method, (Breakdown-Free Block) Conjugate Gradient, (Block) Generalized Conjugate Residual Method With Inner Orthogonalization and Deflated Restarting, or Richardson iterations", Arg::argument),
                                                                                                                          std::forward_as_tuple("enlarge_krylov_subspace=<val>", "Split the initial right-hand side into multiple vectors", Arg::positive),
                                                                                                                          std::forward_as_tuple("gmres_restart=<40>", "Maximum number of Arnoldi vectors generated per cycle", Arg::positive),
                                                                                                                          std::forward_as_tuple("variant=(left|right|flexible)", "Left, right, or variable preconditioning", Arg::argument),
                                                                                                                          std::forward_as_tuple("qr=(cholqr|cgs|mgs)", "Distributed QR factorizations computed with Cholesky QR, Classical or Modified Gram--Schmidt process", Arg::argument),
                                                                                                                          std::forward_as_tuple("deflation_tol=<val>", "Tolerance when deflating right-hand sides inside block methods", Arg::numeric),
                                                                                                                          std::forward_as_tuple("recycle=<val>", "Number of harmonic Ritz vectors to compute", Arg::positive),
                                                                                                                          std::forward_as_tuple("recycle_same_system=(0|1)", "Assume the system is the same as the one for which Ritz vectors have been computed", Arg::argument),
                                                                                                                          std::forward_as_tuple("recycle_strategy=(A|B)", "Generalized eigenvalue problem to solve for recycling", Arg::argument),
                                                                                                                          std::forward_as_tuple("recycle_target=(SM|LM|SR|LR|SI|LI)", "Criterion to select harmonic Ritz vectors", Arg::argument),
                                                                                                                          std::forward_as_tuple("richardson_damping_factor=<1.0>", "Damping factor used in Richardson iterations", Arg::numeric),
  #if HPDDM_SCHWARZ
                                                                                                                          std::forward_as_tuple("", "",
                                                                                                                                                [](std::string &, const std::string &, bool) {
                                                                                                                                                  std::cout << "\n Overlapping Schwarz methods options:";
                                                                                                                                                  return true;
                                                                                                                                                }),
                                                                                                                          std::forward_as_tuple("schwarz_method=(ras|oras|soras|asm|osm|none)", "Symmetric or not, Optimized or Additive, Restricted or not", Arg::argument),
                                                                                                                          std::forward_as_tuple("schwarz_coarse_correction=(deflated|additive|balanced)", "Switch to a multilevel preconditioner", Arg::argument),
  #endif
  #if HPDDM_FETI || HPDDM_BDD
                                                                                                                          std::forward_as_tuple("", "",
                                                                                                                                                [](std::string &, const std::string &, bool) {
                                                                                                                                                  std::cout << "\n Substructuring methods options:";
                                                                                                                                                  return true;
                                                                                                                                                }),
                                                                                                                          std::forward_as_tuple("substructuring_scaling=(multiplicity|stiffness|coefficient)", "Type of scaling used for the preconditioner", Arg::argument),
  #endif
  #if defined(EIGENSOLVER) || HPDDM_FETI || HPDDM_BDD
                                                                                                                          std::forward_as_tuple("eigensolver_tol=<1.0e-6>", "Tolerance for computing eigenvectors by ARPACK or LAPACK", Arg::numeric),
                                                                                                                          std::forward_as_tuple("", "",
                                                                                                                                                [](std::string &, const std::string &, bool) {
                                                                                                                                                  std::cout << "\n GenEO options:";
                                                                                                                                                  return true;
                                                                                                                                                }),
                                                                                                                          std::forward_as_tuple("geneo_nu=<20>", "Number of local eigenvectors to compute for adaptive methods", Arg::integer),
                                                                                                                          std::forward_as_tuple("geneo_threshold=<eps>", "Threshold for selecting local eigenvectors for adaptive methods", Arg::numeric),
    #if defined(MUMPSSUB) || defined(MKL_PARDISOSUB)
                                                                                                                          std::forward_as_tuple("geneo_estimate_nu=(0|1)", "Estimate the number of eigenvalues below a threshold using the inertia of the stencil", Arg::argument),
    #endif
                                                                                                                          std::forward_as_tuple("geneo_force_uniformity=(min|max)", "Ensure that the number of local eigenvectors is the same for all subdomains", Arg::argument),
  #endif
  #ifdef MU_ARPACK
                                                                                                                          std::forward_as_tuple("", "",
                                                                                                                                                [](std::string &, const std::string &, bool) {
                                                                                                                                                  std::cout << "\n ARPACK-specific options:";
                                                                                                                                                  return true;
                                                                                                                                                }),
                                                                                                                          std::forward_as_tuple("arpack_ncv=<val>", "Number of Lanczos basis vectors generated in one iteration", Arg::integer),
  #endif
  #if defined(SUBDOMAIN) || defined(COARSEOPERATOR)
    #ifndef HPDDM_NO_REGEX
      #if defined(DMKL_PARDISO) || defined(MKL_PARDISOSUB)
                                                                                                                          std::forward_as_tuple("", "",
                                                                                                                                                [](std::string &, const std::string &, bool) {
                                                                                                                                                  std::cout << "\n MKL PARDISO-specific options:";
                                                                                                                                                  return true;
                                                                                                                                                }),
                                                                                                                          std::forward_as_tuple("mkl_pardiso_iparm_(2|1[013]|2[1457])=<val>", "Integer control parameters", Arg::integer),
      #endif
      #if defined(DMUMPS) || defined(MUMPSSUB)
                                                                                                                          std::forward_as_tuple("", "",
                                                                                                                                                [](std::string &, const std::string &, bool) {
                                                                                                                                                  std::cout << "\n MUMPS-specific options:";
                                                                                                                                                  return true;
                                                                                                                                                }),
                                                                                                                          std::forward_as_tuple("mumps_icntl_([678]|1[234]|2[34789]|3[567])=<val>", "Integer control parameters", Arg::integer),
                                                                                                                          std::forward_as_tuple("mumps_cntl_([123457])=<val>", "Real control parameters", Arg::numeric),
      #endif
    #endif
    #ifdef DHYPRE
                                                                                                                          std::forward_as_tuple("", "",
                                                                                                                                                [](std::string &, const std::string &, bool) {
                                                                                                                                                  std::cout << "\n Hypre-specific options:";
                                                                                                                                                  return true;
                                                                                                                                                }),
                                                                                                                          std::forward_as_tuple("hypre_solver=(fgmres|pcg|amg)", "Iterative method used by Hypre", Arg::argument),
                                                                                                                          std::forward_as_tuple("hypre_tol=<1.0e-12>", "Relative convergence tolerance", Arg::numeric),
                                                                                                                          std::forward_as_tuple("hypre_max_it=<500>", "Maximum number of iterations", Arg::positive),
                                                                                                                          std::forward_as_tuple("hypre_gmres_restart=<100>", "Maximum size of the Krylov subspace when using FlexGMRES", Arg::positive),
                                                                                                                          std::forward_as_tuple("boomeramg_num_sweeps=<1>", "Number of sweeps", Arg::positive),
                                                                                                                          std::forward_as_tuple("boomeramg_max_levels=<10>", "Maximum number of multigrid levels", Arg::positive),
      #ifndef HPDDM_NO_REGEX
                                                                                                                          std::forward_as_tuple("boomeramg_coarsen_type=([0136-9]|1[01]|2[12])", "Parallel coarsening algorithm", Arg::integer),
                                                                                                                          std::forward_as_tuple("boomeramg_relax_type=([0-9]|1[5-8])", "Smoother", Arg::integer),
                                                                                                                          std::forward_as_tuple("boomeramg_interp_type=([0-9]|1[0-4])", "Parallel interpolation operator", Arg::integer),
      #endif
    #endif
    #ifdef DISSECTIONSUB
                                                                                                                          std::forward_as_tuple("", "",
                                                                                                                                                [](std::string &, const std::string &, bool) {
                                                                                                                                                  std::cout << "\n Dissection-specific options:";
                                                                                                                                                  return true;
                                                                                                                                                }),
                                                                                                                          std::forward_as_tuple("dissection_pivot_tol=<val>", "Tolerance for choosing when to pivot during numerical factorizations", Arg::numeric),
                                                                                                                          std::forward_as_tuple("dissection_kkt_scaling=(0|1)", "Turn on KKT scaling instead of the default diagonal scaling", Arg::argument),
    #endif
                                                                                                                          std::forward_as_tuple("", "", Arg::anything),
    #if !defined(DSUITESPARSE) && !defined(DLAPACK)
                                                                                                                          std::forward_as_tuple(std::string(HPDDM_REGEX_LEVEL) + "p=<1>", "Number of main processes", Arg::positive),
      #if defined(DMUMPS) && !HPDDM_INEXACT_COARSE_OPERATOR
                                                                                                                          std::forward_as_tuple(std::string(HPDDM_REGEX_LEVEL) + "distribution=(centralized|sol)", "Distribution of coarse right-hand sides and solution vectors", Arg::argument),
      #endif
                                                                                                                          std::forward_as_tuple(std::string(HPDDM_REGEX_LEVEL) + "topology=(0|" +
      #if !defined(HPDDM_CONTIGUOUS)
                                                                                                                                                  std::string("1|") +
      #endif
                                                                                                                                                  std::string("2)"),
                                                                                                                                                "Distribution of the main processes", Arg::integer),
    #endif
                                                                                                                          std::forward_as_tuple(std::string(HPDDM_REGEX_LEVEL) + "assembly_hierarchy=<val>", "Hierarchy used for the assembly of the coarse operator", Arg::positive),
    #if HPDDM_INEXACT_COARSE_OPERATOR
                                                                                                                          std::forward_as_tuple(std::string(HPDDM_REGEX_LEVEL) + "aggregate_size=<val>", "Number of main processes per MPI sub-communicators", Arg::positive),
    #endif
                                                                                                                          std::forward_as_tuple(std::string(HPDDM_REGEX_LEVEL) + "dump_matrix=<output_file>", "Save the coarse operator to disk", Arg::argument),
                                                                                                                          std::forward_as_tuple(std::string(HPDDM_REGEX_LEVEL) + "exclude=(0|1)", "Exclude the main processes from the domain decomposition", Arg::argument)
  #endif
  };

  if (reg.size() != 0) {
    if (!app_) app_ = new std::unordered_map<std::string, double>;
    app_->reserve(reg.size());
    for (const auto &x : reg) {
      std::string            def = std::get<0>(x);
      std::string::size_type n   = def.find("=");
      if (n != std::string::npos && n + 2 < def.size()) {
        std::string val = def.substr(n + 2, def.size() - n - 3);
        def             = def.substr(0, n);
        if (std::get<2>(x)(def, val, true)) {
  #if __cpp_rtti || defined(__GXX_RTTI) || defined(__INTEL_RTTI__) || defined(_CPPRTTI)
          auto target = std::get<2>(x).template target<bool (*)(const std::string &, const std::string &, bool)>();
          if (!target || (*target != Arg::argument)) (*app_)[def] = sto<double>(val);
  #else
          (*app_)[def] = sto<double>(val);
  #endif
        }
      }
    }
  }
  std::stack<std::string> pre;
  std::string             p = std::string(HPDDM_PREFIX) + (exact ? prefix : "");
  for (std::vector<std::string>::const_iterator itArg = args.cbegin(); itArg < args.cend(); ++itArg) {
    if (pre.empty()) p = std::string(HPDDM_PREFIX) + (exact ? prefix : "");
    std::string::size_type n = itArg->find("-" + (pre.empty() ? p : std::string(HPDDM_PREFIX)));
    if (n == std::string::npos) {
      if (reg.size() != 0) {
        n = itArg->find_first_not_of("-");
        if (n != 0 && n != std::string::npos && insert<0>(reg, itArg->substr(n), itArg + 1 != args.cend() ? *(itArg + 1) : "")) ++itArg;
      }
    } else if (itArg->substr(0, n).find_first_not_of("-") == std::string::npos) {
      std::string opt    = itArg->substr(n + 1 + (pre.empty() ? p.size() : std::string(HPDDM_PREFIX).size()));
      bool        ending = hasEnding(opt, "_prefix");
      if (!ending && insert<true, exact>(option, (!pre.empty() ? p : "") + opt, itArg + 1 != args.cend() ? *(itArg + 1) : "", prefix)) ++itArg;
      else if (ending) {
        opt = opt.substr(0, opt.size() - 7);
        if (hasEnding(opt, "push")) {
          opt = (exact ? prefix : opt.substr(0, opt.size() - 4));
          p   = (pre.empty() ? "" : p) + opt;
          pre.push(opt);
        } else if (opt.compare("pop") == 0) {
          if (!pre.empty()) {
            p = p.substr(0, p.size() - pre.top().size());
            pre.pop();
          } else std::cout << "WARNING -- there is no prefix to pop right now" << std::endl;
        } else std::cout << "WARNING -- '-" << std::string(HPDDM_PREFIX) << (!pre.empty() ? p : "") + opt << "' is not a registered HPDDM option" << std::endl;
      }
    }
  }
  if (!recursive) {
    for (const auto &x : opt_) {
      const std::string key = x.first;
      const double      val = x.second;
      if (val < -10000000 && key[-val - 10000000] == '#' && hasEnding(key.substr(0, -val - 10000000), "config_file")) {
        std::ifstream cfg(key.substr(-val - 10000000 + 1));
        parse(cfg, display);
      }
    }
    opt_.rehash(opt_.size());
  }
  if (pre.size() > 0) std::cout << "WARNING -- too many prefixes have been pushed" << std::endl;
  if (display && opt_.find("help") != opt_.cend()) {
    size_t max = 0;
    size_t col = getenv("COLUMNS") ? sto<int>(std::getenv("COLUMNS")) : 200;
    for (const auto &x : option) max = std::max(max, std::get<0>(x).size() + std::string(HPDDM_PREFIX).size());
    std::function<void(const std::string &)> wrap = [&](const std::string &text) {
      if (text.size() + max + 4 < col) std::cout << text << std::endl;
      else {
        std::istringstream words(text);
        std::ostringstream wrapped;
        std::string        word;
        size_t             line_length = std::max(10, static_cast<int>(col - max - 4));
        if (words >> word) {
          wrapped << word;
          size_t space_left = line_length - word.length();
          while (words >> word) {
            if (space_left < word.length() + 1) {
              wrapped << std::endl << std::left << std::setw(max + 4) << "" << word;
              space_left = std::max(0, static_cast<int>(line_length - word.length()));
            } else {
              wrapped << ' ' << word;
              space_left -= word.length() + 1;
            }
          }
        }
        std::cout << wrapped.str() << std::endl;
      }
    };
    if (reg.size() != 0) {
      for (const auto &x : reg) max = std::max(max, std::get<0>(x).size());
      std::cout << "Application-specific options:" << std::endl;
      for (const auto &x : reg) {
        std::string s = "  -" + std::get<0>(x);
        if (s.size() > 3) std::cout << std::left << std::setw(max + 4) << s;
        else {
          std::get<2>(x)(s, s, true);
          std::cout << std::setw(max + 4) << "";
        }
        wrap(std::get<1>(x));
      }
      std::cout << std::endl;
    }
    std::cout << "HPDDM options:" << std::endl;
    for (const auto &x : option) {
      std::string s = "  -" + std::string(HPDDM_PREFIX) + std::get<0>(x);
      if (s.size() > 3 + std::string(HPDDM_PREFIX).size()) std::cout << std::left << std::setw(max + 4) << s;
      else {
        std::get<2>(x)(s, s, true);
        std::cout << std::setw(max + 4) << "";
      }
      wrap(std::get<1>(x));
    }
  }
  return 0;
}
inline void Option::version() const
{
  std::vector<std::string> v = {" ┌",
                                " │ HPDDM compilation options: ",
                                " │  HPDDM version: " + std::string(HPDDM_VERSION),
  #ifdef PY_VERSION
                                " │  Python.h version: " PY_VERSION,
  #endif
                                " │  epsilon: " + std::string(HPDDM_STR(HPDDM_EPS)),
                                " │  penalization: " + std::string(HPDDM_STR(HPDDM_PEN)),
                                " │  OpenMP granularity: " + std::string(HPDDM_STR(HPDDM_GRANULARITY)),
                                " │  OpenMP activated: "
  #ifdef _OPENMP
                                "true",
  #else
                                "false",
  #endif
                                " │  numbering: '" + std::string(1, HPDDM_NUMBERING) + "'",
                                " │  regular expression support: "
  #ifdef HPDDM_NO_REGEX
                                "false",
  #else
                                "true",
  #endif
                                " │  C++ RTTI support: "
  #if __cpp_rtti || defined(__GXX_RTTI) || defined(__INTEL_RTTI__) || defined(_CPPRTTI)
                                "true",
  #else
                                "false",
  #endif
                                " │  MPI support: " + std::string(bool(HPDDM_MPI) ? "true" : "false"),
                                " │  MKL support: " + std::string(bool(HPDDM_MKL) ? "true" : "false"),
  #ifdef INTEL_MKL_VERSION
                                " │  MKL version: " + to_string(INTEL_MKL_VERSION),
  #endif
                                " │  Schwarz module activated: " + std::string(bool(HPDDM_SCHWARZ) ? "true" : "false"),
                                " │  FETI module activated: " + std::string(bool(HPDDM_FETI) ? "true" : "false"),
                                " │  BDD module activated: " + std::string(bool(HPDDM_BDD) ? "true" : "false"),
                                " │  Dense module activated: " + std::string(bool(HPDDM_DENSE) ? "true" : "false"),
                                " │  PETSc module activated: " + std::string(bool(HPDDM_PETSC) ? "true" : "false"),
                                " │  PETSc compiled with SLEPc: " + std::string(bool(HPDDM_SLEPC) ? "true" : "false"),
                                " │  Inexact coarse spaces: " + std::string(bool(HPDDM_INEXACT_COARSE_OPERATOR) ? "true" : "false"),
                                " │  QR algorithm: " + std::string(HPDDM_STR(HPDDM_QR)),
                                " │  asynchronous collectives: " + std::string(bool(HPDDM_ICOLLECTIVE) ? "true" : "false"),
                                " │  mixed precision arithmetic: " + std::string(bool(HPDDM_MIXED_PRECISION) ? "true" : "false"),
                                " │  subdomain solver: " + std::string(HPDDM_STR(SUBDOMAIN)),
                                " │  coarse operator solver: " + std::string(HPDDM_STR(COARSEOPERATOR)),
                                " │  eigensolver: " + std::string(HPDDM_STR(EIGENSOLVER)),
                                " └"};
  size_t max = 0;
  for (const auto &x : v) max = std::max(max, x.size());
  output(v, max);
}
} // namespace HPDDM
#endif
#endif // HPDDM_OPTION_IMPL_HPP_
