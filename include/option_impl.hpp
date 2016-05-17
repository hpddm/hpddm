/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
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

#ifndef _HPDDM_OPTION_IMPL_
#define _HPDDM_OPTION_IMPL_

namespace HPDDM {
template<int N>
inline Option::Option(Singleton::construct_key<N>) {
    _app = nullptr;
    _opt = {
#if defined(SUBDOMAIN) || defined(COARSEOPERATOR)
#if HPDDM_SCHWARZ
             { "schwarz_method",                0 },
#endif
#if HPDDM_FETI || HPDDM_BDD
             { "substructuring_scaling",        0 },
#endif
             { "geneo_nu",                      20 },
#ifdef MUMPSSUB
             { "mumps_icntl_14",                80 },
#elif defined(MKL_PARDISOSUB)
             { "mkl_pardiso_iparm_2",           2 },
             { "mkl_pardiso_iparm_10",          13 },
             { "mkl_pardiso_iparm_21",          1 },
#endif
#ifdef DMUMPS
             { "master_mumps_icntl_18",         3 },
             { "master_mumps_icntl_14",         75 },
#elif defined(DHYPRE)
             { "master_hypre_tol",              1.0e-12 },
             { "master_hypre_max_it",           500 },
#elif defined(DMKL_PARDISO)
             { "master_mkl_pardiso_iparm_2",    2 },
             { "master_mkl_pardiso_iparm_10",   13 },
             { "master_mkl_pardiso_iparm_11",   1 },
#endif
#if defined(DHYPRE) || defined(DPASTIX)
             { "master_distribution",           2 }
#endif
#endif
    };
}
template<bool recursive, class Container>
inline int Option::parse(std::vector<std::string>& args, bool display, const Container& reg) {
    if(args.size() == 0 && reg.size() == 0)
        return 0;
    std::vector<std::tuple<std::string, std::string, std::function<bool(std::string&, const std::string&, bool)>>> option {
        std::forward_as_tuple("help", "Display available options.", Arg::anything),
        std::forward_as_tuple("version", "Display information about HPDDM.", Arg::anything),
        std::forward_as_tuple("config_file=<input_file>", "Load options from a file saved on disk.", Arg::argument),
        std::forward_as_tuple("tol=<1.0e-8>", "Relative decrease in residual norm.", Arg::numeric),
        std::forward_as_tuple("max_it=<100>", "Maximum number of iterations.", Arg::integer),
        std::forward_as_tuple("verbosity(=<integer>)", "Level of output (higher means more displayed information).", Arg::anything),
        std::forward_as_tuple("reuse_preconditioner=(0|1)", "Do not factorize again the local matrices when solving subsequent systems.", Arg::argument),
        std::forward_as_tuple("local_operators_not_spd=(0|1)", "Assume local operators are general symmetric (instead of symmetric or Hermitian positive definite).", Arg::argument),
        std::forward_as_tuple("orthogonalization=(cgs|mgs)", "Classical (faster) or Modified (more robust) Gram-Schmidt process.", Arg::argument),
        std::forward_as_tuple("dump_local_matri(ces|x_[[:digit:]]+)=<output_file>", "Save either one or all local matrices to disk.", Arg::argument),
        std::forward_as_tuple("krylov_method=(gmres|bgmres|cg|gcrodr|bgcrodr)", "(Block) Generalized Minimal Residual Method, Conjugate Gradient, or (Block) Generalized Conjugate Residual Method With Inner Orthogonalization and Deflated Restarting.", Arg::argument),
        std::forward_as_tuple("gmres_restart=<40>", "Maximum number of Arnoldi vectors generated per cycle.", Arg::integer),
        std::forward_as_tuple("variant=(left|right|flexible)", "Left, right, or variable preconditioning.", Arg::argument),
        std::forward_as_tuple("qr=(cholqr|cgs|mgs)", "Distributed QR factorizations computed with Cholesky QR, Classical or Modified Gram-Schmidt process.", Arg::argument),
        std::forward_as_tuple("initial_deflation_tol=<val>", "Tolerance when deflating right-hand sides inside Block GMRES or Block GCRODR.", Arg::numeric),
        std::forward_as_tuple("recycle=<val>", "Number of harmonic Ritz vectors to compute.", Arg::integer),
        std::forward_as_tuple("recycle_same_system=(0|1)", "Assume the system is the same as the one for which Ritz vectors have been computed.", Arg::argument),
        std::forward_as_tuple("recycle_strategy=(A|B)", "Generalized eigenvalue problem to solve for recycling.", Arg::argument),
        std::forward_as_tuple("recycle_target=(SM|LM|SR|LR|SI|LI)", "Criterion to select harmonic Ritz vectors.", Arg::argument),
#if HPDDM_SCHWARZ
        std::forward_as_tuple("", "", [](std::string&, const std::string&, bool) { std::cout << "\n Overlapping Schwarz methods options:"; return true; }),
        std::forward_as_tuple("schwarz_method=(ras|oras|soras|asm|osm|none)", "Symmetric or not, Optimized or Additive, Restricted or not.", Arg::argument),
        std::forward_as_tuple("schwarz_coarse_correction=(deflated|additive|balanced)", "Switch to a multilevel preconditioner.", Arg::argument),
#endif
#if HPDDM_FETI || HPDDM_BDD
        std::forward_as_tuple("", "", [](std::string&, const std::string&, bool) { std::cout << "\n Substructuring methods options:"; return true; }),
        std::forward_as_tuple("substructuring_scaling=(multiplicity|stiffness|coefficient)", "Type of scaling used for the preconditioner.", Arg::argument),
#endif
        std::forward_as_tuple("eigensolver_tol=<1.0e-6>", "Tolerance for computing eigenvectors by ARPACK or LAPACK.", Arg::numeric),
        std::forward_as_tuple("", "", [](std::string&, const std::string&, bool) { std::cout << "\n GenEO options:"; return true; }),
        std::forward_as_tuple("geneo_nu=<20>", "Number of local eigenvectors to compute for adaptive methods.", Arg::integer),
        std::forward_as_tuple("geneo_threshold=<eps>", "Threshold for selecting local eigenvectors for adaptive methods.", Arg::numeric),
#if defined(SUBDOMAIN) || defined(COARSEOPERATOR)
#if defined(DMKL_PARDISO) || defined(MKL_PARDISOSUB)
        std::forward_as_tuple("", "", [](std::string&, const std::string&, bool) { std::cout << "\n MKL PARDISO-specific options:"; return true; }),
#endif
#ifdef MKL_PARDISOSUB
        std::forward_as_tuple("mkl_pardiso_iparm_(2|8|1[013]|2[147])=<val>", "Integer control parameters of MKL PARDISO for the subdomain solvers.", Arg::integer),
#endif
#ifdef DMKL_PARDISO
        std::forward_as_tuple("master_mkl_pardiso_iparm_(2|1[013]|2[17])=<val>", "Integer control parameters of MKL PARDISO for the coarse operator solver.", Arg::integer),
#endif
#if defined(DMUMPS) || defined(MUMPSSUB)
        std::forward_as_tuple("", "", [](std::string&, const std::string&, bool) { std::cout << "\n MUMPS-specific options:"; return true; }),
#endif
#ifdef MUMPSSUB
        std::forward_as_tuple("mumps_icntl_([6-9]|[1-3][0-9]|40)=<val>", "Integer control parameters of MUMPS for the subdomain solvers.", Arg::integer),
#endif
#ifdef DMUMPS
        std::forward_as_tuple("master_mumps_icntl_([6-9]|[1-3][0-9]|40)=<val>", "Integer control parameters of MUMPS for the coarse operator solver.", Arg::integer),
#elif defined(DHYPRE)
        std::forward_as_tuple("", "", [](std::string&, const std::string&, bool) { std::cout << "\n Hypre-specific options:"; return true; }),
        std::forward_as_tuple("master_hypre_solver=(fgmres|pcg|amg)", "Solver used by Hypre to solve coarse systems.", Arg::argument),
        std::forward_as_tuple("master_hypre_tol=<1.0e-12>", "Relative convergence tolerance.", Arg::numeric),
        std::forward_as_tuple("master_hypre_max_it=<500>", "Maximum number of iterations.", Arg::integer),
        std::forward_as_tuple("master_hypre_gmres_restart=<100>", "Maximum size of the Krylov subspace when using FlexGMRES.", Arg::integer),
        std::forward_as_tuple("master_boomeramg_coarsen_type=<6>", "Parallel coarsening algorithm.", Arg::integer),
        std::forward_as_tuple("master_boomeramg_relax_type=<3>", "Smoother.", Arg::integer),
        std::forward_as_tuple("master_boomeramg_num_sweeps=<1>", "Number of sweeps.", Arg::integer),
        std::forward_as_tuple("master_boomeramg_max_levels=<10>", "Maximum number of multigrid levels.", Arg::integer),
        std::forward_as_tuple("master_boomeramg_interp_type=<0>", "Parallel interpolation operator.", Arg::integer),
#endif
#ifdef DISSECTIONSUB
        std::forward_as_tuple("", "", [](std::string&, const std::string&, bool) { std::cout << "\n Dissection-specific options:"; return true; }),
        std::forward_as_tuple("dissection_pivot_tol=<val>", "Tolerance for choosing when to pivot during numerical factorizations.", Arg::numeric),
        std::forward_as_tuple("dissection_kkt_scaling=(0|1)", "Turn on KKT scaling instead of the default diagonal scaling.", Arg::argument),
#endif
        std::forward_as_tuple("", "", Arg::anything),
#if !defined(DSUITESPARSE)
        std::forward_as_tuple("master_p=<1>", "Number of master processes.", Arg::integer),
#if !defined(DHYPRE)
        std::forward_as_tuple("master_distribution=(centralized|sol|sol_and_rhs)", "Distribution of coarse right-hand sides and solution vectors.", Arg::argument),
#endif
        std::forward_as_tuple("master_topology=(0|" +
#if !defined(HPDDM_CONTIGUOUS)
            std::string("1|") +
#endif
            std::string("2)"), "Distribution of the master processes.", Arg::integer),
#endif
        std::forward_as_tuple("master_filename=<output_file>", "Save the coarse operator to disk.", Arg::argument),
        std::forward_as_tuple("master_exclude=(0|1)", "Exclude the master processes from the domain decomposition.", Arg::argument)
#if defined(DMUMPS) || defined(DPASTIX) || defined(DMKL_PARDISO)
      , std::forward_as_tuple("master_not_spd=(0|1)", "Assume the coarse operator is general symmetric (instead of symmetric positive definite).", Arg::argument)
#endif
#endif
    };

    if(reg.size() != 0) {
        if(!_app)
            _app = new std::unordered_map<std::string, double>;
        _app->reserve(reg.size());
        for(const auto& x : reg) {
            std::string def = std::get<0>(x);
            std::string::size_type n = def.find("=");
            if(n != std::string::npos && n + 2 < def.size()) {
                std::string val = def.substr(n + 2, def.size() - n - 3);
                def = def.substr(0, n);
                if(std::get<2>(x)(def, val, true)) {
#if __cpp_rtti || defined(__GXX_RTTI) || defined(__INTEL_RTTI__) || defined(_CPPRTTI)
                    auto target = std::get<2>(x).template target<bool (*)(const std::string&, const std::string&, bool)>();
                    if(!target || (*target != Arg::argument))
                        (*_app)[def] = sto<double>(val);
#else
                    (*_app)[def] = sto<double>(val);
#endif
                }
            }
        }
    }
    for(std::vector<std::string>::const_iterator itArg = args.cbegin(); itArg != args.cend(); ++itArg) {
        std::string::size_type n = itArg->find("-" + std::string(HPDDM_PREFIX));
        if(n == std::string::npos) {
            if(reg.size() != 0) {
                n = itArg->find_first_not_of("-");
                if(n != std::string::npos) {
                    std::string opt = itArg->substr(n);
                    insert(*_app, reg, opt, itArg + 1 != args.cend() ? *(itArg + 1) : "");
                }
            }
        }
        else if(itArg->substr(0, n).find_first_not_of("-") == std::string::npos) {
            std::string opt = itArg->substr(n + 1 + std::string(HPDDM_PREFIX).size());
            insert(_opt, option, opt, itArg + 1 != args.cend() ? *(itArg + 1) : "");
        }
    }
    if(!recursive)
        for(const auto& x : _opt) {
            const std::string key = x.first;
            const double val = x.second;
            if(val < -10000000 && key[-val - 10000000] == '_' && key.substr(0, -val - 10000000) == "config_file") {
                std::ifstream cfg(key.substr(-val - 10000000 + 1));
                parse(cfg, display);
            }
        }
    _opt.rehash(_opt.size());
    if(display && _opt.find("help") != _opt.cend()) {
        size_t max = 0;
        int col = getenv("COLUMNS") ? sto<int>(std::getenv("COLUMNS")) : 200;
        for(const auto& x : option)
            max = std::max(max, std::get<0>(x).size() + std::string(HPDDM_PREFIX).size());
        std::function<void(const std::string&)> wrap = [&](const std::string& text) {
            if(text.size() + max + 4 < col)
                std::cout << text << std::endl;
            else {
                std::istringstream words(text);
                std::ostringstream wrapped;
                std::string word;
                size_t line_length = std::max(10, static_cast<int>(col - max - 4));
                if(words >> word) {
                    wrapped << word;
                    size_t space_left = line_length - word.length();
                    while (words >> word) {
                        if(space_left < word.length() + 1) {
                            wrapped << std::endl << std::left << std::setw(max + 4) << "" << word;
                            space_left = std::max(0, static_cast<int>(line_length - word.length()));
                        }
                        else {
                            wrapped << ' ' << word;
                            space_left -= word.length() + 1;
                        }
                    }
                }
                std::cout << wrapped.str() << std::endl;
            }
        };
        if(reg.size() != 0) {
            for(const auto& x : reg)
                max = std::max(max, std::get<0>(x).size());
            std::cout << "Application-specific options:" << std::endl;
            for(const auto& x : reg) {
                std::string s = "  -" + std::get<0>(x);
                if(s.size() > 3)
                    std::cout << std::left << std::setw(max + 4) << s;
                else {
                    std::get<2>(x)(s, s, true);
                    std::cout << std::setw(max + 4) << "";
                }
                wrap(std::get<1>(x));
            }
            std::cout << std::endl;
        }
        std::cout << "HPDDM options:" << std::endl;
        for(const auto& x : option) {
            std::string s = "  -" + std::string(HPDDM_PREFIX) + std::get<0>(x);
            if(s.size() > 3 + std::string(HPDDM_PREFIX).size())
                std::cout << std::left << std::setw(max + 4) << s;
            else {
                std::get<2>(x)(s, s, true);
                std::cout << std::setw(max + 4) << "";
            }
            wrap(std::get<1>(x));
        }
    }
    return 0;
}
inline void Option::version() const {
    std::vector<std::string> v = {
        " ┌",
        " │ HPDDM compilation options: ",
        " │  HPDDM version: " + std::string(HPDDM_STR(HPDDM_VERSION)),
#ifdef PY_VERSION
        " │  Python.h version: "  PY_VERSION,
#endif
        " │  epsilon: " + std::string(HPDDM_STR(HPDDM_EPS)),
        " │  penalization: " + std::string(HPDDM_STR(HPDDM_PEN)),
        " │  OpenMP granularity: " + std::string(HPDDM_STR(HPDDM_GRANULARITY)),
        " │  OpenMP activated? "
#ifdef _OPENMP
            "true",
#else
            "false",
#endif
        " │  numbering: '" + std::string(1, HPDDM_NUMBERING) + "'",
        " │  regular expression support? "
#ifdef HPDDM_NO_REGEX
            "false",
#else
            "true",
#endif
        " │  C++ RTTI support? "
#if __cpp_rtti || defined(__GXX_RTTI) || defined(__INTEL_RTTI__) || defined(_CPPRTTI)
            "true",
#else
            "false",
#endif
        " │  MKL support? " + std::string(bool(HPDDM_MKL) ? "true" : "false"),
#ifdef INTEL_MKL_VERSION
        " │  MKL version: " + to_string(INTEL_MKL_VERSION),
#endif
        " │  Schwarz module activated? " + std::string(bool(HPDDM_SCHWARZ) ? "true" : "false"),
        " │  FETI module activated? " + std::string(bool(HPDDM_FETI) ? "true" : "false"),
        " │  BDD module activated? " + std::string(bool(HPDDM_BDD) ? "true" : "false"),
        " │  QR algorithm: " + std::string(HPDDM_STR(HPDDM_QR)),
        " │  asynchronous collectives? " + std::string(bool(HPDDM_ICOLLECTIVE) ? "true" : "false"),
        " │  optimized matrix-vector products? " + std::string(bool(HPDDM_GMV) ? "true" : "false"),
        " │  subdomain solver: " + std::string(HPDDM_STR(SUBDOMAIN)),
        " │  coarse operator solver: " + std::string(HPDDM_STR(COARSEOPERATOR)),
        " │  eigensolver: " + std::string(HPDDM_STR(EIGENSOLVER)),
        " └"
    };
    size_t max = 0;
    for(const auto& x : v)
        max = std::max(max, x.size());
    output(v, max);
}
} // HPDDM
#endif // _HPDDM_OPTION_IMPL_
