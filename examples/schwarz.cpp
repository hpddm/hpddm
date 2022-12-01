/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2015-10-29

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

#include "schwarz.hpp"

struct CustomOperator : public HPDDM::CustomOperator<HPDDM::MatrixCSR<K>, K> {
    explicit CustomOperator(const HPDDM::MatrixCSR<K>* const A) : HPDDM::CustomOperator<HPDDM::MatrixCSR<K>, K>(A) { }
    template<bool>
    int apply(const K* const in, K* const out, const unsigned short& mu = 1, K* = nullptr, const unsigned short& = 0) const {
        const HPDDM::MatrixCSR<K>* const A = getMatrix();
        for(int i = 0; i < _n; ++i) {
            int mid = (A->_sym ? (A->_ia[i + 1] - A->_ia[0]) : std::distance(A->_ja, std::upper_bound(A->_ja + A->_ia[i] - A->_ia[0], A->_ja + A->_ia[i + 1] - A->_ia[0], i + A->_ia[0]))) - 1;
            for(unsigned short nu = 0; nu < mu; ++nu)
                out[nu * _n + i] = in[nu * _n + i] / A->_a[mid];
        }
        return 0;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
#ifdef MU_SLEPC
    SlepcInitialize(&argc, &argv, nullptr, nullptr);
#elif defined(PETSCSUB)
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
    /*# Init #*/
    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    HPDDM::Option& opt = *HPDDM::Option::get();
    opt.parse(argc, argv, rankWorld == 0, {
        std::forward_as_tuple("overlap=<1>", "Number of grid points in the overlap.", HPDDM::Option::Arg::positive),
#ifdef HPDDM_FROMFILE
        std::forward_as_tuple("matrix_filename=<input_file>", "Name of the file in which the matrix is stored.", HPDDM::Option::Arg::argument),
        std::forward_as_tuple("rhs_filename=<input_file>", "Name of the file in which the RHS is stored.", HPDDM::Option::Arg::argument),
#else
        std::forward_as_tuple("Nx=<100>", "Number of grid points in the x-direction.", HPDDM::Option::Arg::positive),
        std::forward_as_tuple("Ny=<100>", "Number of grid points in the y-direction.", HPDDM::Option::Arg::positive),
        std::forward_as_tuple("generate_random_rhs=<0>", "Number of generated random right-hand sides.", HPDDM::Option::Arg::integer),
        std::forward_as_tuple("symmetric_csr=(0|1)", "Assemble symmetric matrices.", HPDDM::Option::Arg::argument),
        std::forward_as_tuple("nonuniform=(0|1)", "Use a different number of eigenpairs to compute on each subdomain.", HPDDM::Option::Arg::argument),
        std::forward_as_tuple("prefix=<string>", "Use a prefix.", HPDDM::Option::Arg::argument)
#endif
    });
    std::string prefix;
    if(opt.prefix("prefix").size())
        prefix = opt.prefix("prefix");
    if(rankWorld != 0) {
        opt.remove("verbosity");
        if(prefix.size() > 0)
            opt.remove(prefix + "verbosity");
    }
    std::vector<std::vector<int>> mapping;
    mapping.reserve(8);
    std::list<int> o; // at most eight neighbors in 2D
    HPDDM::MatrixCSR<K>* Mat, *MatNeumann = nullptr;
    K* f, *sol;
    HPDDM::underlying_type<K>* d = nullptr;
    int ndof;
    generate(rankWorld, sizeWorld, o, mapping, ndof, Mat, MatNeumann, d, f, sol);
#ifdef HPDDM_FROMFILE
    int mu = 1;
#else
    int mu = opt.app()["generate_random_rhs"];
#endif
    int status = 0;
    if(sizeWorld > 1) {
        /*# Creation #*/
        HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K> A;
        /*# CreationEnd #*/
        if(prefix.size() > 0)
            A.setPrefix(prefix);
        /*# Initialization #*/
        A.Subdomain::initialize(Mat, o, mapping);
        decltype(mapping)().swap(mapping);
        A.multiplicityScaling(d);
        A.initialize(d);
        if(mu != 0)
            A.exchange<true>(f, mu);
        else
            mu = 1;
        /*# InitializationEnd #*/
        if(opt.set(prefix + "schwarz_coarse_correction")) {
            /*# Factorization #*/
            double& ref = opt[prefix + "geneo_nu"];
            unsigned short nu = ref;
#ifdef EIGENSOLVER
            if(nu > 0) {
                if(opt.app().find("nonuniform") != opt.app().cend()) {
                    ref += std::max(static_cast<int>(-ref + 1), HPDDM::pow(-1, rankWorld) * rankWorld);
                    if(rankWorld == 4)
                        ref = 0;
                }
                A.solveGEVP<EIGENSOLVER>(MatNeumann);
                nu = opt[prefix + "geneo_nu"];
            }
            else
#endif
            {
                nu = 1;
                K** deflation = new K*[1];
                *deflation = new K[ndof];
                std::fill(*deflation, *deflation + ndof, 1.0);
                A.setVectors(deflation);
            }
            A.super::initialize(nu);
            A.buildTwo(MPI_COMM_WORLD);
            /*# FactorizationEnd #*/
        }
        A.callNumfact();
        /*# Solution #*/
        int it = HPDDM::IterativeMethod::solve(A, f, sol, mu, A.getCommunicator());
        /*# SolutionEnd #*/
        HPDDM::underlying_type<K>* storage = new HPDDM::underlying_type<K>[2 * mu];
        A.computeResidual(sol, f, storage, mu);
        if(rankWorld == 0)
            for(unsigned short nu = 0; nu < mu; ++nu) {
                if(nu == 0)
                    std::cout << " --- residual = ";
                else
                    std::cout << "                ";
                std::cout << std::scientific << storage[1 + 2 * nu] << " / " << storage[2 * nu];
                if(mu > 1)
                    std::cout << " (rhs #" << nu + 1 << ")";
                std::cout << std::endl;
            }
        if(it > ((HPDDM_MIXED_PRECISION || opt.val<char>(prefix + "krylov_method", HPDDM_KRYLOV_METHOD_GMRES) == HPDDM_KRYLOV_METHOD_BFBCG) ? 60 : 45))
            status = 1;
        else {
            for(unsigned short nu = 0; nu < mu; ++nu)
                 if(storage[1 + 2 * nu] / storage[2 * nu] > 1.0e-2)
                     status = 1;
        }
        delete [] storage;
        char verbosity = opt.val<char>("verbosity", 0);
        MPI_Bcast(&verbosity, 1, MPI_CHAR, 0, A.getCommunicator());
        if(verbosity >= 4)
            A.statistics();
    }
    else {
        mu = std::max(1, mu);
        int it = 0;
        std::string filename = opt.prefix(prefix + "dump_matrices", true);
        if(!filename.empty()) {
            std::ofstream output { filename };
            output << *Mat;
        }
        if(opt[prefix + "schwarz_method"] != HPDDM_SCHWARZ_METHOD_NONE) {
            SUBDOMAIN<K> S;
            S.numfact(Mat);
            S.solve(f, sol, mu);
        }
        else
            it = HPDDM::IterativeMethod::solve(CustomOperator(Mat), f, sol, mu, MPI_COMM_SELF);
        HPDDM::underlying_type<K>* nrmb = new HPDDM::underlying_type<K>[2 * mu];
        for(unsigned short nu = 0; nu < mu; ++nu)
            nrmb[nu] = HPDDM::Blas<K>::nrm2(&ndof, f + nu * ndof, &(HPDDM::i__1));
        K* tmp = new K[mu * ndof];
        HPDDM::Wrapper<K>::csrmm(Mat->_sym, &ndof, &mu, Mat->_a, Mat->_ia, Mat->_ja, sol, tmp);
        ndof *= mu;
        float minus = -1.0;
        HPDDM::Blas<float>::axpy(&ndof, &minus, f, &(HPDDM::i__1), tmp, &(HPDDM::i__1));
        ndof /= mu;
        HPDDM::underlying_type<K>* nrmAx = nrmb + mu;
        for(unsigned short nu = 0; nu < mu; ++nu) {
            nrmAx[nu] = HPDDM::Blas<K>::nrm2(&ndof, tmp + nu * ndof, &(HPDDM::i__1));
            if(nu == 0)
                std::cout << " --- residual = ";
            else
                std::cout << "                ";
            std::cout << std::scientific << nrmAx[nu] << " / " << nrmb[nu];
            if(mu > 1)
                std::cout << " (rhs #" << nu + 1 << ")";
            std::cout << std::endl;
            if(nrmAx[nu] / nrmb[nu] > (std::is_same<double, HPDDM::underlying_type<K>>::value ? 1.0e-6 : 1.0e-2))
                status = 1;
        }
        if(it > 75)
            status = 1;
        delete [] tmp;
        delete [] nrmb;
        delete Mat;
    }
    delete [] d;
    delete MatNeumann;
    delete [] sol;
    delete [] f;
#ifdef MU_SLEPC
    SlepcFinalize();
#elif defined(PETSCSUB)
    PetscFinalize();
#endif
    MPI_Finalize();
    return status;
}
