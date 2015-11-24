/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
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

int main(int argc, char **argv) {
#if !((OMPI_MAJOR_VERSION > 1 || (OMPI_MAJOR_VERSION == 1 && OMPI_MINOR_VERSION >= 7)) || MPICH_NUMVERSION >= 30000000)
    MPI_Init(&argc, &argv);
#else
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);
#endif
    /*# Init #*/
    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    HPDDM::Option& opt = *HPDDM::Option::get();
    opt.parse(argc, argv, rankWorld == 0, {
        std::forward_as_tuple("Nx=<100>", "Number of grid points in the x-direction.", HPDDM::Option::Arg::integer),
        std::forward_as_tuple("Ny=<100>", "Number of grid points in the y-direction.", HPDDM::Option::Arg::integer),
        std::forward_as_tuple("overlap=<1>", "Number of grid points in the overlap.", HPDDM::Option::Arg::integer),
        std::forward_as_tuple("symmetric_csr=(0|1)", "Assemble symmetric matrices.", HPDDM::Option::Arg::argument),
        std::forward_as_tuple("nonuniform=(0|1)", "Use a different number of eigenpairs to compute on each subdomain.", HPDDM::Option::Arg::argument)
    });
    if(rankWorld != 0)
        opt.remove("verbosity");
    std::vector<std::vector<int>> mapping;
    mapping.reserve(8);
    std::list<int> o; // at most eight neighbors in 2D
    HPDDM::MatrixCSR<K>* Mat, *MatNeumann = nullptr;
    K* f, *sol;
    HPDDM::underlying_type<K>* d;
    int ndof;
    generate(rankWorld, sizeWorld, o, mapping, ndof, Mat, MatNeumann, d, f, sol);
    int status = 0;
    if(sizeWorld > 1) {
        /*# Creation #*/
        HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K> A;
        /*# CreationEnd #*/
        /*# Initialization #*/
        A.Subdomain::initialize(Mat, o, mapping);
        decltype(mapping)().swap(mapping);
        A.multiplicityScaling(d);
        A.initialize(d);
        /*# InitializationEnd #*/
        if(opt.set("schwarz_coarse_correction")) {
            /*# Factorization #*/
            unsigned short nu = opt["geneo_nu"];
            if(nu > 0) {
                if(opt.app().find("nonuniform") != opt.app().cend())
                    nu += std::max(-opt["geneo_nu"] + 1, std::pow(-1, rankWorld) * rankWorld);
                HPDDM::underlying_type<K> threshold = std::max(0.0, opt.val("geneo_threshold"));
                A.solveGEVP<HPDDM::Arpack>(MatNeumann, nu, threshold);
                opt["geneo_nu"] = nu;
            }
            else {
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
        int it;
        /*# Solution #*/
        if(opt["krylov_method"] == 1)
            it = HPDDM::IterativeMethod::CG(A, sol, f, A.getCommunicator());
        else
            it = HPDDM::IterativeMethod::GMRES(A, sol, f, 1, A.getCommunicator());
        /*# SolutionEnd #*/
        HPDDM::underlying_type<K> storage[2];
        A.computeError(sol, f, storage);
        if(rankWorld == 0)
            std::cout << std::scientific << " --- error = " << storage[1] << " / " << storage[0] << std::endl;
        if(it > 45 || storage[1] / storage[0] > 1.0e-2)
            status = 1;
    }
    else {
        SUBDOMAIN<K> S;
        S.numfact(Mat);
        S.solve(f, sol);
        HPDDM::underlying_type<K> nrmb = HPDDM::Blas<K>::nrm2(&ndof, f, &(HPDDM::i__1));
        K* tmp = new K[ndof];
        HPDDM::Wrapper<K>::csrmv(Mat->_sym, &ndof, Mat->_a, Mat->_ia, Mat->_ja, sol, tmp);
        HPDDM::Blas<K>::axpy(&ndof, &(HPDDM::Wrapper<K>::d__2), f, &(HPDDM::i__1), tmp, &(HPDDM::i__1));
        HPDDM::underlying_type<K> nrmAx = HPDDM::Blas<K>::nrm2(&ndof, tmp, &(HPDDM::i__1));
        std::cout << std::scientific << " --- error = " << nrmAx << " / " << nrmb << std::endl;
        if(nrmAx / nrmb > (std::is_same<double, HPDDM::underlying_type<K>>::value ? 1.0e-8 : 1.0e-2))
            status = 1;
        delete [] tmp;
        delete Mat;
    }
    delete [] d;

    if(opt.set("schwarz_coarse_correction") && opt["geneo_nu"] > 0)
        delete MatNeumann;
    delete [] sol;
    delete [] f;
    MPI_Finalize();
    return status;
}
