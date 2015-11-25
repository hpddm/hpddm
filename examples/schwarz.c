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

#include "schwarz.h"

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
    const HpddmOption* const opt = HpddmOptionGet();
    HpddmOptionParse(opt, argc, argv, rankWorld == 0);
    {
        char* val[3] = { "Nx=<100>", "Ny=<100>", "overlap=<1>" };
        char* desc[3] = { "Number of grid points in the x-direction.", "Number of grid points in the y-direction.", "Number of grid points in the overlap." };
        HpddmOptionParseInts(opt, argc, argv, 3, val, desc);
        val[0] = "symmetric_csr=(0|1)"; desc[0] = "Assemble symmetric matrices.";
        val[1] = "nonuniform=(0|1)"; desc[1] = "Use a different number of eigenpairs to compute on each subdomain.";
        HpddmOptionParseArgs(opt, argc, argv, 2, val, desc);
    }
    if(rankWorld != 0)
        HpddmOptionRemove(opt, "verbosity");
    int sizes[8];
    int* connectivity[8];
    int o[8];
    int neighbors = 0;
    HpddmMatrixCSR* Mat, *MatNeumann = NULL;
    K* f, *sol;
    underlying_type* d;
    int ndof;
    generate(rankWorld, sizeWorld, &neighbors, o, sizes, connectivity, &ndof, &Mat, &MatNeumann, &d, &f, &sol);
    int status = 0;
    if(sizeWorld > 1) {
        HpddmSchwarz* A = HpddmSchwarzCreate(Mat, neighbors, o, sizes, connectivity);
        HpddmSchwarzMultiplicityScaling(A, d);
        HpddmSchwarzInitialize(A, d);
        if(HpddmOptionSet(opt, "schwarz_coarse_correction")) {
            unsigned short nu = HpddmOptionVal(opt, "geneo_nu");
            if(nu > 0) {
                if(HpddmOptionApp(opt, "nonuniform"))
                    nu += MAX((int)(-HpddmOptionVal(opt, "geneo_nu") + 1), pow(-1, rankWorld) * rankWorld);
                underlying_type threshold = MAX(0.0, HpddmOptionVal(opt, "geneo_threshold"));
                HpddmSchwarzSolveGEVP(A, MatNeumann, &nu, threshold);
                *HpddmOptionAddr(opt, "geneo_nu") = nu;
            }
            else {
                nu = 1;
                K** deflation = malloc(sizeof(K*));
                *deflation = malloc(sizeof(K) * ndof);
                for(int i = 0; i < ndof; ++i)
                    deflation[0][i] = 1.0;
                HpddmSetVectors(HpddmSchwarzPreconditioner(A), deflation);
            }
            HpddmInitializeCoarseOperator(HpddmSchwarzPreconditioner(A), nu);
            HpddmSchwarzBuildCoarseOperator(A, MPI_COMM_WORLD);
            /*# FactorizationEnd #*/
        }
        HpddmSchwarzCallNumfact(A);
        int it;
        const MPI_Comm* comm = HpddmGetCommunicator(HpddmSchwarzPreconditioner(A));
        /*# Solution #*/
        if(HpddmOptionVal(opt, "krylov_method") == 1)
            it = HpddmCG(A, sol, f, comm);
        else
            it = HpddmGMRES(A, sol, f, 1, comm);
        /*# SolutionEnd #*/
        underlying_type storage[2];
        HpddmSchwarzComputeError(A, sol, f, storage);
        if(rankWorld == 0)
            printf(" --- error = %e / %e\n", storage[1], storage[0]);
        if(it > 45 || storage[1] / storage[0] > 1.0e-2)
            status = 1;
        HpddmSchwarzDestroy(A);
    }
    else {
        HpddmSubdomain* S = NULL;
        HpddmSubdomainNumfact(&S, Mat);
        HpddmSubdomainSolve(S, f, sol);
        int one = 1;
        underlying_type nrmb = nrm2(&ndof, f, &one);
        K* tmp = malloc(sizeof(K) * ndof);
        HpddmCsrmv(Mat, sol, tmp);
        K minus = -1;
        axpy(&ndof, &minus, f, &one, tmp, &one);
        underlying_type nrmAx = nrm2(&ndof, tmp, &one);
        printf(" --- error = %e / %e\n", nrmAx, nrmb);
        if(nrmAx / nrmb > (sizeof(underlying_type) == sizeof(double) ? 1.0e-8 : 1.0e-2))
            status = 1;
        free(tmp);
        HpddmSubdomainDestroy(S);
        HpddmMatrixCSRDestroy(Mat);
    }
    free(d);

    if(HpddmOptionSet(opt, "schwarz_coarse_correction") && HpddmOptionVal(opt, "geneo_nu") > 0)
        HpddmMatrixCSRDestroy(MatNeumann);
    free(sol);
    free(f);
    MPI_Finalize();
    return status;
}
