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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    /*# Init #*/
    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    const HpddmOption* const opt = HpddmOptionGet();
    HpddmOptionParse(opt, argc, argv, rankWorld == 0);
    {
        char* val[4] = { "Nx=<100>", "Ny=<100>", "overlap=<1>", "generate_random_rhs=<0>" };
        char* desc[4] = { "Number of grid points in the x-direction.", "Number of grid points in the y-direction.", "Number of grid points in the overlap.", "Number of generated random right-hand sides." };
        HpddmOptionParseInts(opt, argc, argv, 4, val, desc);
        val[0] = "symmetric_csr=(0|1)"; desc[0] = "Assemble symmetric matrices.";
        val[1] = "nonuniform=(0|1)"; desc[1] = "Use a different number of eigenpairs to compute on each subdomain.";
        HpddmOptionParseArgs(opt, argc, argv, 2, val, desc);
    }
    int sizes[8];
    int* connectivity[8];
    int o[8];
    int neighbors = 0;
    HpddmMatrixCSR* Mat, *MatNeumann = NULL;
    K* f, *sol;
    underlying_type* d;
    int ndof;
    generate(rankWorld, sizeWorld, &neighbors, o, sizes, connectivity, &ndof, &Mat, &MatNeumann, &d, &f, &sol);
    unsigned short mu = HpddmOptionApp(opt, "generate_random_rhs");
    int status = 0;
    if(sizeWorld > 1) {
        HpddmSchwarz* A = HpddmSchwarzCreate(Mat, neighbors, o, sizes, connectivity);
        for(int i = 0; i < neighbors; ++i)
            free(connectivity[i]);
        HpddmSchwarzMultiplicityScaling(A, d);
        HpddmSchwarzInitialize(A, d);
        if(mu != 0)
            HpddmSchwarzScaledExchange(A, f, mu);
        else
            mu = 1;
        if(HpddmOptionSet(opt, "schwarz_coarse_correction")) {
            double* addr = HpddmOptionAddr(opt, "geneo_nu");
            unsigned short nu = *addr;
            if(nu > 0) {
                if(HpddmOptionApp(opt, "nonuniform"))
                    *addr += MAX((int)(-*addr + 1), pow(-1, rankWorld) * rankWorld);
                HpddmSchwarzSolveGEVP(A, MatNeumann);
                nu = HpddmOptionVal(opt, "geneo_nu");
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
        if(rankWorld != 0)
            HpddmOptionRemove(opt, "verbosity");
        const MPI_Comm* comm = HpddmGetCommunicator(HpddmSchwarzPreconditioner(A));
        /*# Solution #*/
        int it = HpddmSolve(A, f, sol, mu, comm);
        /*# SolutionEnd #*/
        underlying_type* storage = malloc(sizeof(underlying_type) * 2 * mu);
        HpddmSchwarzComputeResidual(A, sol, f, storage, mu);
        if(rankWorld == 0)
            for(unsigned short nu = 0; nu < mu; ++nu) {
                if(nu == 0)
                    printf(" --- residual = ");
                else
                    printf("                ");
                printf("%e / %e", storage[1 + 2 * nu], storage[2 * nu]);
                if(mu > 1)
                    printf(" (rhs #%d)", nu + 1);
                printf("\n");
            }
        if(it > ((int)HpddmOptionVal(opt, "krylov_method") == 6 ? 60 : 45))
            status = 1;
        else {
            for(unsigned short nu = 0; nu < mu; ++nu)
                 if(storage[1 + 2 * nu] / storage[2 * nu] > 1.0e-2)
                     status = 1;
        }
        free(storage);
        if(HpddmOptionVal(opt, "geneo_nu") == 0)
            HpddmDestroyVectors(HpddmSchwarzPreconditioner(A));
        HpddmSchwarzDestroy(A);
    }
    else {
        HpddmSubdomain* S = NULL;
        HpddmSubdomainNumfact(&S, Mat);
        mu = MAX(1, mu);
        HpddmSubdomainSolve(S, f, sol, mu);
        int one = 1;
        underlying_type* nrmb = malloc(sizeof(underlying_type) * 2 * mu);
        for(unsigned short nu = 0; nu < mu; ++nu)
            nrmb[nu] = nrm2(&ndof, f + nu * ndof, &one);
        K* tmp = malloc(sizeof(K) * mu * ndof);
        HpddmCSRMM(Mat, sol, tmp, mu);
        K minus = -1;
        ndof *= mu;
        axpy(&ndof, &minus, f, &one, tmp, &one);
        ndof /= mu;
        underlying_type* nrmAx = nrmb + mu;
        for(unsigned short nu = 0; nu < mu; ++nu) {
            nrmAx[nu] = nrm2(&ndof, tmp + nu * ndof, &one);
            if(nu == 0)
                printf(" --- residual = ");
            else
                printf("                ");
            printf("%e / %e", nrmAx[nu], nrmb[nu]);
            if(mu > 1)
                printf(" (rhs #%d)", nu + 1);
            printf("\n");
            if(nrmAx[nu] / nrmb[nu] > (sizeof(underlying_type) == sizeof(double) ? 1.0e-6 : 1.0e-2))
                status = 1;
        }
        free(tmp);
        free(nrmb);
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
