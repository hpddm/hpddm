/*
  This file is part of HPDDM.

  Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
       Date: 2016-08-04

  Copyright (C) 2016-     Centre National de la Recherche Scientifique

  Note:      Reference PETSc implementation available at
                                                      https://bit.ly/2GB8ATC
             Contributed by
                                                                  Mark Adams

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

#define HPDDM_PETSC 1
#include <HPDDM.h>
#include <petsc.h>
#define SIZE_ARRAY_R 4

PetscErrorCode elem_3d_elast_v_25(PetscScalar*);
PetscErrorCode AssembleSystem(Mat, Vec, PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscInt, PetscMPIInt, PetscMPIInt,
                              PetscInt, PetscInt);
PetscErrorCode ComputeError(Mat, Vec, Vec);

int main(int argc, char** argv)
{
    PetscErrorCode ierr;
    PetscInt m, nn, M, j, k, ne = 4;
    PetscReal* coords;
    Vec x, rhs;
    Mat A;
    KSP ksp;
    PetscMPIInt npe, rank;
    PetscInitialize(&argc, &argv, NULL, NULL);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &npe);
    CHKERRQ(ierr);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Linear elasticity in 3D", "");
    CHKERRQ(ierr);
    {
        char nestring[256];
        ierr = PetscSNPrintf(nestring, sizeof nestring, "number of elements in each direction, ne+1 must be a multiple of %D (sizes^{1/3})",
                             (PetscInt)(PetscPowReal((PetscReal)npe, 1.0 / 3.0) + 0.5));
        CHKERRQ(ierr);
        ierr = PetscOptionsInt("-ne", nestring, "", ne, &ne, NULL);
        CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);
    ierr = HpddmRegisterKSP();
    CHKERRQ(ierr);
    nn = ne + 1;
    M = 3 * nn * nn * nn;
    if (npe == 2) {
        if (rank == 1)
            m = 0;
        else
            m = nn * nn * nn;
        npe = 1;
    }
    else {
        m = nn * nn * nn / npe;
        if (rank == npe - 1) m = nn * nn * nn - (npe - 1) * m;
    }
    m *= 3;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
    int i;
    {
        PetscInt Istart, Iend, jj, ic;
        const PetscInt NP = (PetscInt)(PetscPowReal((PetscReal)npe, 1.0 / 3.0) + 0.5);
        const PetscInt ipx = rank % NP, ipy = (rank % (NP * NP)) / NP, ipz = rank / (NP * NP);
        const PetscInt Ni0 = ipx * (nn / NP), Nj0 = ipy * (nn / NP), Nk0 = ipz * (nn / NP);
        const PetscInt Ni1 = Ni0 + (m > 0 ? (nn / NP) : 0), Nj1 = Nj0 + (nn / NP), Nk1 = Nk0 + (nn / NP);
        PetscInt *d_nnz, *o_nnz, osz[4] = {0, 9, 15, 19}, nbc;
        if (npe != NP * NP * NP) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "npe=%d: npe^{1/3} must be integer", npe);
        if (nn != NP * (nn / NP)) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "-ne %d: (ne+1)%(npe^{1/3}) must equal zero", ne);
        ierr = PetscMalloc1(m + 1, &d_nnz);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(m + 1, &o_nnz);
        CHKERRQ(ierr);
        for (i = Ni0, ic = 0; i < Ni1; i++) {
            for (j = Nj0; j < Nj1; j++) {
                for (k = Nk0; k < Nk1; k++) {
                    nbc = 0;
                    if (i == Ni0 || i == Ni1 - 1) nbc++;
                    if (j == Nj0 || j == Nj1 - 1) nbc++;
                    if (k == Nk0 || k == Nk1 - 1) nbc++;
                    for (jj = 0; jj < 3; jj++, ic++) {
                        d_nnz[ic] = 3 * (27 - osz[nbc]);
                        o_nnz[ic] = 3 * osz[nbc];
                    }
                }
            }
        }
        if (ic != m) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "ic %D does not equal m %D", ic, m);
        ierr = MatCreate(PETSC_COMM_WORLD, &A);
        CHKERRQ(ierr);
        ierr = MatSetSizes(A, m, m, M, M);
        CHKERRQ(ierr);
        ierr = MatSetBlockSize(A, 3);
        CHKERRQ(ierr);
        ierr = MatSetType(A, MATAIJ);
        CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(A, 0, d_nnz);
        CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(A, 0, d_nnz, 0, o_nnz);
        CHKERRQ(ierr);
        ierr = PetscFree(d_nnz);
        CHKERRQ(ierr);
        ierr = PetscFree(o_nnz);
        CHKERRQ(ierr);
        ierr = MatGetOwnershipRange(A, &Istart, &Iend);
        CHKERRQ(ierr);
        if (m != Iend - Istart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "m %D does not equal Iend %D - Istart %D", m, Iend, Istart);
        ierr = VecCreate(PETSC_COMM_WORLD, &x);
        CHKERRQ(ierr);
        ierr = VecSetSizes(x, m, M);
        CHKERRQ(ierr);
        ierr = VecSetBlockSize(x, 3);
        CHKERRQ(ierr);
        ierr = VecSetFromOptions(x);
        CHKERRQ(ierr);
        ierr = VecDuplicate(x, &rhs);
        CHKERRQ(ierr);
        ierr = PetscMalloc1(m + 1, &coords);
        CHKERRQ(ierr);
        coords[m] = -99.0;
        PetscReal h = 1.0 / ne;
        for (i = Ni0, ic = 0; i < Ni1; i++) {
            for (j = Nj0; j < Nj1; j++) {
                for (k = Nk0; k < Nk1; k++, ic++) {
                    coords[3 * ic] = h * (PetscReal)i;
                    coords[3 * ic + 1] = h * (PetscReal)j;
                    coords[3 * ic + 2] = h * (PetscReal)k;
                }
            }
        }
    }
    PetscReal s_r[SIZE_ARRAY_R] = {30, 0.1, 20, 10};
    PetscReal x_r[SIZE_ARRAY_R] = {0.5, 0.4, 0.4, 0.4};
    PetscReal y_r[SIZE_ARRAY_R] = {0.5, 0.5, 0.4, 0.4};
    PetscReal z_r[SIZE_ARRAY_R] = {0.5, 0.45, 0.4, 0.35};
    PetscReal r[SIZE_ARRAY_R] = {0.5, 0.5, 0.4, 0.4};
    AssembleSystem(A, rhs, s_r[0], x_r[0], y_r[0], z_r[0], r[0], ne, npe, rank, nn, m);
    ierr = KSPSetOperators(ksp, A, A);
    CHKERRQ(ierr);
    MatNullSpace matnull;
    Vec vec_coords;
    PetscScalar* c;
    ierr = VecCreate(MPI_COMM_WORLD, &vec_coords);
    CHKERRQ(ierr);
    ierr = VecSetBlockSize(vec_coords, 3);
    CHKERRQ(ierr);
    ierr = VecSetSizes(vec_coords, m, PETSC_DECIDE);
    CHKERRQ(ierr);
    ierr = VecSetUp(vec_coords);
    CHKERRQ(ierr);
    ierr = VecGetArray(vec_coords, &c);
    CHKERRQ(ierr);
    for (i = 0; i < m; i++) c[i] = coords[i];
    ierr = VecRestoreArray(vec_coords, &c);
    CHKERRQ(ierr);
    ierr = MatNullSpaceCreateRigidBody(vec_coords, &matnull);
    CHKERRQ(ierr);
    ierr = MatSetNearNullSpace(A, matnull);
    CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&matnull);
    CHKERRQ(ierr);
    ierr = VecDestroy(&vec_coords);
    CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    CHKERRQ(ierr);
    MPI_Barrier(PETSC_COMM_WORLD);
    double time = MPI_Wtime();
    ierr = KSPSetUp(ksp);
    CHKERRQ(ierr);
    MPI_Barrier(PETSC_COMM_WORLD);
    time = MPI_Wtime() - time;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "--- PC setup = %f\n", time);
    CHKERRQ(ierr);
    float t_time[SIZE_ARRAY_R];
    int t_its[SIZE_ARRAY_R];
    for (j = 0; j < 2; ++j) {
        {
            if (j == 1) {
                ierr = KSPSetType(ksp, "hpddm");
                CHKERRQ(ierr);
                ierr = KSPSetFromOptions(ksp);
                CHKERRQ(ierr);
                ierr = VecZeroEntries(x);
                CHKERRQ(ierr);
            }
            ierr = KSPSolve(ksp, rhs, x);
            CHKERRQ(ierr);
            if (j == 1) {
                const HpddmOption* const opt = HpddmOptionGet();
                int previous = HpddmOptionVal(opt, "krylov_method");
                if (previous == HPDDM_KRYLOV_METHOD_GCRODR || previous == HPDDM_KRYLOV_METHOD_BGCRODR) HpddmDestroyRecycling();
            }
            ierr = KSPReset(ksp);
            CHKERRQ(ierr);
            ierr = KSPSetOperators(ksp, A, A);
            CHKERRQ(ierr);
            ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
            CHKERRQ(ierr);
            ierr = KSPSetUp(ksp);
            CHKERRQ(ierr);
        }
        for (i = 0; i < SIZE_ARRAY_R; ++i) {
            ierr = VecZeroEntries(x);
            CHKERRQ(ierr);
            MPI_Barrier(PETSC_COMM_WORLD);
            time = MPI_Wtime();
            ierr = KSPSolve(ksp, rhs, x);
            CHKERRQ(ierr);
            MPI_Barrier(PETSC_COMM_WORLD);
            t_time[i] = MPI_Wtime() - time;
            PetscInt its;
            ierr = KSPGetIterationNumber(ksp, &its);
            CHKERRQ(ierr);
            t_its[i] = its;
            ierr = ComputeError(A, rhs, x);
            CHKERRQ(ierr);
            if (i == (SIZE_ARRAY_R - 1))
                AssembleSystem(A, rhs, s_r[0], x_r[0], y_r[0], z_r[0], r[0], ne, npe, rank, nn, m);
            else
                AssembleSystem(A, rhs, s_r[i + 1], x_r[i + 1], y_r[i + 1], z_r[i + 1], r[i + 1], ne, npe, rank, nn, m);
            ierr = KSPSetOperators(ksp, A, A);
            CHKERRQ(ierr);
            ierr = KSPSetUp(ksp);
            CHKERRQ(ierr);
        }
        for (i = 0; i < SIZE_ARRAY_R; ++i) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t%d\t%f\n", i + 1, t_its[i], t_time[i]);
            CHKERRQ(ierr);
            if (i > 0) {
                t_its[0] += t_its[i];
                t_time[0] += t_time[i];
            }
        }
        if (SIZE_ARRAY_R > 1) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, "------------------------\n\t%d\t%f\n", t_its[0], t_time[0]);
            CHKERRQ(ierr);
        }
    }
    ierr = KSPDestroy(&ksp);
    CHKERRQ(ierr);
    ierr = VecDestroy(&x);
    CHKERRQ(ierr);
    ierr = VecDestroy(&rhs);
    CHKERRQ(ierr);
    ierr = MatDestroy(&A);
    CHKERRQ(ierr);
    ierr = PetscFree(coords);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}

PetscErrorCode elem_3d_elast_v_25(PetscScalar* dd)
{
    PetscErrorCode ierr;
    PetscScalar DD[] = {
        0.18981481481481474,       5.27777777777777568E-002,  5.27777777777777568E-002,  -5.64814814814814659E-002,
        -1.38888888888889072E-002, -1.38888888888889089E-002, -8.24074074074073876E-002, -5.27777777777777429E-002,
        1.38888888888888725E-002,  4.90740740740740339E-002,  1.38888888888889124E-002,  4.72222222222222071E-002,
        4.90740740740740339E-002,  4.72222222222221932E-002,  1.38888888888888968E-002,  -8.24074074074073876E-002,
        1.38888888888888673E-002,  -5.27777777777777429E-002, -7.87037037037036785E-002, -4.72222222222221932E-002,
        -4.72222222222222071E-002, 1.20370370370370180E-002,  -1.38888888888888742E-002, -1.38888888888888829E-002,
        5.27777777777777568E-002,  0.18981481481481474,       5.27777777777777568E-002,  1.38888888888889124E-002,
        4.90740740740740269E-002,  4.72222222222221932E-002,  -5.27777777777777637E-002, -8.24074074074073876E-002,
        1.38888888888888725E-002,  -1.38888888888889037E-002, -5.64814814814814728E-002, -1.38888888888888985E-002,
        4.72222222222221932E-002,  4.90740740740740478E-002,  1.38888888888888968E-002,  -1.38888888888888673E-002,
        1.20370370370370058E-002,  -1.38888888888888742E-002, -4.72222222222221932E-002, -7.87037037037036785E-002,
        -4.72222222222222002E-002, 1.38888888888888742E-002,  -8.24074074074073598E-002, -5.27777777777777568E-002,
        5.27777777777777568E-002,  5.27777777777777568E-002,  0.18981481481481474,       1.38888888888889055E-002,
        4.72222222222222002E-002,  4.90740740740740269E-002,  -1.38888888888888829E-002, -1.38888888888888829E-002,
        1.20370370370370180E-002,  4.72222222222222002E-002,  1.38888888888888985E-002,  4.90740740740740339E-002,
        -1.38888888888888985E-002, -1.38888888888888968E-002, -5.64814814814814520E-002, -5.27777777777777568E-002,
        1.38888888888888777E-002,  -8.24074074074073876E-002, -4.72222222222222002E-002, -4.72222222222221932E-002,
        -7.87037037037036646E-002, 1.38888888888888794E-002,  -5.27777777777777568E-002, -8.24074074074073598E-002,
        -5.64814814814814659E-002, 1.38888888888889124E-002,  1.38888888888889055E-002,  0.18981481481481474,
        -5.27777777777777568E-002, -5.27777777777777499E-002, 4.90740740740740269E-002,  -1.38888888888889072E-002,
        -4.72222222222221932E-002, -8.24074074074073876E-002, 5.27777777777777568E-002,  -1.38888888888888812E-002,
        -8.24074074074073876E-002, -1.38888888888888742E-002, 5.27777777777777499E-002,  4.90740740740740269E-002,
        -4.72222222222221863E-002, -1.38888888888889089E-002, 1.20370370370370162E-002,  1.38888888888888673E-002,
        1.38888888888888742E-002,  -7.87037037037036785E-002, 4.72222222222222002E-002,  4.72222222222222071E-002,
        -1.38888888888889072E-002, 4.90740740740740269E-002,  4.72222222222222002E-002,  -5.27777777777777568E-002,
        0.18981481481481480,       5.27777777777777568E-002,  1.38888888888889020E-002,  -5.64814814814814728E-002,
        -1.38888888888888951E-002, 5.27777777777777637E-002,  -8.24074074074073876E-002, 1.38888888888888881E-002,
        1.38888888888888742E-002,  1.20370370370370232E-002,  -1.38888888888888812E-002, -4.72222222222221863E-002,
        4.90740740740740339E-002,  1.38888888888888933E-002,  -1.38888888888888812E-002, -8.24074074074073876E-002,
        -5.27777777777777568E-002, 4.72222222222222071E-002,  -7.87037037037036924E-002, -4.72222222222222140E-002,
        -1.38888888888889089E-002, 4.72222222222221932E-002,  4.90740740740740269E-002,  -5.27777777777777499E-002,
        5.27777777777777568E-002,  0.18981481481481477,       -4.72222222222222071E-002, 1.38888888888888968E-002,
        4.90740740740740131E-002,  1.38888888888888812E-002,  -1.38888888888888708E-002, 1.20370370370370267E-002,
        5.27777777777777568E-002,  1.38888888888888812E-002,  -8.24074074074073876E-002, 1.38888888888889124E-002,
        -1.38888888888889055E-002, -5.64814814814814589E-002, -1.38888888888888812E-002, -5.27777777777777568E-002,
        -8.24074074074073737E-002, 4.72222222222222002E-002,  -4.72222222222222002E-002, -7.87037037037036924E-002,
        -8.24074074074073876E-002, -5.27777777777777637E-002, -1.38888888888888829E-002, 4.90740740740740269E-002,
        1.38888888888889020E-002,  -4.72222222222222071E-002, 0.18981481481481480,       5.27777777777777637E-002,
        -5.27777777777777637E-002, -5.64814814814814728E-002, -1.38888888888889037E-002, 1.38888888888888951E-002,
        -7.87037037037036785E-002, -4.72222222222222002E-002, 4.72222222222221932E-002,  1.20370370370370128E-002,
        -1.38888888888888725E-002, 1.38888888888888812E-002,  4.90740740740740408E-002,  4.72222222222222002E-002,
        -1.38888888888888951E-002, -8.24074074074073876E-002, 1.38888888888888812E-002,  5.27777777777777637E-002,
        -5.27777777777777429E-002, -8.24074074074073876E-002, -1.38888888888888829E-002, -1.38888888888889072E-002,
        -5.64814814814814728E-002, 1.38888888888888968E-002,  5.27777777777777637E-002,  0.18981481481481480,
        -5.27777777777777568E-002, 1.38888888888888916E-002,  4.90740740740740339E-002,  -4.72222222222222210E-002,
        -4.72222222222221932E-002, -7.87037037037036924E-002, 4.72222222222222002E-002,  1.38888888888888742E-002,
        -8.24074074074073876E-002, 5.27777777777777429E-002,  4.72222222222222002E-002,  4.90740740740740269E-002,
        -1.38888888888888951E-002, -1.38888888888888846E-002, 1.20370370370370267E-002,  1.38888888888888916E-002,
        1.38888888888888725E-002,  1.38888888888888725E-002,  1.20370370370370180E-002,  -4.72222222222221932E-002,
        -1.38888888888888951E-002, 4.90740740740740131E-002,  -5.27777777777777637E-002, -5.27777777777777568E-002,
        0.18981481481481480,       -1.38888888888888968E-002, -4.72222222222221932E-002, 4.90740740740740339E-002,
        4.72222222222221932E-002,  4.72222222222222071E-002,  -7.87037037037036646E-002, -1.38888888888888742E-002,
        5.27777777777777499E-002,  -8.24074074074073737E-002, 1.38888888888888933E-002,  1.38888888888889020E-002,
        -5.64814814814814589E-002, 5.27777777777777568E-002,  -1.38888888888888794E-002, -8.24074074074073876E-002,
        4.90740740740740339E-002,  -1.38888888888889037E-002, 4.72222222222222002E-002,  -8.24074074074073876E-002,
        5.27777777777777637E-002,  1.38888888888888812E-002,  -5.64814814814814728E-002, 1.38888888888888916E-002,
        -1.38888888888888968E-002, 0.18981481481481480,       -5.27777777777777499E-002, 5.27777777777777707E-002,
        1.20370370370370180E-002,  1.38888888888888812E-002,  -1.38888888888888812E-002, -7.87037037037036785E-002,
        4.72222222222222002E-002,  -4.72222222222222071E-002, -8.24074074074073876E-002, -1.38888888888888742E-002,
        -5.27777777777777568E-002, 4.90740740740740616E-002,  -4.72222222222222002E-002, 1.38888888888888846E-002,
        1.38888888888889124E-002,  -5.64814814814814728E-002, 1.38888888888888985E-002,  5.27777777777777568E-002,
        -8.24074074074073876E-002, -1.38888888888888708E-002, -1.38888888888889037E-002, 4.90740740740740339E-002,
        -4.72222222222221932E-002, -5.27777777777777499E-002, 0.18981481481481480,       -5.27777777777777568E-002,
        -1.38888888888888673E-002, -8.24074074074073598E-002, 5.27777777777777429E-002,  4.72222222222222002E-002,
        -7.87037037037036785E-002, 4.72222222222222002E-002,  1.38888888888888708E-002,  1.20370370370370128E-002,
        1.38888888888888760E-002,  -4.72222222222222002E-002, 4.90740740740740478E-002,  -1.38888888888888951E-002,
        4.72222222222222071E-002,  -1.38888888888888985E-002, 4.90740740740740339E-002,  -1.38888888888888812E-002,
        1.38888888888888881E-002,  1.20370370370370267E-002,  1.38888888888888951E-002,  -4.72222222222222210E-002,
        4.90740740740740339E-002,  5.27777777777777707E-002,  -5.27777777777777568E-002, 0.18981481481481477,
        1.38888888888888829E-002,  5.27777777777777707E-002,  -8.24074074074073598E-002, -4.72222222222222140E-002,
        4.72222222222222140E-002,  -7.87037037037036646E-002, -5.27777777777777707E-002, -1.38888888888888829E-002,
        -8.24074074074073876E-002, -1.38888888888888881E-002, 1.38888888888888881E-002,  -5.64814814814814589E-002,
        4.90740740740740339E-002,  4.72222222222221932E-002,  -1.38888888888888985E-002, -8.24074074074073876E-002,
        1.38888888888888742E-002,  5.27777777777777568E-002,  -7.87037037037036785E-002, -4.72222222222221932E-002,
        4.72222222222221932E-002,  1.20370370370370180E-002,  -1.38888888888888673E-002, 1.38888888888888829E-002,
        0.18981481481481469,       5.27777777777777429E-002,  -5.27777777777777429E-002, -5.64814814814814659E-002,
        -1.38888888888889055E-002, 1.38888888888889055E-002,  -8.24074074074074153E-002, -5.27777777777777429E-002,
        -1.38888888888888760E-002, 4.90740740740740408E-002,  1.38888888888888968E-002,  -4.72222222222222071E-002,
        4.72222222222221932E-002,  4.90740740740740478E-002,  -1.38888888888888968E-002, -1.38888888888888742E-002,
        1.20370370370370232E-002,  1.38888888888888812E-002,  -4.72222222222222002E-002, -7.87037037037036924E-002,
        4.72222222222222071E-002,  1.38888888888888812E-002,  -8.24074074074073598E-002, 5.27777777777777707E-002,
        5.27777777777777429E-002,  0.18981481481481477,       -5.27777777777777499E-002, 1.38888888888889107E-002,
        4.90740740740740478E-002,  -4.72222222222221932E-002, -5.27777777777777568E-002, -8.24074074074074153E-002,
        -1.38888888888888812E-002, -1.38888888888888846E-002, -5.64814814814814659E-002, 1.38888888888888812E-002,
        1.38888888888888968E-002,  1.38888888888888968E-002,  -5.64814814814814520E-002, 5.27777777777777499E-002,
        -1.38888888888888812E-002, -8.24074074074073876E-002, 4.72222222222221932E-002,  4.72222222222222002E-002,
        -7.87037037037036646E-002, -1.38888888888888812E-002, 5.27777777777777429E-002,  -8.24074074074073598E-002,
        -5.27777777777777429E-002, -5.27777777777777499E-002, 0.18981481481481474,       -1.38888888888888985E-002,
        -4.72222222222221863E-002, 4.90740740740740339E-002,  1.38888888888888829E-002,  1.38888888888888777E-002,
        1.20370370370370249E-002,  -4.72222222222222002E-002, -1.38888888888888933E-002, 4.90740740740740339E-002,
        -8.24074074074073876E-002, -1.38888888888888673E-002, -5.27777777777777568E-002, 4.90740740740740269E-002,
        -4.72222222222221863E-002, 1.38888888888889124E-002,  1.20370370370370128E-002,  1.38888888888888742E-002,
        -1.38888888888888742E-002, -7.87037037037036785E-002, 4.72222222222222002E-002,  -4.72222222222222140E-002,
        -5.64814814814814659E-002, 1.38888888888889107E-002,  -1.38888888888888985E-002, 0.18981481481481474,
        -5.27777777777777499E-002, 5.27777777777777499E-002,  4.90740740740740339E-002,  -1.38888888888889055E-002,
        4.72222222222221932E-002,  -8.24074074074074153E-002, 5.27777777777777499E-002,  1.38888888888888829E-002,
        1.38888888888888673E-002,  1.20370370370370058E-002,  1.38888888888888777E-002,  -4.72222222222221863E-002,
        4.90740740740740339E-002,  -1.38888888888889055E-002, -1.38888888888888725E-002, -8.24074074074073876E-002,
        5.27777777777777499E-002,  4.72222222222222002E-002,  -7.87037037037036785E-002, 4.72222222222222140E-002,
        -1.38888888888889055E-002, 4.90740740740740478E-002,  -4.72222222222221863E-002, -5.27777777777777499E-002,
        0.18981481481481469,       -5.27777777777777499E-002, 1.38888888888889072E-002,  -5.64814814814814659E-002,
        1.38888888888889003E-002,  5.27777777777777429E-002,  -8.24074074074074153E-002, -1.38888888888888812E-002,
        -5.27777777777777429E-002, -1.38888888888888742E-002, -8.24074074074073876E-002, -1.38888888888889089E-002,
        1.38888888888888933E-002,  -5.64814814814814589E-002, 1.38888888888888812E-002,  5.27777777777777429E-002,
        -8.24074074074073737E-002, -4.72222222222222071E-002, 4.72222222222222002E-002,  -7.87037037037036646E-002,
        1.38888888888889055E-002,  -4.72222222222221932E-002, 4.90740740740740339E-002,  5.27777777777777499E-002,
        -5.27777777777777499E-002, 0.18981481481481474,       4.72222222222222002E-002,  -1.38888888888888985E-002,
        4.90740740740740339E-002,  -1.38888888888888846E-002, 1.38888888888888812E-002,  1.20370370370370284E-002,
        -7.87037037037036785E-002, -4.72222222222221932E-002, -4.72222222222222002E-002, 1.20370370370370162E-002,
        -1.38888888888888812E-002, -1.38888888888888812E-002, 4.90740740740740408E-002,  4.72222222222222002E-002,
        1.38888888888888933E-002,  -8.24074074074073876E-002, 1.38888888888888708E-002,  -5.27777777777777707E-002,
        -8.24074074074074153E-002, -5.27777777777777568E-002, 1.38888888888888829E-002,  4.90740740740740339E-002,
        1.38888888888889072E-002,  4.72222222222222002E-002,  0.18981481481481477,       5.27777777777777429E-002,
        5.27777777777777568E-002,  -5.64814814814814659E-002, -1.38888888888888846E-002, -1.38888888888888881E-002,
        -4.72222222222221932E-002, -7.87037037037036785E-002, -4.72222222222221932E-002, 1.38888888888888673E-002,
        -8.24074074074073876E-002, -5.27777777777777568E-002, 4.72222222222222002E-002,  4.90740740740740269E-002,
        1.38888888888889020E-002,  -1.38888888888888742E-002, 1.20370370370370128E-002,  -1.38888888888888829E-002,
        -5.27777777777777429E-002, -8.24074074074074153E-002, 1.38888888888888777E-002,  -1.38888888888889055E-002,
        -5.64814814814814659E-002, -1.38888888888888985E-002, 5.27777777777777429E-002,  0.18981481481481469,
        5.27777777777777429E-002,  1.38888888888888933E-002,  4.90740740740740339E-002,  4.72222222222222071E-002,
        -4.72222222222222071E-002, -4.72222222222222002E-002, -7.87037037037036646E-002, 1.38888888888888742E-002,
        -5.27777777777777568E-002, -8.24074074074073737E-002, -1.38888888888888951E-002, -1.38888888888888951E-002,
        -5.64814814814814589E-002, -5.27777777777777568E-002, 1.38888888888888760E-002,  -8.24074074074073876E-002,
        -1.38888888888888760E-002, -1.38888888888888812E-002, 1.20370370370370249E-002,  4.72222222222221932E-002,
        1.38888888888889003E-002,  4.90740740740740339E-002,  5.27777777777777568E-002,  5.27777777777777429E-002,
        0.18981481481481474,       1.38888888888888933E-002,  4.72222222222222071E-002,  4.90740740740740339E-002,
        1.20370370370370180E-002,  1.38888888888888742E-002,  1.38888888888888794E-002,  -7.87037037037036785E-002,
        4.72222222222222071E-002,  4.72222222222222002E-002,  -8.24074074074073876E-002, -1.38888888888888846E-002,
        5.27777777777777568E-002,  4.90740740740740616E-002,  -4.72222222222222002E-002, -1.38888888888888881E-002,
        4.90740740740740408E-002,  -1.38888888888888846E-002, -4.72222222222222002E-002, -8.24074074074074153E-002,
        5.27777777777777429E-002,  -1.38888888888888846E-002, -5.64814814814814659E-002, 1.38888888888888933E-002,
        1.38888888888888933E-002,  0.18981481481481477,       -5.27777777777777568E-002, -5.27777777777777637E-002,
        -1.38888888888888742E-002, -8.24074074074073598E-002, -5.27777777777777568E-002, 4.72222222222222002E-002,
        -7.87037037037036924E-002, -4.72222222222222002E-002, 1.38888888888888812E-002,  1.20370370370370267E-002,
        -1.38888888888888794E-002, -4.72222222222222002E-002, 4.90740740740740478E-002,  1.38888888888888881E-002,
        1.38888888888888968E-002,  -5.64814814814814659E-002, -1.38888888888888933E-002, 5.27777777777777499E-002,
        -8.24074074074074153E-002, 1.38888888888888812E-002,  -1.38888888888888846E-002, 4.90740740740740339E-002,
        4.72222222222222071E-002,  -5.27777777777777568E-002, 0.18981481481481477,       5.27777777777777637E-002,
        -1.38888888888888829E-002, -5.27777777777777568E-002, -8.24074074074073598E-002, 4.72222222222222071E-002,
        -4.72222222222222140E-002, -7.87037037037036924E-002, 5.27777777777777637E-002,  1.38888888888888916E-002,
        -8.24074074074073876E-002, 1.38888888888888846E-002,  -1.38888888888888951E-002, -5.64814814814814589E-002,
        -4.72222222222222071E-002, 1.38888888888888812E-002,  4.90740740740740339E-002,  1.38888888888888829E-002,
        -1.38888888888888812E-002, 1.20370370370370284E-002,  -1.38888888888888881E-002, 4.72222222222222071E-002,
        4.90740740740740339E-002,  -5.27777777777777637E-002, 5.27777777777777637E-002,  0.18981481481481477,
    };
    PetscFunctionBeginUser;
    ierr = PetscMemcpy(dd, DD, sizeof(PetscScalar) * 576);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode AssembleSystem(Mat A, Vec b, PetscScalar soft_alpha, PetscScalar x_r, PetscScalar y_r, PetscScalar z_r, PetscScalar r,
                              PetscInt ne, PetscMPIInt npe, PetscMPIInt rank, PetscInt nn, PetscInt m)
{
    PetscErrorCode ierr;
    PetscReal h = 1.0 / ne;
    PetscScalar DD[24][24], DD2[24][24];
    PetscScalar DD1[24][24];
    const PetscInt NP = (PetscInt)(PetscPowReal((PetscReal)npe, 1.0 / 3.0) + 0.5);
    const PetscInt ipx = rank % NP, ipy = (rank % (NP * NP)) / NP, ipz = rank / (NP * NP);
    const PetscInt Ni0 = ipx * (nn / NP), Nj0 = ipy * (nn / NP), Nk0 = ipz * (nn / NP);
    const PetscInt Ni1 = Ni0 + (m > 0 ? (nn / NP) : 0), Nj1 = Nj0 + (nn / NP), Nk1 = Nk0 + (nn / NP);
    const PetscInt NN = nn / NP, id0 = ipz * nn * nn * NN + ipy * nn * NN * NN + ipx * NN * NN * NN;
    PetscScalar vv[24], v2[24];
    PetscInt i, j, k;
    {
        ierr = elem_3d_elast_v_25((PetscScalar*)DD1);
        CHKERRQ(ierr);
        for (i = 0; i < 24; i++) {
            for (j = 0; j < 24; j++) {
                if (i < 12 || j < 12) {
                    if (i == j)
                        DD2[i][j] = 0.1 * DD1[i][j];
                    else
                        DD2[i][j] = 0.0;
                }
                else
                    DD2[i][j] = DD1[i][j];
            }
        }
        for (i = 0; i < 24; i++) {
            if (i % 3 == 0)
                vv[i] = h * h;
            else if (i % 3 == 1)
                vv[i] = 2.0 * h * h;
            else
                vv[i] = 0.0;
        }
        for (i = 0; i < 24; i++) {
            if (i % 3 == 0 && i >= 12)
                v2[i] = h * h;
            else if (i % 3 == 1 && i >= 12)
                v2[i] = 2.0 * h * h;
            else
                v2[i] = 0.0;
        }
    }
    ierr = MatZeroEntries(A);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(b);
    CHKERRQ(ierr);
    PetscInt ii, jj, kk;
    for (i = Ni0, ii = 0; i < Ni1; i++, ii++) {
        for (j = Nj0, jj = 0; j < Nj1; j++, jj++) {
            for (k = Nk0, kk = 0; k < Nk1; k++, kk++) {
                PetscReal x = h * (PetscReal)i;
                PetscReal y = h * (PetscReal)j;
                PetscReal z = h * (PetscReal)k;
                PetscInt id = id0 + ii + NN * jj + NN * NN * kk;
                if (i < ne && j < ne && k < ne) {
                    PetscReal radius = PetscSqrtReal((x - 0.5 + h / 2) * (x - 0.5 + h / 2) + (y - 0.5 + h / 2) * (y - 0.5 + h / 2) +
                                                     (z - 0.5 + h / 2) * (z - 0.5 + h / 2));
                    PetscReal alpha = 1.0;
                    PetscInt jx, ix, idx[8];
                    idx[0] = id;
                    idx[1] = id + 1;
                    idx[2] = id + NN + 1;
                    idx[3] = id + NN;
                    idx[4] = id + NN * NN;
                    idx[5] = id + 1 + NN * NN;
                    idx[6] = id + NN + 1 + NN * NN;
                    idx[7] = id + NN + NN * NN;
                    if (i == Ni1 - 1 && Ni1 != nn) {
                        idx[1] += NN * (NN * NN - 1);
                        idx[2] += NN * (NN * NN - 1);
                        idx[5] += NN * (NN * NN - 1);
                        idx[6] += NN * (NN * NN - 1);
                    }
                    if (j == Nj1 - 1 && Nj1 != nn) {
                        idx[2] += NN * NN * (nn - 1);
                        idx[3] += NN * NN * (nn - 1);
                        idx[6] += NN * NN * (nn - 1);
                        idx[7] += NN * NN * (nn - 1);
                    }
                    if (k == Nk1 - 1 && Nk1 != nn) {
                        idx[4] += NN * (nn * nn - NN * NN);
                        idx[5] += NN * (nn * nn - NN * NN);
                        idx[6] += NN * (nn * nn - NN * NN);
                        idx[7] += NN * (nn * nn - NN * NN);
                    }
                    if (radius < r) alpha = soft_alpha;

                    for (ix = 0; ix < 24; ix++) {
                        for (jx = 0; jx < 24; jx++) DD[ix][jx] = alpha * DD1[ix][jx];
                    }
                    if (k > 0) {
                        ierr = MatSetValuesBlocked(A, 8, idx, 8, idx, (const PetscScalar*)DD, ADD_VALUES);
                        CHKERRQ(ierr);
                        ierr = VecSetValuesBlocked(b, 8, idx, (const PetscScalar*)vv, ADD_VALUES);
                        CHKERRQ(ierr);
                    }
                    else {
                        for (ix = 0; ix < 24; ix++) {
                            for (jx = 0; jx < 24; jx++) DD[ix][jx] = alpha * DD2[ix][jx];
                        }
                        ierr = MatSetValuesBlocked(A, 8, idx, 8, idx, (const PetscScalar*)DD, ADD_VALUES);
                        CHKERRQ(ierr);
                        ierr = VecSetValuesBlocked(b, 8, idx, (const PetscScalar*)v2, ADD_VALUES);
                        CHKERRQ(ierr);
                    }
                }
            }
        }
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ComputeError(Mat A, Vec rhs, Vec x)
{
    Vec err;
    PetscReal norm_b, norm_err;
    PetscFunctionBeginUser;
    PetscErrorCode ierr = MatCreateVecs(A, &err, NULL);
    CHKERRQ(ierr);
    ierr = MatMult(A, x, err);
    CHKERRQ(ierr);
    ierr = VecAXPY(err, -1.0, rhs);
    CHKERRQ(ierr);
    ierr = VecNorm(rhs, NORM_2, &norm_b);
    CHKERRQ(ierr);
    ierr = VecNorm(err, NORM_2, &norm_err);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, " error = %le / %le\n", norm_err, norm_b);
    CHKERRQ(ierr);
    ierr = VecDestroy(&err);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
