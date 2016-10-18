 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2016-08-04

   Copyright (C) 2016-     Centre National de la Recherche Scientifique

   Note:      Reference PETSc implementation available at
                                                         http://goo.gl/2sXsYP
              Contributed by
                                                            Andrei Draganescu

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

#include <HPDDM.h>
#include <petsc.h>
#define SIZE_ARRAY_NU 4

PetscErrorCode ComputeMatrix(DM, Mat, Mat);
PetscErrorCode ComputeRHS(DM, Vec, PetscScalar);
PetscErrorCode ComputeError(Mat, Vec, Vec);

struct HpddmCustomOperator {
    Vec _b;
    Vec _x;
    Mat _A;
    PC _M;
    void (*_mv)(const HpddmCustomOperator* const, const K*, K*, int);
    void (*_precond)(const HpddmCustomOperator* const, const K*, K*, int);
};
void mv(const HpddmCustomOperator* const H, const K* in, K* out, int mu)
{
    int n, nu;
    MatGetLocalSize(H->_A, &n, NULL);
    for (nu = 0; nu < mu; ++nu) {
        VecPlaceArray(H->_b, in + nu * n);
        VecPlaceArray(H->_x, out + nu * n);
        MatMult(H->_A, H->_b, H->_x);
        VecResetArray(H->_x);
        VecResetArray(H->_b);
    }
}
void precond(const HpddmCustomOperator* const H, const K* in, K* out, int mu)
{
    int n, nu;
    MatGetLocalSize(H->_A, &n, NULL);
    for (nu = 0; nu < mu; ++nu) {
        VecPlaceArray(H->_b, in + nu * n);
        VecPlaceArray(H->_x, out + nu * n);
        PCApply(H->_M, H->_b, H->_x);
        VecResetArray(H->_x);
        VecResetArray(H->_b);
    }
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv)
{
    PC pc;
    DM da;
    PetscErrorCode ierr;
    Vec x, rhs;
    Mat A, jac;
    PetscInitialize(&argc, &argv, NULL, NULL);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Laplacian in 2D", "");
    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);
    const HpddmOption* const opt = HpddmOptionGet();
    {
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        HpddmOptionParse(opt, argc, argv, rank == 0);
        if (rank) HpddmOptionRemove(opt, "verbosity");
    }
    MPI_Barrier(PETSC_COMM_WORLD);
    double time = MPI_Wtime();
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, -10, -10, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                        0, 0, &da);
    CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &rhs);
    CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &x);
    CHKERRQ(ierr);
    ierr = DMCreateMatrix(da, &A);
    CHKERRQ(ierr);
    ierr = DMCreateMatrix(da, &jac);
    CHKERRQ(ierr);
    ierr = ComputeMatrix(da, jac, A);
    CHKERRQ(ierr);
    MPI_Barrier(PETSC_COMM_WORLD);
    time = MPI_Wtime() - time;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "--- Mat assembly = %f\n", time);
    CHKERRQ(ierr);
    MPI_Barrier(PETSC_COMM_WORLD);
    time = MPI_Wtime();
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetDM(ksp, da);
    CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A);
    CHKERRQ(ierr);
    ierr = KSPSetDMActive(ksp, PETSC_FALSE);
    CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);
    CHKERRQ(ierr);
    MPI_Barrier(PETSC_COMM_WORLD);
    time = MPI_Wtime() - time;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "--- PC setup = %f\n", time);
    CHKERRQ(ierr);
    PetscScalar nus[SIZE_ARRAY_NU] = {0.1, 10.0, 0.001, 100.0};
    float t_time[SIZE_ARRAY_NU];
    int t_its[SIZE_ARRAY_NU];
    int i;
    {
        {
            ierr = KSPSolve(ksp, rhs, x);
            CHKERRQ(ierr);
        }
        for (i = 0; i < SIZE_ARRAY_NU; ++i) {
            ierr = VecZeroEntries(x);
            CHKERRQ(ierr);
            ierr = ComputeRHS(da, rhs, nus[i]);
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
        }
        for (i = 0; i < SIZE_ARRAY_NU; ++i) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t%d\t%f\n", i + 1, t_its[i], t_time[i]);
            CHKERRQ(ierr);
            if (i > 0) {
                t_its[0] += t_its[i];
                t_time[0] += t_time[i];
            }
        }
        if (SIZE_ARRAY_NU > 1) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, "------------------------\n\t%d\t%f\n", t_its[0], t_time[0]);
            CHKERRQ(ierr);
        }
    }
    {
        ierr = KSPGetPC(ksp, &pc);
        CHKERRQ(ierr);
        HpddmCustomOperator H;
        H._A = A;
        H._M = pc;
        H._mv = mv;
        H._precond = precond;
        H._b = rhs;
        H._x = x;
        int n;
        MatGetLocalSize(A, &n, NULL);
        {
            ierr = VecZeroEntries(x);
            K* pt_rhs;
            K* pt_x;
            VecGetArray(rhs, &pt_rhs);
            VecGetArray(x, &pt_x);
            int previous = HpddmOptionVal(opt, "verbosity");
            if (previous > 0) HpddmOptionRemove(opt, "verbosity");
            HpddmCustomOperatorSolve(&H, n, H._mv, H._precond, pt_rhs, pt_x, 1, &PETSC_COMM_WORLD);
            if (previous > 0) {
                char buffer[20];
                snprintf(buffer, 20, "%d", previous);
                char* concat = malloc(strlen("-hpddm_verbosity ") + strlen(buffer) + 1);
                strcpy(concat, "-hpddm_verbosity ");
                strcat(concat, buffer);
                HpddmOptionParseString(opt, concat);
                free(concat);
            }
            VecRestoreArray(x, &pt_x);
            VecRestoreArray(rhs, &pt_rhs);
            previous = HpddmOptionVal(opt, "krylov_method");
            if(previous == 4 || previous == 5) HpddmDestroyRecycling();
        }
        for (i = 0; i < SIZE_ARRAY_NU; ++i) {
            ierr = VecZeroEntries(x);
            CHKERRQ(ierr);
            ierr = ComputeRHS(da, rhs, nus[i]);
            CHKERRQ(ierr);
            K* pt_rhs;
            K* pt_x;
            VecGetArray(rhs, &pt_rhs);
            VecGetArray(x, &pt_x);
            MPI_Barrier(PETSC_COMM_WORLD);
            time = MPI_Wtime();
            t_its[i] = HpddmCustomOperatorSolve(&H, n, H._mv, H._precond, pt_rhs, pt_x, 1, &PETSC_COMM_WORLD);
            MPI_Barrier(PETSC_COMM_WORLD);
            t_time[i] = MPI_Wtime() - time;
            VecRestoreArray(x, &pt_x);
            VecRestoreArray(rhs, &pt_rhs);
            ierr = ComputeError(A, rhs, x);
            CHKERRQ(ierr);
        }
        for (i = 0; i < SIZE_ARRAY_NU; ++i) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t%d\t%f\n", i + 1, t_its[i], t_time[i]);
            CHKERRQ(ierr);
            if (i > 0) {
                t_its[0] += t_its[i];
                t_time[0] += t_time[i];
            }
        }
        if (SIZE_ARRAY_NU > 1) {
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
    ierr = MatDestroy(&jac);
    CHKERRQ(ierr);
    ierr = DMDestroy(&da);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DM da, Vec b, PetscScalar nu)
{
    PetscErrorCode ierr;
    PetscInt i, j, mx, my, xm, ym, xs, ys;
    PetscScalar Hx, Hy;
    PetscScalar** array;
    PetscFunctionBeginUser;
    ierr = DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    CHKERRQ(ierr);
    Hx = 1.0 / (PetscReal)(mx);
    Hy = 1.0 / (PetscReal)(my);
    ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, b, &array);
    CHKERRQ(ierr);
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            array[j][i] = PetscExpScalar(-(((PetscReal)i + 0.5) * Hx) * (((PetscReal)i + 0.5) * Hx) / nu) *
                          PetscExpScalar(-(((PetscReal)j + 0.5) * Hy) * (((PetscReal)j + 0.5) * Hy) / nu) * Hx * Hy * nu;
        }
    }
    ierr = DMDAVecRestoreArray(da, b, &array);
    CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);
    CHKERRQ(ierr);
    MatNullSpace nullspace;
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
    CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace, b);
    CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(DM da, Mat J, Mat jac)
{
    PetscErrorCode ierr;
    PetscInt i, j, mx, my, xm, ym, xs, ys, num, numi, numj;
    PetscScalar v[5], Hx, Hy, HydHx, HxdHy;
    MatStencil row, col[5];
    PetscFunctionBeginUser;
    ierr = DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    CHKERRQ(ierr);
    Hx = 1.0 / (PetscReal)(mx);
    Hy = 1.0 / (PetscReal)(my);
    HxdHy = Hx / Hy;
    HydHx = Hy / Hx;
    ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);
    CHKERRQ(ierr);
    for (j = ys; j < ys + ym; j++) {
        for (i = xs; i < xs + xm; i++) {
            row.i = i;
            row.j = j;
            if (i == 0 || j == 0 || i == mx - 1 || j == my - 1) {
                num = 0;
                numi = 0;
                numj = 0;
                if (j != 0) {
                    v[num] = -HxdHy;
                    col[num].i = i;
                    col[num].j = j - 1;
                    num++;
                    numj++;
                }
                if (i != 0) {
                    v[num] = -HydHx;
                    col[num].i = i - 1;
                    col[num].j = j;
                    num++;
                    numi++;
                }
                if (i != mx - 1) {
                    v[num] = -HydHx;
                    col[num].i = i + 1;
                    col[num].j = j;
                    num++;
                    numi++;
                }
                if (j != my - 1) {
                    v[num] = -HxdHy;
                    col[num].i = i;
                    col[num].j = j + 1;
                    num++;
                    numj++;
                }
                v[num] = (PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx;
                col[num].i = i;
                col[num].j = j;
                num++;
                ierr = MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES);
                CHKERRQ(ierr);
            }
            else {
                v[0] = -HxdHy;
                col[0].i = i;
                col[0].j = j - 1;
                v[1] = -HydHx;
                col[1].i = i - 1;
                col[1].j = j;
                v[2] = 2.0 * (HxdHy + HydHx);
                col[2].i = i;
                col[2].j = j;
                v[3] = -HydHx;
                col[3].i = i + 1;
                col[3].j = j;
                v[4] = -HxdHy;
                col[4].i = i;
                col[4].j = j + 1;
                ierr = MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES);
                CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    MatNullSpace nullspace;
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
    CHKERRQ(ierr);
    ierr = MatSetNullSpace(J, nullspace);
    CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeError"
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
