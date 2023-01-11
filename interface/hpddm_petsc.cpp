 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
              Stefano Zampini <stefano.zampini@kaust.edu.sa>
        Date: 2019-07-23

   Copyright (C) 2019-     Centre National de la Recherche Scientifique

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

#include <petsc/private/pcimpl.h>

#ifdef PCHPDDM
#if PETSC_HAVE_SLEPC
#include <slepc.h>
#endif

#include <petsc/private/petschpddm.h>

#if HPDDM_SLEPC
static PetscBool SlepcInit = PETSC_TRUE;
#endif

PetscErrorCode PetscFinalize_HPDDM(void)
{
  PetscFunctionBegin;
#if HPDDM_SLEPC
  if (!SlepcInit) PetscCall(SlepcFinalize()); /* HPDDM initialized SLEPc, now shut it down */
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_hpddm_petsc(void)
{
  PetscFunctionBegin;
#if HPDDM_SLEPC
  PetscCall(SlepcInitialized(&SlepcInit));
  PetscCall(SlepcInitializeNoArguments());
#endif
  PetscCall(PetscRegisterFinalize(PetscFinalize_HPDDM));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PETSC_EXTERN PetscErrorCode KSPHPDDM_Internal(const char* prefix, const MPI_Comm& comm, PetscMPIInt redistribute, PetscInt n, PetscScalar* a, int lda, PetscScalar* b, int ldb, PetscInt k, PetscScalar* vr, const PetscBool symmetric)
{
  EPS          eps = nullptr;
  SVD          svd;
  Mat          X = nullptr, Y = nullptr;
  Vec          Vr = nullptr, Vi;
  PetscInt     nconv, i, nrow = n, rbegin = 0;
  PetscBLASInt info;
  PetscMPIInt  rank, size;
  MPI_Comm     subcomm;
  MPI_Group    world, worker;

  PetscFunctionBegin;
  if (redistribute <= 1) {
    PetscCall(MatCreateDense(PETSC_COMM_SELF, n, n, n, n, a, &X));
    PetscCall(MatDenseSetLDA(X, lda));
    if (b) {
      PetscCall(MatCreateDense(PETSC_COMM_SELF, n, n, n, n, b, &Y));
      PetscCall(MatDenseSetLDA(Y, ldb));
    }
  } else {
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCallMPI(MPI_Comm_group(comm, &world));
    PetscMPIInt* ranks = new PetscMPIInt[redistribute];
    std::iota(ranks, ranks + redistribute, 0);
    PetscCallMPI(MPI_Group_incl(world, redistribute, ranks, &worker));
    delete [] ranks;
    PetscCallMPI(MPI_Comm_create(comm, worker, &subcomm));
    PetscCallMPI(MPI_Group_free(&worker));
    PetscCallMPI(MPI_Group_free(&world));
    if (subcomm != MPI_COMM_NULL) {
      IS             row, col;
      PetscInt       ncol;
      const PetscInt *ia, *ja;
      char           type[256];
      PetscCall(MatCreate(subcomm, &X));
      PetscCall(MatSetSizes(X, PETSC_DECIDE, PETSC_DECIDE, n, n));
      PetscCall(MatSetOptionsPrefix(X, prefix));
      PetscObjectOptionsBegin((PetscObject)X);
#if defined(PETSC_HAVE_ELEMENTAL)
      std::string str(b ? MATELEMENTAL : MATDENSE);
#else
      std::string str(MATDENSE);
#endif
      str.copy(type, str.size() + 1);
      type[str.size()] = '\0';
      PetscCall(PetscOptionsFList("-mat_type", "Matrix type", "MatSetType", MatList, type, type, 256, nullptr));
      PetscOptionsEnd();
      nrow = PETSC_DECIDE;
      PetscCall(PetscSplitOwnership(subcomm, &nrow, &n));
      if (b) {
        PetscCall(MatCreate(subcomm, &Y));
        PetscCall(MatSetSizes(Y, PETSC_DECIDE, PETSC_DECIDE, n, n));
      }
      for (const Mat& m : { X, Y }) {
        if (m == Y && !b) continue;
        PetscCall(MatSetType(m, type));
        PetscCall(MatMPIAIJSetPreallocation(m, nrow, nullptr, n - nrow, nullptr));
        PetscCall(MatMPIDenseSetPreallocation(m, m == X ? a : b));
        PetscCall(MatSetUp(m));
        PetscCall(MatSetOption(m, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
        PetscCall(MatSetOption(m, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
      }
      if (std::string(reinterpret_cast<PetscObject>(X)->type_name).find(MATDENSE) == std::string::npos) {
        PetscCall(MatGetOwnershipIS(X, &row, &col));
        PetscCall(ISGetLocalSize(row, &nrow));
        PetscCall(ISGetIndices(row, &ia));
        PetscCall(ISGetLocalSize(col, &ncol));
        PetscCall(ISGetIndices(col, &ja));
        for (PetscInt j = 0; j < ncol; ++j) {
          for (PetscInt i = 0; i < nrow; ++i) {
            if (HPDDM::abs(a[ia[i] + ja[j] * lda]) > std::numeric_limits<HPDDM::underlying_type<PetscScalar>>::epsilon()) PetscCall(MatSetValues(X, 1, ia + i, 1, ja + j, a + ia[i] + ja[j] * lda, INSERT_VALUES));
            if (b && HPDDM::abs(b[ia[i] + ja[j] * ldb]) > std::numeric_limits<HPDDM::underlying_type<PetscScalar>>::epsilon()) PetscCall(MatSetValues(Y, 1, ia + i, 1, ja + j, b + ia[i] + ja[j] * ldb, INSERT_VALUES));
          }
        }
        PetscCall(ISRestoreIndices(col, &ja));
        PetscCall(ISRestoreIndices(row, &ia));
        PetscCall(ISDestroy(&row));
        PetscCall(ISDestroy(&col));
        PetscCall(MatCreateVecs(X, nullptr, &Vr));
        PetscCall(VecGetLocalSize(Vr, &nrow));
        PetscCall(VecGetOwnershipRange(Vr, &rbegin, nullptr));
        PetscCall(MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY));
        if (b) {
          PetscCall(MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY));
          PetscCall(MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY));
        }
      } else {
        Mat loc;
        PetscCall(MatGetOwnershipRange(X, &rbegin, nullptr));
        PetscCall(MatDenseGetLocalMatrix(X, &loc));
        PetscCall(MatDenseResetArray(loc));
        PetscCall(MatDenseSetLDA(loc, lda));
        PetscCall(MatDensePlaceArray(loc, a + rbegin));
        if (b) {
          PetscCall(MatDenseGetLocalMatrix(Y, &loc));
          PetscCall(MatDenseResetArray(loc));
          PetscCall(MatDenseSetLDA(loc, ldb));
          PetscCall(MatDensePlaceArray(loc, b + rbegin));
        }
      }
    }
  }
  if (X) {
    if (Y || !symmetric) {
      PetscCall(EPSCreate(PetscObjectComm((PetscObject)X), &eps));
      PetscCall(EPSSetOperators(eps, X, Y));
      if (redistribute <= 1) PetscCall(EPSSetType(eps, EPSLAPACK));
      PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE));
      PetscCall(EPSSetDimensions(eps, k, PETSC_DEFAULT, PETSC_DEFAULT));
      PetscCall(EPSSetOptionsPrefix(eps, prefix));
      if (symmetric) PetscCall(EPSSetProblemType(eps, EPS_GHEP));
      PetscCall(EPSSetFromOptions(eps));
      PetscCall(EPSSolve(eps));
      PetscCall(EPSGetConverged(eps, &nconv));
    } else {
      PetscCall(SVDCreate(PetscObjectComm((PetscObject)X), &svd));
      PetscCall(SVDSetOperators(svd, X, nullptr));
      if (redistribute <= 1) PetscCall(SVDSetType(svd, SVDLAPACK));
      PetscCall(SVDSetWhichSingularTriplets(svd, SVD_SMALLEST));
      PetscCall(SVDSetDimensions(svd, k, PETSC_DEFAULT, PETSC_DEFAULT));
      PetscCall(SVDSetOptionsPrefix(svd, prefix));
      PetscCall(SVDSetFromOptions(svd));
      PetscCall(SVDSolve(svd));
      PetscCall(SVDGetConverged(svd, &nconv));
    }
    if (!Vr) PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)X), 1, nrow, n, nullptr, &Vr));
    if (std::is_same<PetscReal, PetscScalar>::value) PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)X), 1, nrow, n, nullptr, &Vi));
    info = 0;
    for (i = 0; i < std::min(nconv, k); ++i) {
      PetscScalar eigr, eigi = PetscScalar();
      PetscCall(VecPlaceArray(Vr, vr + i * n + rbegin));
      if (eps) {
        if (std::is_same<PetscReal, PetscScalar>::value && i != k - 1) PetscCall(VecPlaceArray(Vi, vr + (i + 1) * n + rbegin));
        PetscCall(EPSGetEigenpair(eps, i - info, &eigr, std::is_same<PetscReal, PetscScalar>::value && i != k - 1 ? &eigi : nullptr, Vr, std::is_same<PetscReal, PetscScalar>::value && i != k - 1 ? Vi : nullptr));
        if (HPDDM::abs(eigi) > 100 * PETSC_MACHINE_EPSILON) {
          ++i;
          ++info;
        }
        if (std::is_same<PetscReal, PetscScalar>::value && i != k - 1) PetscCall(VecResetArray(Vi));
      } else PetscCall(SVDGetSingularTriplet(svd, i, nullptr, nullptr, Vr));
      PetscCall(VecResetArray(Vr));
    }
    PetscCheck(i == k, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unhandled mismatch %" PetscInt_FMT " != %" PetscInt_FMT, i, k);
    if (std::is_same<PetscReal, PetscScalar>::value) PetscCall(VecDestroy(&Vi));
    PetscCall(VecDestroy(&Vr));
    if (eps) PetscCall(EPSDestroy(&eps));
    else PetscCall(SVDDestroy(&svd));
    PetscCall(MatDestroy(&Y));
    PetscCall(MatDestroy(&X));
    if (redistribute > 1) {
      PetscCall(MPIU_Allreduce(MPI_IN_PLACE, vr, n * k, HPDDM::Wrapper<PetscScalar>::mpi_type(), MPI_SUM, subcomm));
      PetscCallMPI(MPI_Comm_free(&subcomm));
    }
  }
  if (redistribute > 1 && redistribute < size) PetscCallMPI(MPI_Bcast(vr, n * k, HPDDM::Wrapper<PetscScalar>::mpi_type(), 0, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
