 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
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
#if HPDDM_SLEPC
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
#if HPDDM_SLEPC
  if (!SlepcInit) { /* HPDDM initialized SLEPc, now shut it down */
    ierr = SlepcFinalize();CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_hpddm_petsc(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if HPDDM_SLEPC
  ierr = SlepcInitialized(&SlepcInit);CHKERRQ(ierr);
  ierr = SlepcInitializeNoArguments();CHKERRQ(ierr);
#endif
  ierr = PetscRegisterFinalize(PetscFinalize_HPDDM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PETSC_EXTERN PetscErrorCode KSPHPDDM_Internal(const char* prefix, const MPI_Comm& comm, PetscMPIInt redistribute, PetscInt n, PetscScalar* a, int lda, PetscScalar* b, int ldb, PetscInt k, PetscScalar* vr, const PetscBool symmetric)
{
  EPS            eps = nullptr;
  SVD            svd;
  Mat            X = nullptr, Y = nullptr;
  Vec            Vr = nullptr, Vi;
  PetscInt       nconv, i, nrow = n, rbegin = 0;
  PetscBLASInt   info;
  PetscMPIInt    rank, size;
  MPI_Comm       subcomm;
  MPI_Group      world, worker;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (redistribute <= 1) {
    ierr = MatCreateDense(PETSC_COMM_SELF, n, n, n, n, a, &X);CHKERRQ(ierr);
#if PETSC_VERSION_GE(3, 14, 0)
    ierr = MatDenseSetLDA(X, lda);CHKERRQ(ierr);
#else
    ierr = MatSeqDenseSetLDA(X, lda);CHKERRQ(ierr);
#endif
    if (b) {
      ierr = MatCreateDense(PETSC_COMM_SELF, n, n, n, n, b, &Y);CHKERRQ(ierr);
#if PETSC_VERSION_GE(3, 14, 0)
      ierr = MatDenseSetLDA(Y, ldb);CHKERRQ(ierr);
#else
      ierr = MatSeqDenseSetLDA(Y, ldb);CHKERRQ(ierr);
#endif
    }
  } else {
    ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
    ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
    ierr = MPI_Comm_group(comm, &world);CHKERRMPI(ierr);
    PetscMPIInt* ranks = new PetscMPIInt[redistribute];
    std::iota(ranks, ranks + redistribute, 0);
    ierr = MPI_Group_incl(world, redistribute, ranks, &worker);CHKERRMPI(ierr);
    delete [] ranks;
    ierr = MPI_Comm_create(comm, worker, &subcomm);CHKERRMPI(ierr);
    ierr = MPI_Group_free(&worker);CHKERRMPI(ierr);
    ierr = MPI_Group_free(&world);CHKERRMPI(ierr);
    if (subcomm != MPI_COMM_NULL) {
      IS             row, col;
      PetscInt       ncol;
      const PetscInt *ia, *ja;
      char           type[256];
      ierr = MatCreate(subcomm, &X);CHKERRQ(ierr);
      ierr = MatSetSizes(X, PETSC_DECIDE, PETSC_DECIDE, n, n);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(X, prefix);CHKERRQ(ierr);
      ierr = PetscObjectOptionsBegin((PetscObject)X);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
      std::string str(b ? MATELEMENTAL : MATDENSE);
#else
      std::string str(MATDENSE);
#endif
      str.copy(type, str.size() + 1);
      type[str.size()] = '\0';
      ierr = PetscOptionsFList("-mat_type", "Matrix type", "MatSetType", MatList, type, type, 256, nullptr);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
      nrow = PETSC_DECIDE;
      ierr = PetscSplitOwnership(subcomm, &nrow, &n);CHKERRQ(ierr);
      if (b) {
        ierr = MatCreate(subcomm, &Y);CHKERRQ(ierr);
        ierr = MatSetSizes(Y, PETSC_DECIDE, PETSC_DECIDE, n, n);CHKERRQ(ierr);
      }
      for (const Mat& m : { X, Y }) {
        if (m == Y && !b)
          continue;
        ierr = MatSetType(m, type);CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(m, nrow, nullptr, n - nrow, nullptr);CHKERRQ(ierr);
        ierr = MatMPIDenseSetPreallocation(m, m == X ? a : b);CHKERRQ(ierr);
        ierr = MatSetUp(m);CHKERRQ(ierr);
        ierr = MatSetOption(m, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
        ierr = MatSetOption(m, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
      }
      if (std::string(reinterpret_cast<PetscObject>(X)->type_name).find(MATDENSE) == std::string::npos) {
        ierr = MatGetOwnershipIS(X, &row, &col);CHKERRQ(ierr);
        ierr = ISGetLocalSize(row, &nrow);CHKERRQ(ierr);
        ierr = ISGetIndices(row, &ia);CHKERRQ(ierr);
        ierr = ISGetLocalSize(col, &ncol);CHKERRQ(ierr);
        ierr = ISGetIndices(col, &ja);CHKERRQ(ierr);
        for (PetscInt j = 0; j < ncol; ++j) {
          for (PetscInt i = 0; i < nrow; ++i) {
            if (std::abs(a[ia[i] + ja[j] * lda]) > std::numeric_limits<HPDDM::underlying_type<PetscScalar>>::epsilon()) {
              ierr = MatSetValues(X, 1, ia + i, 1, ja + j, a + ia[i] + ja[j] * lda, INSERT_VALUES);CHKERRQ(ierr);
            }
            if (b && std::abs(b[ia[i] + ja[j] * ldb]) > std::numeric_limits<HPDDM::underlying_type<PetscScalar>>::epsilon()) {
              ierr = MatSetValues(Y, 1, ia + i, 1, ja + j, b + ia[i] + ja[j] * ldb, INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        }
        ierr = ISRestoreIndices(col, &ja);CHKERRQ(ierr);
        ierr = ISRestoreIndices(row, &ia);CHKERRQ(ierr);
        ierr = ISDestroy(&row);CHKERRQ(ierr);
        ierr = ISDestroy(&col);CHKERRQ(ierr);
        ierr = MatCreateVecs(X, nullptr, &Vr);CHKERRQ(ierr);
        ierr = VecGetLocalSize(Vr, &nrow);CHKERRQ(ierr);
        ierr = VecGetOwnershipRange(Vr, &rbegin, nullptr);CHKERRQ(ierr);
      } else {
        Mat loc;
        ierr = MatGetOwnershipRange(X, &rbegin, nullptr);CHKERRQ(ierr);
        ierr = MatDenseGetLocalMatrix(X, &loc);CHKERRQ(ierr);
        ierr = MatDenseResetArray(loc);CHKERRQ(ierr);
#if PETSC_VERSION_GE(3, 14, 0)
        ierr = MatDenseSetLDA(loc, lda);CHKERRQ(ierr);
#else
        ierr = MatSeqDenseSetLDA(loc, lda);CHKERRQ(ierr);
#endif
        ierr = MatDensePlaceArray(loc, a + rbegin);CHKERRQ(ierr);
        if (b) {
          ierr = MatDenseGetLocalMatrix(Y, &loc);CHKERRQ(ierr);
          ierr = MatDenseResetArray(loc);CHKERRQ(ierr);
#if PETSC_VERSION_GE(3, 14, 0)
          ierr = MatDenseSetLDA(loc, ldb);CHKERRQ(ierr);
#else
          ierr = MatSeqDenseSetLDA(loc, ldb);CHKERRQ(ierr);
#endif
          ierr = MatDensePlaceArray(loc, b + rbegin);CHKERRQ(ierr);
        }
      }
      ierr = MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      if (b) {
        ierr = MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
    }
  }
  if (X) {
    if (Y || !symmetric) {
      ierr = EPSCreate(PetscObjectComm((PetscObject)X), &eps);CHKERRQ(ierr);
      ierr = EPSSetOperators(eps, X, Y);CHKERRQ(ierr);
      if (redistribute <= 1) {
        ierr = EPSSetType(eps, EPSLAPACK);CHKERRQ(ierr);
      }
      ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);
      ierr = EPSSetDimensions(eps, k, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = EPSSetOptionsPrefix(eps, prefix);CHKERRQ(ierr);
      if (symmetric) {
        ierr = EPSSetProblemType(eps, EPS_GHEP);CHKERRQ(ierr);
      }
      ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
      ierr = EPSSolve(eps);CHKERRQ(ierr);
      ierr = EPSGetConverged(eps, &nconv);CHKERRQ(ierr);
    } else {
      ierr = SVDCreate(PetscObjectComm((PetscObject)X), &svd);CHKERRQ(ierr);
      ierr = SVDSetOperators(svd, X, nullptr);CHKERRQ(ierr);
      if (redistribute <= 1) {
        ierr = SVDSetType(svd, SVDLAPACK);CHKERRQ(ierr);
      }
      ierr = SVDSetWhichSingularTriplets(svd, SVD_SMALLEST);CHKERRQ(ierr);
      ierr = SVDSetDimensions(svd, k, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = SVDSetOptionsPrefix(svd, prefix);CHKERRQ(ierr);
      ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);
      ierr = SVDSolve(svd);CHKERRQ(ierr);
      ierr = SVDGetConverged(svd, &nconv);CHKERRQ(ierr);
    }
    if (!Vr) {
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)X), 1, nrow, n, nullptr, &Vr);CHKERRQ(ierr);
    }
    if (std::is_same<PetscReal, PetscScalar>::value) {
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)X), 1, nrow, n, nullptr, &Vi);CHKERRQ(ierr);
    }
    info = 0;
    for (i = 0; i < std::min(nconv, k); ++i) {
      PetscScalar eigr, eigi = PetscScalar();
      ierr = VecPlaceArray(Vr, vr + i * n + rbegin);CHKERRQ(ierr);
      if (eps) {
        if (std::is_same<PetscReal, PetscScalar>::value && i != k - 1) {
          ierr = VecPlaceArray(Vi, vr + (i + 1) * n + rbegin);CHKERRQ(ierr);
        }
        ierr = EPSGetEigenpair(eps, i - info, &eigr, std::is_same<PetscReal, PetscScalar>::value && i != k - 1 ? &eigi : nullptr, Vr, std::is_same<PetscReal, PetscScalar>::value && i != k - 1 ? Vi : nullptr);CHKERRQ(ierr);
        if (std::abs(eigi) > 100 * PETSC_MACHINE_EPSILON) {
          ++i;
          ++info;
        }
        if (std::is_same<PetscReal, PetscScalar>::value && i != k - 1) {
          ierr = VecResetArray(Vi);CHKERRQ(ierr);
        }
      } else {
        ierr = SVDGetSingularTriplet(svd, i, nullptr, nullptr, Vr);CHKERRQ(ierr);
      }
      ierr = VecResetArray(Vr);CHKERRQ(ierr);
    }
    if (i != k) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Unhandled mismatch %" PetscInt_FMT " != %" PetscInt_FMT, i, k); // LCOV_EXCL_LINE
    if (std::is_same<PetscReal, PetscScalar>::value) {
      ierr = VecDestroy(&Vi);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&Vr);CHKERRQ(ierr);
    if (eps) {
      ierr = EPSDestroy(&eps);CHKERRQ(ierr);
    } else {
      ierr = SVDDestroy(&svd);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Y);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    if (redistribute > 1) {
      ierr = MPIU_Allreduce(MPI_IN_PLACE, vr, n * k, HPDDM::Wrapper<PetscScalar>::mpi_type(), MPI_SUM, subcomm);CHKERRMPI(ierr);
      ierr = MPI_Comm_free(&subcomm);CHKERRMPI(ierr);
    }
  }
  if (redistribute > 1 && redistribute < size) {
    ierr = MPI_Bcast(vr, n * k, HPDDM::Wrapper<PetscScalar>::mpi_type(), 0, comm);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}
