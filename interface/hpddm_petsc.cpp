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

PETSC_EXTERN PetscErrorCode KSPHPDDM_Internal(const char* prefix, int n, PetscScalar* a, int lda, PetscScalar* b, int ldb, int k, PetscScalar* vr)
{
  EPS            eps;
  Mat            X, Y = NULL;
  Vec            Vr, Vi;
  PetscInt       nconv, i;
  PetscBLASInt   info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  ierr = EPSCreate(PETSC_COMM_SELF, &eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, X, Y);CHKERRQ(ierr);
  ierr = EPSSetType(eps, EPSLAPACK);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps, k, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(eps, prefix);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(eps, &nconv);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, nullptr, &Vr);CHKERRQ(ierr);
  if (std::is_same<PetscReal, PetscScalar>::value) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, nullptr, &Vi);CHKERRQ(ierr);
  }
  info = 0;
  for (i = 0; i < k; ++i) {
    PetscScalar eigr, eigi = PetscScalar();
    ierr = VecPlaceArray(Vr, vr + i * n);CHKERRQ(ierr);
    if (std::is_same<PetscReal, PetscScalar>::value && i != k - 1) {
      ierr = VecPlaceArray(Vi, vr + (i + 1) * n);CHKERRQ(ierr);
    }
    ierr = EPSGetEigenpair(eps, i - info, &eigr, std::is_same<PetscReal, PetscScalar>::value && i != k - 1 ? &eigi : nullptr, Vr, std::is_same<PetscReal, PetscScalar>::value && i != k - 1 ? Vi : nullptr);CHKERRQ(ierr);
    if (std::abs(eigi) > 100 * PETSC_MACHINE_EPSILON) {
      ++i;
      ++info;
    }
    if (std::is_same<PetscReal, PetscScalar>::value && i != k - 1) {
      ierr = VecResetArray(Vi);CHKERRQ(ierr);
    }
    ierr = VecResetArray(Vr);CHKERRQ(ierr);
  }
  if (i != k) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Unhandled mismatch %D != %D", i, k);
  if (std::is_same<PetscReal, PetscScalar>::value) {
    ierr = VecDestroy(&Vi);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&Vr);CHKERRQ(ierr);
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&Y);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
