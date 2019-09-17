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
