/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2021-01-08

   Copyright (C) 2021-     Centre National de la Recherche Scientifique

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

#ifndef _HPDDM_SLEPC_
#define _HPDDM_SLEPC_

#include <slepc.h>

#include "HPDDM_PETSc.hpp"
#include "HPDDM_eigensolver.hpp"

namespace HPDDM {
#ifdef MU_SLEPC
#undef HPDDM_CHECK_COARSEOPERATOR
#undef HPDDM_CHECK_SUBDOMAIN
#define HPDDM_CHECK_EIGENSOLVER
#include "HPDDM_preprocessor_check.hpp"
#define EIGENSOLVER HPDDM::Slepc
/* Class: Slepc
 *
 *  A class inheriting from <Eigensolver> to use <Slepc> for sparse eigenvalue problems.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Slepc : public Eigensolver<K> {
    public:
        Slepc(int n, int nu)                                                       : Eigensolver<K>(n, nu) { }
        Slepc(underlying_type<K> threshold, int n, int nu)                         : Eigensolver<K>(threshold, n, nu) { }
        Slepc(underlying_type<K> tol, underlying_type<K> threshold, int n, int nu) : Eigensolver<K>(tol, threshold, n, nu) { }
        /* Function: solve
         *
         *  Computes eigenvectors of the generalized eigenvalue problem Ax = l Bx.
         *
         * Parameters:
         *    A              - Left-hand side matrix.
         *    B              - Right-hand side matrix.
         *    ev             - Array of eigenvectors.
         *    communicator   - MPI communicator for selecting the threshold criterion. */
        template<template<class> class Solver>
        PetscErrorCode solve(MatrixCSR<K>* const& A, MatrixCSR<K>* const& B, K**& ev, const MPI_Comm& communicator, Solver<K>* const& = nullptr, std::ios_base::openmode mode = std::ios_base::out) {
            Mat P, Q;
            EPS eps;
            ST st;
            Vec vr, vi = nullptr;
            PetscInt nconv;
            PetscErrorCode ierr;
            K* evr = nullptr;
            const Option& opt = *Option::get();
            PetscFunctionBeginUser;
            if(Eigensolver<K>::_nu) {
                ierr = convert(A, P);CHKERRQ(ierr);
                ierr = convert(B, Q);CHKERRQ(ierr);
                ierr = EPSCreate(PETSC_COMM_SELF, &eps);CHKERRQ(ierr);
                ierr = EPSSetOperators(eps, P, Q);CHKERRQ(ierr);
                ierr = EPSSetTarget(eps, 0.0);CHKERRQ(ierr);
                ierr = EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
                ierr = EPSGetST(eps, &st);CHKERRQ(ierr);
                ierr = STSetType(st, STSINVERT);CHKERRQ(ierr);
                ierr = EPSSetOptionsPrefix(eps, std::string("slepc_" + std::string(HPDDM_PREFIX) + opt.getPrefix()).c_str());CHKERRQ(ierr);
                ierr = EPSSetDimensions(eps, Eigensolver<K>::_nu, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
                ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
                ierr = EPSSolve(eps);CHKERRQ(ierr);
                ierr = EPSGetConverged(eps, &nconv);CHKERRQ(ierr);
                Eigensolver<K>::_nu = std::min(static_cast<int>(nconv), Eigensolver<K>::_nu);
                if(Eigensolver<K>::_nu) {
                    evr = new K[Eigensolver<K>::_nu];
                    ev = new K*[Eigensolver<K>::_nu];
                    *ev = new K[Eigensolver<K>::_n * Eigensolver<K>::_nu];
                    for(unsigned short i = 1; i < Eigensolver<K>::_nu; ++i)
                        ev[i] = *ev + i * Eigensolver<K>::_n;
                    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, Eigensolver<K>::_n, nullptr, &vr);CHKERRQ(ierr);
                    if(std::is_same<PetscScalar, PetscReal>::value) {
                        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, Eigensolver<K>::_n, nullptr, &vi);CHKERRQ(ierr);
                    }
                    for(unsigned short i = 0; i < Eigensolver<K>::_nu; ++i) {
                        bool conjugate = false;
                        PetscScalar evi;
                        ierr = EPSGetEigenvalue(eps, i, evr + i, &evi);CHKERRQ(ierr);
                        ierr = VecPlaceArray(vr, ev[i]);CHKERRQ(ierr);
                        if(std::is_same<PetscScalar, PetscReal>::value && std::abs(evi) > HPDDM_EPS && i < Eigensolver<K>::_nu - 1) {
                            evr[i + 1] = evi;
                            ierr = VecPlaceArray(vi, ev[i + 1]);CHKERRQ(ierr);
                            conjugate = true;
                        }
                        ierr = EPSGetEigenvector(eps, i, vr, conjugate ? vi : nullptr);CHKERRQ(ierr);
                        ierr = VecResetArray(vr);CHKERRQ(ierr);
                        if(conjugate) {
                            ierr = VecResetArray(vi);CHKERRQ(ierr);
                            ++i;
                        }
                    }
                    ierr = VecDestroy(&vi);CHKERRQ(ierr);
                    ierr = VecDestroy(&vr);CHKERRQ(ierr);
                }
                ierr = EPSDestroy(&eps);CHKERRQ(ierr);
                ierr = MatDestroy(&Q);CHKERRQ(ierr);
                ierr = MatDestroy(&P);CHKERRQ(ierr);
                std::string name = Eigensolver<K>::dump(evr, ev, communicator, mode);
                ignore(name);
            }
            else {
                ev = new K*[1];
                *ev = nullptr;
            }
            if(Eigensolver<K>::_threshold > 0.0)
                Eigensolver<K>::selectNu(evr, ev, communicator);
            delete [] evr;
            PetscFunctionReturn(0);
        }
};
#endif // MU_SLEPC
} // HPDDM
#endif // _HPDDM_SLEPC_
