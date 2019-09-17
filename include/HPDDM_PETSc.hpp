 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2018-02-05

   Copyright (C) 2018-     Centre National de la Recherche Scientifique

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

#ifndef _HPDDM_PETSC_
#define _HPDDM_PETSC_

namespace HPDDM {
struct PETScOperator : public EmptyOperator<PetscScalar, PetscInt> {
    typedef EmptyOperator<PetscScalar, PetscInt> super;
    const KSP     _ksp;
    const PetscInt _bs;
    PetscErrorCode (*const _apply)(PC, Mat, Mat);
    PETScOperator(const PETScOperator&) = delete;
    PETScOperator(const KSP& ksp, PetscInt n, PetscInt bs, PetscErrorCode (*apply)(PC, Mat, Mat) = nullptr) : super(bs * n), _ksp(ksp), _bs(bs), _apply(apply) {
        PC pc;
        KSPGetPC(ksp, &pc);
        PCSetFromOptions(pc);
        PCSetUp(pc);
    }
    void GMV(const PetscScalar* const in, PetscScalar* const out, const int& mu = 1) const {
        Mat A;
        KSPGetOperators(_ksp, &A, NULL);
        PetscInt N;
        MatGetSize(A, &N, NULL);
        PetscBool hasMatMatMult;
        MatHasOperation(A, MATOP_MATMAT_MULT, &hasMatMatMult);
        MPI_Comm comm;
        PetscObjectGetComm((PetscObject)A, &comm);
        PetscMPIInt size;
        MPI_Comm_size(comm, &size);
        if(mu == 1 || !hasMatMatMult || size == 1) {
            Vec right, left;
            VecCreateMPIWithArray(comm, _bs, super::_n, N, NULL, &right);
            VecCreateMPIWithArray(comm, _bs, super::_n, N, NULL, &left);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                VecPlaceArray(right, in + nu * super::_n);
                VecPlaceArray(left, out + nu * super::_n);
                MatMult(A, right, left);
                VecResetArray(left);
                VecResetArray(right);
            }
            VecDestroy(&right);
            VecDestroy(&left);
        }
        else {
            PetscInt N;
            MatGetSize(A, &N, NULL);
            Mat B, C;
            MatCreateDense(comm, super::_n, PETSC_DECIDE, N, mu, const_cast<PetscScalar*>(in), &B);
            MatCreateDense(comm, super::_n, PETSC_DECIDE, N, mu, out, &C);
            MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C);
            MatDestroy(&C);
            MatDestroy(&B);
        }
    }
    template<bool = true>
    void apply(const PetscScalar* const in, PetscScalar* const out, const unsigned short& mu = 1, PetscScalar* = nullptr, const unsigned short& = 0) const {
        PC pc;
        KSPGetPC(_ksp, &pc);
        PCType type;
        PCGetType(pc, &type);
        PetscBool isBJacobi, isSolve;
        if(mu > 1) {
            PetscStrcmp(type, PCBJACOBI, &isBJacobi);
            if(!isBJacobi) {
                for(const PCType& t : { PCLU, PCCHOLESKY, PCILU, PCICC }) {
                    PetscStrcmp(type, t, &isSolve);
                    if(isSolve)
                        break;
                }
            }
            else
                isSolve = PETSC_FALSE;
        }
        else
            isBJacobi = isSolve = PETSC_FALSE;
        Mat F = NULL;
        if(isBJacobi) {
            KSP* subksp;
            PetscInt n_local, first_local;
            PCBJacobiGetSubKSP(pc, &n_local, &first_local, &subksp);
            if(n_local > 1)
                isBJacobi = PETSC_FALSE;
            else {
                KSPSetUp(subksp[0]);
                PC subpc;
                KSPGetPC(subksp[0], &subpc);
                PCGetType(subpc, &type);
                isBJacobi = PETSC_FALSE;
                for(const PCType& t : { PCLU, PCCHOLESKY, PCILU, PCICC }) {
                    PetscStrcmp(type, t, &isBJacobi);
                    if(isBJacobi) {
                        PCFactorGetMatrix(subpc, &F);
                        break;
                    }
                }
            }
        }
        else if(isSolve)
            PCFactorGetMatrix(pc, &F);
        MPI_Comm comm;
        if(F) {
            PetscInt N;
            MatGetSize(F, &N, NULL);
            Mat B, C;
            PetscObjectGetComm((PetscObject)F, &comm);
            MatCreateDense(comm, super::_n, PETSC_DECIDE, N, mu, const_cast<PetscScalar*>(in), &B);
            MatCreateDense(comm, super::_n, PETSC_DECIDE, N, mu, out, &C);
            MatMatSolve(F, B, C);
            MatDestroy(&C);
            MatDestroy(&B);
        }
        else {
            PetscObjectGetComm((PetscObject)pc, &comm);
            Mat A;
            KSPGetOperators(_ksp, &A, NULL);
            PetscInt N;
            MatGetSize(A, &N, NULL);
            if(_apply) {
                Mat B, C;
                MatCreateDense(comm, super::_n, PETSC_DECIDE, N, mu, const_cast<PetscScalar*>(in), &B);
                MatCreateDense(comm, super::_n, PETSC_DECIDE, N, mu, out, &C);
                _apply(pc, B, C);
                MatDestroy(&C);
                MatDestroy(&B);
            }
            else {
                Vec right, left;
                VecCreateMPIWithArray(comm, _bs, super::_n, N, NULL, &right);
                VecCreateMPIWithArray(comm, _bs, super::_n, N, NULL, &left);
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    VecPlaceArray(right, in + nu * super::_n);
                    VecPlaceArray(left, out + nu * super::_n);
                    PCApply(pc, right, left);
                    VecResetArray(left);
                    VecResetArray(right);
                }
                VecDestroy(&right);
                VecDestroy(&left);
            }
        }
    }
    std::string prefix() const {
        const char* prefix = nullptr;
        if(_ksp)
            KSPGetOptionsPrefix(_ksp, &prefix);
        return prefix ? prefix : "";
    }
};
} // HPDDM
#endif // _HPDDM_PETSC_
