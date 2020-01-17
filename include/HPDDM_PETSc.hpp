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

#include <petsc.h>

#include "HPDDM_iterative.hpp"

namespace HPDDM {
    static inline PetscErrorCode apply(KSP ksp, PetscInt bs, const PetscScalar* const in, PetscScalar* const out, const unsigned short& mu = 1, PetscErrorCode (*const apply)(PC, Mat, Mat) = nullptr) {
        PetscErrorCode ierr;
        PC pc;
        ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
        PCType type;
        ierr = PCGetType(pc, &type);CHKERRQ(ierr);
        PetscBool isBJacobi, isSolve;
        if(mu > 1) {
            ierr = PetscStrcmp(type, PCBJACOBI, &isBJacobi);CHKERRQ(ierr);
            if(!isBJacobi) {
                for(const PCType& t : { PCLU, PCCHOLESKY, PCILU, PCICC }) {
                    ierr = PetscStrcmp(type, t, &isSolve);CHKERRQ(ierr);
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
            ierr = PCBJacobiGetSubKSP(pc, &n_local, &first_local, &subksp);CHKERRQ(ierr);
            if(n_local > 1)
                isBJacobi = PETSC_FALSE;
            else {
                ierr = KSPSetUp(subksp[0]);CHKERRQ(ierr);
                PC subpc;
                ierr = KSPGetPC(subksp[0], &subpc);CHKERRQ(ierr);
                ierr = PCGetType(subpc, &type);CHKERRQ(ierr);
                isBJacobi = PETSC_FALSE;
                for(const PCType& t : { PCLU, PCCHOLESKY, PCILU, PCICC }) {
                    ierr = PetscStrcmp(type, t, &isBJacobi);CHKERRQ(ierr);
                    if(isBJacobi) {
                        ierr = PCFactorGetMatrix(subpc, &F);CHKERRQ(ierr);
                        break;
                    }
                }
            }
        }
        else if(isSolve) {
            ierr = PCFactorGetMatrix(pc, &F);CHKERRQ(ierr);
        }
        MPI_Comm comm;
        if(F) {
            PetscInt n, N;
            ierr = MatGetLocalSize(F, &n, NULL);CHKERRQ(ierr);
            ierr = MatGetSize(F, &N, NULL);CHKERRQ(ierr);
            Mat B, C;
            ierr = PetscObjectGetComm((PetscObject)F, &comm);CHKERRQ(ierr);
            ierr = MatCreateDense(comm, n, PETSC_DECIDE, N, mu, const_cast<PetscScalar*>(in), &B);CHKERRQ(ierr);
            ierr = MatCreateDense(comm, n, PETSC_DECIDE, N, mu, out, &C);CHKERRQ(ierr);
            ierr = MatMatSolve(F, B, C);CHKERRQ(ierr);
            ierr = MatDestroy(&C);CHKERRQ(ierr);
            ierr = MatDestroy(&B);CHKERRQ(ierr);
        }
        else {
            ierr = PetscObjectGetComm((PetscObject)pc, &comm);CHKERRQ(ierr);
            Mat A;
            ierr = KSPGetOperators(ksp, &A, NULL);CHKERRQ(ierr);
            PetscInt n, N;
            ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
            ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
            if(apply) {
                Mat B, C;
                ierr = MatCreateDense(comm, n, PETSC_DECIDE, N, mu, const_cast<PetscScalar*>(in), &B);CHKERRQ(ierr);
                ierr = MatCreateDense(comm, n, PETSC_DECIDE, N, mu, out, &C);CHKERRQ(ierr);
                ierr = apply(pc, B, C);CHKERRQ(ierr);
                ierr = MatDestroy(&C);CHKERRQ(ierr);
                ierr = MatDestroy(&B);CHKERRQ(ierr);
            }
            else {
                Vec right, left;
                ierr = VecCreateMPIWithArray(comm, bs, n, N, NULL, &right);CHKERRQ(ierr);
                ierr = VecCreateMPIWithArray(comm, bs, n, N, NULL, &left);CHKERRQ(ierr);
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    ierr = VecPlaceArray(right, in + nu * n);CHKERRQ(ierr);
                    ierr = VecPlaceArray(left, out + nu * n);CHKERRQ(ierr);
                    ierr = PCApply(pc, right, left);CHKERRQ(ierr);
                    ierr = VecResetArray(left);CHKERRQ(ierr);
                    ierr = VecResetArray(right);CHKERRQ(ierr);
                }
                ierr = VecDestroy(&right);CHKERRQ(ierr);
                ierr = VecDestroy(&left);CHKERRQ(ierr);
            }
        }
        PetscFunctionReturn(0);
    }
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
        MatHasOperation(A, MATOP_MAT_MULT, &hasMatMatMult);
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
    template<bool = false>
    PetscErrorCode apply(const PetscScalar* const in, PetscScalar* const out, const unsigned short& mu = 1, PetscScalar* = nullptr, const unsigned short& = 0) const {
        PetscErrorCode ierr = HPDDM::apply(_ksp, _bs, in, out, mu, _apply);CHKERRQ(ierr);
        PetscFunctionReturn(0);
    }
    std::string prefix() const {
        const char* prefix = nullptr;
        if(_ksp)
            KSPGetOptionsPrefix(_ksp, &prefix);
        return prefix ? prefix : "";
    }
};

#ifdef PETSCSUB
#undef HPDDM_CHECK_COARSEOPERATOR
#define HPDDM_CHECK_SUBDOMAIN
#include "HPDDM_preprocessor_check.hpp"
#define SUBDOMAIN HPDDM::PetscSub
template<class K>
class PetscSub {
    private:
        KSP _ksp;
    public:
        PetscSub() : _ksp() { }
        PetscSub(const PetscSub&) = delete;
        ~PetscSub() { dtor(); }
        static constexpr char _numbering = 'C';
        PetscErrorCode dtor() {
            PetscErrorCode ierr = KSPDestroy(&_ksp);CHKERRQ(ierr);
            PetscFunctionReturn(0);
        }
        template<char N = HPDDM_NUMBERING>
        PetscErrorCode numfact(MatrixCSR<K>* const& A, bool detection = false, K* const& schur = nullptr) {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            Mat P;
            PC pc;
            PetscErrorCode ierr;
            const Option& opt = *Option::get();
            if(!_ksp) {
                ierr = KSPCreate(PETSC_COMM_SELF, &_ksp);CHKERRQ(ierr);
            }
            if(N == 'C') {
                ierr = MatCreate(PETSC_COMM_SELF, &P);CHKERRQ(ierr);
                ierr = MatSetSizes(P, A->_n, A->_m, A->_n, A->_m);CHKERRQ(ierr);
                if(!A->_sym) {
                    ierr = MatSetType(P, MATSEQAIJ);CHKERRQ(ierr);
                    ierr = MatSeqAIJSetPreallocationCSR(P, A->_ia, A->_ja, A->_a);CHKERRQ(ierr);
                }
                else {
                    PetscInt* I = new PetscInt[A->_n + 1];
                    PetscInt* J = new PetscInt[A->_nnz];
                    PetscScalar* C = new PetscScalar[A->_nnz];
                    Wrapper<K>::template csrcsc<N, 'C'>(&A->_n, A->_a, A->_ja, A->_ia, C, J, I);
                    ierr = MatSetType(P, MATSEQSBAIJ);CHKERRQ(ierr);
                    ierr = MatSeqSBAIJSetPreallocationCSR(P, 1, I, J, C);CHKERRQ(ierr);
                    delete [] C;
                    delete [] J;
                    delete [] I;
                }
            }
            else
                std::cerr << "Not implemented" << std::endl;
            ierr = KSPSetOperators(_ksp, P, P);CHKERRQ(ierr);
            ierr = MatDestroy(&P);CHKERRQ(ierr);
            ierr = KSPSetType(_ksp, KSPPREONLY);CHKERRQ(ierr);
            ierr = KSPSetOptionsPrefix(_ksp, std::string("petsc_" + std::string(HPDDM_PREFIX) + opt.getPrefix()).c_str());CHKERRQ(ierr);
            ierr = KSPGetPC(_ksp, &pc);CHKERRQ(ierr);
            ierr = PCSetType(pc, A->_sym ? PCCHOLESKY : PCLU);CHKERRQ(ierr);
            ierr = KSPSetFromOptions(_ksp);CHKERRQ(ierr);
            ierr = KSPSetUp(_ksp);CHKERRQ(ierr);
            if(opt.val<char>("verbosity", 0) >= 4) {
                ierr = KSPView(_ksp, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
            }
            PetscFunctionReturn(0);
        }
        PetscErrorCode solve(K* const x, const unsigned short& n = 1) const {
            PetscErrorCode ierr;
            PetscInt m;
            Mat A;
            ierr = KSPGetOperators(_ksp, &A, NULL);CHKERRQ(ierr);
            ierr = MatGetLocalSize(A, &m, NULL);CHKERRQ(ierr);
            K* b = new K[n * m];
            std::copy_n(x, n * m, b);
            ierr = apply(_ksp, 1, b, x, n);CHKERRQ(ierr);
            delete [] b;
            PetscFunctionReturn(0);
        }
        PetscErrorCode solve(const K* const b, K* const x, const unsigned short& n = 1) const {
            PetscErrorCode ierr = apply(_ksp, 1, b, x, n);CHKERRQ(ierr);
            PetscFunctionReturn(0);
        }
};
#endif // PETSCSUB
} // HPDDM
#endif // _HPDDM_PETSC_
