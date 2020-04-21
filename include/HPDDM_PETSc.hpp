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

#if defined(_KSPIMPL_H) && PETSC_VERSION_GE(3, 13, 1)
#include <../src/ksp/pc/impls/asm/asm.h>
#endif

namespace HPDDM {
#if PETSC_VERSION_LT(3, 13, 1)
struct PETScOperator
#else
class PETScOperator
#endif
                     : public EmptyOperator<PetscScalar, PetscInt> {
    public:
        typedef EmptyOperator<PetscScalar, PetscInt> super;
        const KSP _ksp;
    private:
        PetscErrorCode (*const _apply)(PC, Mat, Mat);
        Vec _b, _x;
        Mat _B, _X;
    public:
        PETScOperator(const PETScOperator&) = delete;
        PETScOperator(const KSP& ksp, PetscInt n, PetscErrorCode (*apply)(PC, Mat, Mat) = nullptr) : super(n), _ksp(ksp), _apply(apply), _b(), _x(), _B(), _X() {
            PC pc;
            KSPGetPC(ksp, &pc);
            PCSetFromOptions(pc);
            PCSetUp(pc);
        }
        PETScOperator(const KSP& ksp, PetscInt n, PetscInt, PetscErrorCode (*apply)(PC, Mat, Mat) = nullptr) : PETScOperator(ksp, n, apply) { }
        ~PETScOperator() {
            MatDestroy(const_cast<Mat*>(&_X));
            MatDestroy(const_cast<Mat*>(&_B));
            VecDestroy(const_cast<Vec*>(&_x));
            VecDestroy(const_cast<Vec*>(&_b));
        }
        PetscErrorCode GMV(const PetscScalar* const in, PetscScalar* const out, const int& mu = 1) const {
            Mat A;
            PetscErrorCode ierr;
            ierr = KSPGetOperators(_ksp, &A, NULL);CHKERRQ(ierr);
            PetscBool hasMatMatMult;
#if PETSC_VERSION_LT(3, 13, 0)
            ierr = MatHasOperation(A, MATOP_MAT_MULT, &hasMatMatMult);CHKERRQ(ierr);
#else
            ierr = PetscObjectTypeCompareAny((PetscObject)A, &hasMatMatMult, MATSEQAIJ, MATMPIAIJ, "");CHKERRQ(ierr);
#endif
            if(mu == 1 || !hasMatMatMult) {
                if(!_b) {
                    PetscInt n, N;
                    ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
                    ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
                    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, (super::_n / n) * N, NULL, const_cast<Vec*>(&_b));CHKERRQ(ierr);
                    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, (super::_n / n) * N, NULL, const_cast<Vec*>(&_x));CHKERRQ(ierr);
                }
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    ierr = VecPlaceArray(_b, in + nu * super::_n);CHKERRQ(ierr);
                    ierr = VecPlaceArray(_x, out + nu * super::_n);CHKERRQ(ierr);
                    ierr = MatMult(A, _b, _x);CHKERRQ(ierr);
                    ierr = VecResetArray(_x);CHKERRQ(ierr);
                    ierr = VecResetArray(_b);CHKERRQ(ierr);
                }
            }
            else {
                PetscInt M = 0;
                if(_B) {
                    ierr = MatGetSize(_B, NULL, &M);CHKERRQ(ierr);
                }
                if(M != mu) {
                    PetscInt n, N;
                    ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
                    ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
                    ierr = MatDestroy(const_cast<Mat*>(&_X));CHKERRQ(ierr);
                    ierr = MatDestroy(const_cast<Mat*>(&_B));CHKERRQ(ierr);
                    ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, (super::_n / n) * N, mu, const_cast<PetscScalar*>(in), const_cast<Mat*>(&_B));CHKERRQ(ierr);
                    ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, (super::_n / n) * N, mu, out, const_cast<Mat*>(&_X));CHKERRQ(ierr);
                }
                else {
                    ierr = MatDensePlaceArray(_B, const_cast<PetscScalar*>(in));CHKERRQ(ierr);
                    ierr = MatDensePlaceArray(_X, out);CHKERRQ(ierr);
                }
                ierr = MatMatMult(A, _B, MAT_REUSE_MATRIX, PETSC_DEFAULT, const_cast<Mat*>(&_X));CHKERRQ(ierr);
            }
            PetscFunctionReturn(0);
        }
#if !defined(PETSC_HAVE_HPDDM) || defined(_KSPIMPL_H) || PETSC_VERSION_LT(3, 13, 1)
        template<bool = false>
        PetscErrorCode apply(const PetscScalar* const in, PetscScalar* const out, const unsigned short& mu = 1, PetscScalar* = nullptr, const unsigned short& = 0) const {
            PetscErrorCode ierr;
            PC pc;
            KSP* subksp;
            ierr = KSPGetPC(_ksp, &pc);CHKERRQ(ierr);
            PCType type;
            unsigned short distance = 6;
            if(mu > 1) {
                std::initializer_list<std::string> list = { PCBJACOBI, PCASM, PCLU, PCCHOLESKY, PCILU, PCICC };
                ierr = PCGetType(pc, &type);CHKERRQ(ierr);
                std::initializer_list<std::string>::const_iterator it = std::find(list.begin(), list.end(), type);
                distance = std::distance(list.begin(), it);
            }
            Mat F = NULL;
            if(distance == 0 || distance == 1) {
                PetscInt n_local;
                if(distance == 0) {
                    ierr = PCBJacobiGetSubKSP(pc, &n_local, NULL, &subksp);CHKERRQ(ierr);
                }
                else {
#if defined(__ASM_H)
                    PCASMType type;
                    ierr = PCASMGetType(pc, &type);CHKERRQ(ierr);
                    std::initializer_list<PCASMType> list = { PC_ASM_RESTRICT };
                    if(std::find(list.begin(), list.end(), type) != list.end()) {
                        ierr = PCASMGetSubKSP(pc, &n_local, NULL, &subksp);CHKERRQ(ierr);
                    }
                    else
#endif
                        n_local = 0;
                }
                if(n_local == 1) {
                    ierr = KSPSetUp(subksp[0]);CHKERRQ(ierr);
                    PC subpc;
                    ierr = KSPGetPC(subksp[0], &subpc);CHKERRQ(ierr);
                    ierr = PCGetType(subpc, &type);CHKERRQ(ierr);
                    std::initializer_list<std::string> list = { PCLU, PCCHOLESKY, PCILU, PCICC };
                    if(std::find(list.begin(), list.end(), type) != list.end()) {
                        ierr = PCFactorGetMatrix(subpc, &F);CHKERRQ(ierr);
                    }
                }
            }
            else if(distance < 6) {
                ierr = PCFactorGetMatrix(pc, &F);CHKERRQ(ierr);
            }
            if(F || _apply) {
                PetscInt M = 0;
                if(_B) {
                    ierr = MatGetSize(_B, NULL, &M);CHKERRQ(ierr);
                }
                if(M != mu) {
                    Mat A;
                    PetscInt n, N;
                    ierr = KSPGetOperators(_ksp, &A, NULL);CHKERRQ(ierr);
                    ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
                    ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
                    ierr = MatDestroy(const_cast<Mat*>(&_X));CHKERRQ(ierr);
                    ierr = MatDestroy(const_cast<Mat*>(&_B));CHKERRQ(ierr);
                    ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, (super::_n / n) * N, mu, const_cast<PetscScalar*>(in), const_cast<Mat*>(&_B));CHKERRQ(ierr);
                    ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, (super::_n / n) * N, mu, out, const_cast<Mat*>(&_X));CHKERRQ(ierr);
                }
                else {
                    ierr = MatDensePlaceArray(_B, const_cast<PetscScalar*>(in));CHKERRQ(ierr);
                    ierr = MatDensePlaceArray(_X, out);CHKERRQ(ierr);
                }
                if(distance == 1) {
#if defined(__ASM_H)
                    PC_ASM *osm = (PC_ASM*)pc->data;
                    Mat A;
                    PetscInt n, N, o;
                    ierr = KSPGetOperators(_ksp, &A, NULL);CHKERRQ(ierr);
                    ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
                    ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
                    ierr = VecGetLocalSize(osm->x[0], &o);CHKERRQ(ierr);
                    Vec x;
                    Mat X, Y;
                    PetscScalar* array;
                    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, (super::_n / n) * N, NULL, &x);CHKERRQ(ierr);
                    ierr = MatCreateSeqDense(PETSC_COMM_SELF, o, mu, NULL, &X);CHKERRQ(ierr);
                    ierr = MatDenseGetArray(X, &array);CHKERRQ(ierr);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        ierr = VecPlaceArray(x, in + nu * super::_n);CHKERRQ(ierr);
                        ierr = VecScatterBegin(osm->restriction, x, osm->lx, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
                        ierr = VecScatterEnd(osm->restriction, x, osm->lx, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
                        ierr = VecResetArray(x);CHKERRQ(ierr);
                        ierr = VecPlaceArray(osm->x[0], array + nu * o);CHKERRQ(ierr);
                        ierr = VecScatterBegin(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
                        ierr = VecScatterEnd(osm->lrestriction[0], osm->lx, osm->x[0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
                        ierr = VecResetArray(osm->x[0]);CHKERRQ(ierr);
                    }
                    ierr = MatDenseRestoreArray(X, &array);CHKERRQ(ierr);
                    ierr = MatCreateSeqDense(PETSC_COMM_SELF, o, mu, NULL, &Y);CHKERRQ(ierr);
                    ierr = MatMatSolve(F, X, Y);CHKERRQ(ierr);
                    ierr = MatDestroy(&X);CHKERRQ(ierr);
                    ierr = MatDenseGetArray(Y, &array);CHKERRQ(ierr);
                    std::fill_n(out, mu * super::_n, PetscScalar());
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        ierr = VecSet(osm->ly, 0.0);CHKERRQ(ierr);
                        ierr = VecPlaceArray(osm->y[0], array + nu * o);CHKERRQ(ierr);
                        if(osm->lprolongation) {
                            ierr = VecScatterBegin(osm->lprolongation[0], osm->y[0], osm->ly, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
                            ierr = VecScatterEnd(osm->lprolongation[0], osm->y[0], osm->ly, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
                        }
                        else {
                            ierr = VecScatterBegin(osm->lrestriction[0], osm->y[0], osm->ly, ADD_VALUES, SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
                            ierr = VecScatterEnd(osm->lrestriction[0], osm->y[0], osm->ly, ADD_VALUES, SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
                        }
                        ierr = VecResetArray(osm->y[0]);CHKERRQ(ierr);
                        ierr = VecPlaceArray(x, out + nu * super::_n);CHKERRQ(ierr);
                        ierr = VecScatterBegin(osm->restriction, osm->ly, x, ADD_VALUES, SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
                        ierr = VecScatterEnd(osm->restriction, osm->ly, x, ADD_VALUES, SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
                        ierr = VecResetArray(x);CHKERRQ(ierr);
                    }
                    ierr = MatDenseRestoreArray(Y, &array);CHKERRQ(ierr);
                    ierr = MatDestroy(&Y);CHKERRQ(ierr);
                    ierr = VecDestroy(&x);CHKERRQ(ierr);
#endif
                }
                else if(distance == 0) {
                    Mat B, X;
                    ierr = MatDenseGetLocalMatrix(_B, &B);CHKERRQ(ierr);
                    ierr = MatDenseGetLocalMatrix(_X, &X);CHKERRQ(ierr);
                    ierr = MatMatSolve(F, B, X);CHKERRQ(ierr);
                }
                else {
                    ierr = _apply ? _apply(pc, _B, _X) : MatMatSolve(F, _B, _X);CHKERRQ(ierr);
                }
            }
            else {
                if(!_b) {
                    Mat A;
                    PetscInt n, N;
                    ierr = KSPGetOperators(_ksp, &A, NULL);CHKERRQ(ierr);
                    ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
                    ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
                    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, (super::_n / n) * N, NULL, const_cast<Vec*>(&_b));CHKERRQ(ierr);
                    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, (super::_n / n) * N, NULL, const_cast<Vec*>(&_x));CHKERRQ(ierr);
                }
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    ierr = VecPlaceArray(_b, in + nu * super::_n);CHKERRQ(ierr);
                    ierr = VecPlaceArray(_x, out + nu * super::_n);CHKERRQ(ierr);
                    ierr = PCApply(pc, _b, _x);CHKERRQ(ierr);
                    ierr = VecResetArray(_x);CHKERRQ(ierr);
                    ierr = VecResetArray(_b);CHKERRQ(ierr);
                }
            }
            PetscFunctionReturn(0);
        }
#endif
        std::string prefix() const {
            const char* prefix = nullptr;
            if(_ksp)
                KSPGetOptionsPrefix(_ksp, &prefix);
            return prefix ? prefix : "";
        }
};

#if defined(PETSCSUB)
#undef HPDDM_CHECK_COARSEOPERATOR
#define HPDDM_CHECK_SUBDOMAIN
#include "HPDDM_preprocessor_check.hpp"
#define SUBDOMAIN HPDDM::PetscSub
template<class K>
class PetscSub {
    private:
        PETScOperator* _op;
    public:
        PetscSub() : _op() { }
        PetscSub(const PetscSub&) = delete;
        ~PetscSub() { dtor(); }
        static constexpr char _numbering = 'C';
        PetscErrorCode dtor() {
            if(_op) {
                PetscErrorCode ierr = KSPDestroy(const_cast<KSP*>(&_op->_ksp));CHKERRQ(ierr);
            }
            delete _op;
            _op = nullptr;
            PetscFunctionReturn(0);
        }
        template<char N = HPDDM_NUMBERING>
        PetscErrorCode numfact(MatrixCSR<K>* const& A, bool detection = false, K* const& schur = nullptr) {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            Mat P;
            KSP ksp;
            PC pc;
            PetscErrorCode ierr;
            const Option& opt = *Option::get();
            if(!_op) {
                ierr = KSPCreate(PETSC_COMM_SELF, &ksp);CHKERRQ(ierr);
            }
            else
                ksp = _op->_ksp;
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
            ierr = KSPSetOperators(ksp, P, P);CHKERRQ(ierr);
            ierr = MatDestroy(&P);CHKERRQ(ierr);
            ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
            ierr = KSPSetOptionsPrefix(ksp, std::string("petsc_" + std::string(HPDDM_PREFIX) + opt.getPrefix()).c_str());CHKERRQ(ierr);
            ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
            ierr = PCSetType(pc, A->_sym ? PCCHOLESKY : PCLU);CHKERRQ(ierr);
            ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
            ierr = KSPSetUp(ksp);CHKERRQ(ierr);
            if(opt.val<char>("verbosity", 0) >= 4) {
                ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
            }
            if(!_op) {
                _op = new PETScOperator(ksp, A->_n);
            }
            PetscFunctionReturn(0);
        }
        PetscErrorCode solve(K* const x, const unsigned short& n = 1) const {
            if(_op) {
                K* b = new K[n * _op->super::_n];
                std::copy_n(x, n * _op->super::_n, b);
                PetscErrorCode ierr = _op->apply(b, x, n);CHKERRQ(ierr);
                delete [] b;
            }
            PetscFunctionReturn(0);
        }
        PetscErrorCode solve(const K* const b, K* const x, const unsigned short& n = 1) const {
            if(_op) {
                PetscErrorCode ierr = _op->apply(b, x, n);CHKERRQ(ierr);
            }
            PetscFunctionReturn(0);
        }
};
#endif // PETSCSUB
} // HPDDM
#endif // _HPDDM_PETSC_
