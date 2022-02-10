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
class PETScOperator : public EmptyOperator<PetscScalar, PetscInt> {
    public:
        typedef EmptyOperator<PetscScalar, PetscInt> super;
        const KSP _ksp;
    private:
        Vec _b, _x;
        Mat _X[2], _C, _Y;
    public:
        PETScOperator(const PETScOperator&) = delete;
        PETScOperator(const KSP& ksp, PetscInt n) : super(n), _ksp(ksp), _b(), _x(), _C(), _Y() {
            PC pc;
            KSPGetPC(ksp, &pc);
            PCSetFromOptions(pc);
            PCSetUp(pc);
            std::fill_n(_X, 2, nullptr);
        }
        PETScOperator(const KSP& ksp, PetscInt n, PetscInt) : PETScOperator(ksp, n) { }
        ~PETScOperator() {
            MatDestroy(&_Y);
            MatDestroy(&_C);
            MatDestroy(_X);
            MatDestroy(_X + 1);
            VecDestroy(&_x);
            VecDestroy(&_b);
        }
        template<class K>
        PetscErrorCode GMV(const K* const in, K* const out, const int& mu = 1) const {
            Mat            A, *x = const_cast<Mat*>(_X);
            PetscBool      flg;
            PetscErrorCode ierr;

            PetscFunctionBeginUser;
            ierr = KSPGetOperators(_ksp, &A, NULL);CHKERRQ(ierr);
            ierr = PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQKAIJ, MATMPIKAIJ, "");CHKERRQ(ierr);
            if(flg) {
                PetscBool id;
                ierr = MatKAIJGetScaledIdentity(A, &id);CHKERRQ(ierr);
                if(id) {
                    Mat a;
                    const PetscScalar* S, *T;
                    PetscInt bs, n, N;
                    ierr = MatGetBlockSize(A, &bs);CHKERRQ(ierr);
                    ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
                    ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
                    const unsigned short eta = mu / bs;
                    PetscCheck(eta * bs == mu, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled case %d != %d", static_cast<int>(eta * bs), static_cast<int>(mu)); // LCOV_EXCL_LINE
                    ierr = MatKAIJGetSRead(A, nullptr, nullptr, &S);CHKERRQ(ierr);
                    ierr = MatKAIJGetTRead(A, nullptr, nullptr, &T);CHKERRQ(ierr);
                    const PetscBLASInt m = eta * n;
                    PetscScalar* work = new PetscScalar[m]();
                    if(!T)
                        std::copy_n(in, m, work);
                    else if(m)
                        Blas<PetscScalar>::axpy(&m, T, in, &i__1, work, &i__1);
                    PetscInt P = 0, Q = 0;
                    bool reset = false;
                    if(_X[1]) {
                        ierr = MatGetSize(_X[1], &P, &Q);CHKERRQ(ierr);
                    }
                    if(P != N / bs || Q != mu) {
                        ierr = MatDestroy(x);CHKERRQ(ierr);
                        ierr = MatDestroy(x + 1);CHKERRQ(ierr);
                        ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), n / bs, PETSC_DECIDE, N / bs, mu, work, x + 1);CHKERRQ(ierr);
                        ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), n / bs, PETSC_DECIDE, N / bs, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : NULL, x);CHKERRQ(ierr);
                        if(!std::is_same<PetscScalar, K>::value) {
                            ierr = MatAssemblyBegin(_X[0], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                            ierr = MatAssemblyEnd(_X[0], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                        }
                        ierr = MatKAIJGetAIJ(A, &a);CHKERRQ(ierr);
                        ierr = MatProductCreateWithMat(a, _X[1], NULL, _X[0]);CHKERRQ(ierr);
                        ierr = MatProductSetType(_X[0], MATPRODUCT_AB);CHKERRQ(ierr);
                        ierr = MatProductSetFromOptions(_X[0]);CHKERRQ(ierr);
                        ierr = MatProductSymbolic(_X[0]);CHKERRQ(ierr);
                    }
                    else {
                        ierr = MatDensePlaceArray(_X[1], work);CHKERRQ(ierr);
                        if(std::is_same<PetscScalar, K>::value) {
                            ierr = MatDensePlaceArray(_X[0], reinterpret_cast<PetscScalar*>(out));CHKERRQ(ierr);
                        }
                        reset = true;
                    }
                    ierr = MatProductNumeric(_X[0]);CHKERRQ(ierr);
                    if(m && S)
                        Blas<PetscScalar>::axpy(&m, S, in, &i__1, out, &i__1);
                    delete [] work;
                    ierr = MatKAIJRestoreTRead(A, &T);CHKERRQ(ierr);
                    ierr = MatKAIJRestoreSRead(A, &S);CHKERRQ(ierr);
                    if(reset) {
                        if(std::is_same<PetscScalar, K>::value) {
                            ierr = MatDenseResetArray(_X[0]);CHKERRQ(ierr);
                        }
                        ierr = MatDenseResetArray(_X[1]);CHKERRQ(ierr);
                    }
                    if(!std::is_same<PetscScalar, K>::value) {
                        const PetscScalar* work;
                        ierr = MatDenseGetArrayRead(_X[0], &work);CHKERRQ(ierr);
                        std::copy_n(work, m, out);
                        ierr = MatDenseRestoreArrayRead(_X[0], &work);CHKERRQ(ierr);
                    }
                    PetscFunctionReturn(0);
                }
            }
            if(mu == 1) {
                const PetscScalar *read;
                PetscScalar       *write;
                if(!_b) {
                    PetscInt N;
                    ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
                    if(std::is_same<PetscScalar, K>::value) {
                        ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, N, NULL, const_cast<Vec*>(&_b));CHKERRQ(ierr);
                        ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, N, NULL, const_cast<Vec*>(&_x));CHKERRQ(ierr);
                    }
                    else {
                        ierr = VecCreateMPI(PetscObjectComm((PetscObject)_ksp), super::_n, N, const_cast<Vec*>(&_b));CHKERRQ(ierr);
                        ierr = VecCreateMPI(PetscObjectComm((PetscObject)_ksp), super::_n, N, const_cast<Vec*>(&_x));CHKERRQ(ierr);
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    ierr = VecGetArrayWrite(_b, &write);CHKERRQ(ierr);
                    std::copy_n(in, super::_n, write);
                    ierr = VecGetArrayRead(_x, &read);CHKERRQ(ierr);
                }
                else {
                    ierr = VecPlaceArray(_b, reinterpret_cast<const PetscScalar*>(in));CHKERRQ(ierr);
                    ierr = VecPlaceArray(_x, reinterpret_cast<PetscScalar*>(out));CHKERRQ(ierr);
                }
                ierr = MatMult(A, _b, _x);CHKERRQ(ierr);
                if(std::is_same<PetscScalar, K>::value) {
                    ierr = VecResetArray(_x);CHKERRQ(ierr);
                    ierr = VecResetArray(_b);CHKERRQ(ierr);
                }
                else {
                    std::copy_n(read, super::_n, out);
                    ierr = VecRestoreArrayRead(_x, &read);CHKERRQ(ierr);
                    ierr = VecRestoreArrayWrite(_b, &write);CHKERRQ(ierr);
                }
            }
            else {
                PC pc;
                Mat *ptr;
                PetscContainer container = NULL;
                PetscInt M = 0;
                bool reset = false;
                ierr = KSPGetPC(_ksp, &pc);CHKERRQ(ierr);
                ierr = PetscObjectTypeCompare((PetscObject)pc, PCHPDDM, &flg);CHKERRQ(ierr);
                if(flg) {
                    ierr = PetscObjectQuery((PetscObject)A, "_HPDDM_MatProduct", (PetscObject*)&container);CHKERRQ(ierr);
                    if(container) {
                        ierr = PetscContainerGetPointer(container, (void**)&ptr);CHKERRQ(ierr);
                        if(ptr[1] != _X[1])
                            for(unsigned short i = 0; i < 2; ++i) {
                                ierr = MatDestroy(x + i);CHKERRQ(ierr);
                                x[i] = ptr[i];
                                ierr = PetscObjectReference((PetscObject)x[i]);CHKERRQ(ierr);
                            }
                    }
                }
                if(_X[0]) {
                    ierr = MatGetSize(_X[0], NULL, &M);CHKERRQ(ierr);
                }
                if(M != mu) {
                    PetscInt N;
                    ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
                    ierr = MatDestroy(x);CHKERRQ(ierr);
                    ierr = MatDestroy(x + 1);CHKERRQ(ierr);
                    if(flg) {
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
                        PCHPDDMCoarseCorrectionType type;
                        ierr = PCHPDDMGetCoarseCorrectionType(pc, &type);CHKERRQ(ierr);
                        ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, NULL, x + 1);CHKERRQ(ierr);
                        ierr = MatAssemblyBegin(_X[1], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                        ierr = MatAssemblyEnd(_X[1], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                        if(type == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
                            ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, NULL, x);CHKERRQ(ierr);
                            ierr = MatAssemblyBegin(_X[0], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                            ierr = MatAssemblyEnd(_X[0], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                        }
                        else {
                            ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : NULL, x);CHKERRQ(ierr);
                        }
#endif
                    }
                    else {
                        ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(const_cast<K*>(in)) : NULL, x + 1);CHKERRQ(ierr);
                        ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : NULL, x);CHKERRQ(ierr);
                        if(!std::is_same<PetscScalar, K>::value) {
                            ierr = MatAssemblyBegin(_X[1], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                            ierr = MatAssemblyEnd(_X[1], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                            ierr = MatAssemblyBegin(_X[0], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                            ierr = MatAssemblyEnd(_X[0], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                        }
                    }
                    ierr = MatProductCreateWithMat(A, _X[1], NULL, _X[0]);CHKERRQ(ierr);
                    ierr = MatProductSetType(_X[0], MATPRODUCT_AB);CHKERRQ(ierr);
                    ierr = MatProductSetFromOptions(_X[0]);CHKERRQ(ierr);
                    ierr = MatProductSymbolic(_X[0]);CHKERRQ(ierr);
                    if(flg) {
                        reset = true;
                        if(!container) {
                            ierr = PetscContainerCreate(PetscObjectComm((PetscObject)A), &container);CHKERRQ(ierr);
                            ierr = PetscObjectCompose((PetscObject)A, "_HPDDM_MatProduct", (PetscObject)container);CHKERRQ(ierr);
                        }
                        ierr = PetscContainerSetPointer(container, x);CHKERRQ(ierr);
                    }
                }
                else {
                    reset = true;
                    if(container) {
                        ierr = MatProductReplaceMats(NULL, _X[1], NULL, _X[0]);CHKERRQ(ierr);
                    }
                }
                if(std::is_same<PetscScalar, K>::value) {
                    if (reset) {
                        ierr = MatDensePlaceArray(_X[1], reinterpret_cast<PetscScalar*>(const_cast<K*>(in)));CHKERRQ(ierr);
                        ierr = MatDensePlaceArray(_X[0], reinterpret_cast<PetscScalar*>(out));CHKERRQ(ierr);
                    }
                }
                else {
                    PetscScalar* work;
                    ierr = MatDenseGetArrayWrite(_X[1], &work);CHKERRQ(ierr);
                    std::copy_n(in, mu * super::_n, work);
                    ierr = MatDenseRestoreArrayWrite(_X[1], &work);CHKERRQ(ierr);
                }
                ierr = MatProductNumeric(_X[0]);CHKERRQ(ierr);
                if(std::is_same<PetscScalar, K>::value) {
                    if(reset) {
                        ierr = MatDenseResetArray(_X[0]);CHKERRQ(ierr);
                        ierr = MatDenseResetArray(_X[1]);CHKERRQ(ierr);
                    }
                }
                else {
                    const PetscScalar* work;
                    ierr = MatDenseGetArrayRead(_X[0], &work);CHKERRQ(ierr);
                    std::copy_n(work, mu * super::_n, out);
                    ierr = MatDenseRestoreArrayRead(_X[0], &work);CHKERRQ(ierr);
                }
            }
            PetscFunctionReturn(0);
        }
#if !defined(PETSC_HAVE_HPDDM) || defined(_KSPIMPL_H) || defined(PETSCSUB)
        template<bool = false, class K>
        PetscErrorCode apply(const K* const in, K* const out, const unsigned short& mu = 1, K* = nullptr, const unsigned short& = 0) const {
            PC                pc;
            Mat               A;
            const PetscScalar *read;
            PetscScalar       *write;
            PetscInt          N;
            PetscBool         match;
            PetscErrorCode    ierr;

            PetscFunctionBeginUser;
            ierr = KSPGetPC(_ksp, &pc);CHKERRQ(ierr);
            ierr = KSPGetOperators(_ksp, &A, NULL);CHKERRQ(ierr);
            ierr = MatGetSize(A, &N, NULL);CHKERRQ(ierr);
            ierr = PetscObjectTypeCompareAny((PetscObject)A, &match, MATSEQKAIJ, MATMPIKAIJ, "");CHKERRQ(ierr);
            if(match) {
                PetscInt bs, n;
                PetscBool id;
                ierr = MatGetLocalSize(A, &n, NULL);CHKERRQ(ierr);
                ierr = MatGetBlockSize(A, &bs);CHKERRQ(ierr);
                ierr = MatKAIJGetScaledIdentity(A, &id);CHKERRQ(ierr);
                const unsigned short eta = (id == PETSC_TRUE ? mu / bs : mu);
                PetscCheck(id != PETSC_TRUE || eta * bs == mu, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled case %d != %d", static_cast<int>(eta * bs), static_cast<int>(mu)); // LCOV_EXCL_LINE
                if(!_b) {
                    if(std::is_same<PetscScalar, K>::value) {
                        ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, n, N, NULL, const_cast<Vec*>(&_b));CHKERRQ(ierr);
                        ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, n, N, NULL, const_cast<Vec*>(&_x));CHKERRQ(ierr);
                    }
                    else {
                        ierr = VecCreateMPI(PetscObjectComm((PetscObject)_ksp), n, N, const_cast<Vec*>(&_b));CHKERRQ(ierr);
                        ierr = VecCreateMPI(PetscObjectComm((PetscObject)_ksp), n, N, const_cast<Vec*>(&_x));CHKERRQ(ierr);
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    ierr = VecGetArrayWrite(_b, &write);CHKERRQ(ierr);
                    ierr = VecGetArrayRead(_x, &read);CHKERRQ(ierr);
                }
                for(unsigned short nu = 0; nu < eta; ++nu) {
                    if(id)
                        Wrapper<K>::template imatcopy<'T'>(bs, super::_n, const_cast<K*>(in + nu * n), super::_n, bs);
                    if(std::is_same<PetscScalar, K>::value) {
                        ierr = VecPlaceArray(_b, reinterpret_cast<const PetscScalar*>(in + nu * n));CHKERRQ(ierr);
                        ierr = VecPlaceArray(_x, reinterpret_cast<PetscScalar*>(out + nu * n));CHKERRQ(ierr);
                    }
                    else
                        std::copy_n(in + nu * n, n, write);
                    ierr = PCApply(pc, _b, _x);CHKERRQ(ierr);
                    if(std::is_same<PetscScalar, K>::value) {
                        ierr = VecResetArray(_x);CHKERRQ(ierr);
                        ierr = VecResetArray(_b);CHKERRQ(ierr);
                    }
                    else
                        std::copy_n(read, n, out + nu * n);
                    if(id) {
                        Wrapper<K>::template imatcopy<'T'>(super::_n, bs, const_cast<K*>(in + nu * n), bs, super::_n);
                        Wrapper<K>::template imatcopy<'T'>(super::_n, bs, out + nu * n, bs, super::_n);
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    ierr = VecRestoreArrayRead(_x, &read);CHKERRQ(ierr);
                    ierr = VecRestoreArrayWrite(_b, &write);CHKERRQ(ierr);
                }
                PetscFunctionReturn(0);
            }
            if(mu > 1) {
                PetscInt M = 0;
                bool reset = false;
                if(_Y) {
                    ierr = MatGetSize(_Y, NULL, &M);CHKERRQ(ierr);
                }
                if(M != mu) {
                    ierr = MatDestroy(const_cast<Mat*>(&_Y));CHKERRQ(ierr);
                    ierr = MatDestroy(const_cast<Mat*>(&_C));CHKERRQ(ierr);
                    ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(const_cast<K*>(in)) : NULL, const_cast<Mat*>(&_C));CHKERRQ(ierr);
                    ierr = MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : NULL, const_cast<Mat*>(&_Y));CHKERRQ(ierr);
                    if(!std::is_same<PetscScalar, K>::value) {
                        ierr = MatAssemblyBegin(_C, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                        ierr = MatAssemblyEnd(_C, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                    }
                }
                else if(std::is_same<PetscScalar, K>::value) {
                    ierr = MatDensePlaceArray(_C, reinterpret_cast<PetscScalar*>(const_cast<K*>(in)));CHKERRQ(ierr);
                    ierr = MatDensePlaceArray(_Y, reinterpret_cast<PetscScalar*>(out));CHKERRQ(ierr);
                    reset = true;
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    PetscScalar* work;
                    ierr = MatDenseGetArrayWrite(_C, &work);CHKERRQ(ierr);
                    std::copy_n(in, mu * super::_n, work);
                    ierr = MatDenseRestoreArrayWrite(_C, &work);CHKERRQ(ierr);
                }
                ierr = PCMatApply(pc, _C, _Y);CHKERRQ(ierr);
                if(reset) {
                    ierr = MatDenseResetArray(_Y);CHKERRQ(ierr);
                    ierr = MatDenseResetArray(_C);CHKERRQ(ierr);
                }
                else if(!std::is_same<PetscScalar, K>::value) {
                    const PetscScalar* work;
                    ierr = MatDenseGetArrayRead(_Y, &work);CHKERRQ(ierr);
                    std::copy_n(work, mu * super::_n, out);
                    ierr = MatDenseRestoreArrayRead(_Y, &work);CHKERRQ(ierr);
                }
            }
            else {
                if(!_b) {
                    if(std::is_same<PetscScalar, K>::value) {
                        ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, N, NULL, const_cast<Vec*>(&_b));CHKERRQ(ierr);
                        ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, N, NULL, const_cast<Vec*>(&_x));CHKERRQ(ierr);
                    }
                    else {
                        ierr = VecCreateMPI(PetscObjectComm((PetscObject)_ksp), super::_n, N, const_cast<Vec*>(&_b));CHKERRQ(ierr);
                        ierr = VecCreateMPI(PetscObjectComm((PetscObject)_ksp), super::_n, N, const_cast<Vec*>(&_x));CHKERRQ(ierr);
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    ierr = VecGetArrayWrite(_b, &write);CHKERRQ(ierr);
                    std::copy_n(in, super::_n, write);
                    ierr = VecGetArrayRead(_x, &read);CHKERRQ(ierr);
                }
                else {
                    ierr = VecPlaceArray(_b, reinterpret_cast<const PetscScalar*>(in));CHKERRQ(ierr);
                    ierr = VecPlaceArray(_x, reinterpret_cast<PetscScalar*>(out));CHKERRQ(ierr);
                }
                ierr = PCApply(pc, _b, _x);CHKERRQ(ierr);
                if(std::is_same<PetscScalar, K>::value) {
                    ierr = VecResetArray(_x);CHKERRQ(ierr);
                    ierr = VecResetArray(_b);CHKERRQ(ierr);
                }
                else {
                    std::copy_n(read, super::_n, out);
                    ierr = VecRestoreArrayRead(_x, &read);CHKERRQ(ierr);
                    ierr = VecRestoreArrayWrite(_b, &write);CHKERRQ(ierr);
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

template<class K>
inline PetscErrorCode convert(MatrixCSR<K>* const& A, Mat& P) {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = MatCreate(PETSC_COMM_SELF, &P);CHKERRQ(ierr);
    ierr = MatSetSizes(P, A->_n, A->_m, A->_n, A->_m);CHKERRQ(ierr);
    if(!A->_sym) {
        ierr = MatSetType(P, MATSEQAIJ);CHKERRQ(ierr);
        if(std::is_same<int, PetscInt>::value) {
            ierr = MatSeqAIJSetPreallocationCSR(P, reinterpret_cast<PetscInt*>(A->_ia), reinterpret_cast<PetscInt*>(A->_ja), A->_a);CHKERRQ(ierr);
        }
        else {
            PetscInt* I = new PetscInt[A->_n + 1];
            PetscInt* J = new PetscInt[A->_nnz];
            std::copy_n(A->_ia, A->_n + 1, I);
            std::copy_n(A->_ja, A->_nnz, J);
            ierr = MatSeqAIJSetPreallocationCSR(P, I, J, A->_a);CHKERRQ(ierr);
            delete [] J;
            delete [] I;
        }
    }
    else {
        PetscInt* I = new PetscInt[A->_n + 1];
        PetscInt* J = new PetscInt[A->_nnz];
        PetscScalar* C = new PetscScalar[A->_nnz];
        static_assert(sizeof(int) <= sizeof(PetscInt), "Unsupported PetscInt type");
        Wrapper<K>::template csrcsc<'C', 'C'>(&A->_n, A->_a, A->_ja, A->_ia, C, reinterpret_cast<int*>(J), reinterpret_cast<int*>(I));
        ierr = MatSetType(P, MATSEQSBAIJ);CHKERRQ(ierr);
        if(!std::is_same<int, PetscInt>::value) {
            int* ia = reinterpret_cast<int*>(I);
            int* ja = reinterpret_cast<int*>(J);
            for(unsigned int i = A->_n + 1; i-- > 0; )
                I[i] = ia[i];
            for(unsigned int i = A->_nnz; i-- > 0; )
                J[i] = ja[i];
        }
        ierr = MatSeqSBAIJSetPreallocationCSR(P, 1, I, J, C);CHKERRQ(ierr);
        delete [] C;
        delete [] J;
        delete [] I;
    }
    PetscFunctionReturn(0);
}

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
        PetscErrorCode numfact(MatrixCSR<K>* const& A, bool = false, K* const& = nullptr) {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            KSP ksp;
            PC  pc;
            Mat P;
            PetscErrorCode ierr;
            const Option& opt = *Option::get();
            PetscFunctionBeginUser;
            if(!_op) {
                ierr = KSPCreate(PETSC_COMM_SELF, &ksp);CHKERRQ(ierr);
            }
            else
                ksp = _op->_ksp;
            if(N == 'C') {
                ierr = convert(A, P);CHKERRQ(ierr);
            }
            else {
                P = nullptr;
                std::cerr << "Not implemented" << std::endl;
            }
            std::string prefix("petsc_" + std::string(HPDDM_PREFIX) + opt.getPrefix());
            ierr = MatSetOptionsPrefix(P, prefix.c_str());CHKERRQ(ierr);
            ierr = KSPSetOptionsPrefix(ksp, prefix.c_str());CHKERRQ(ierr);
            ierr = KSPSetOperators(ksp, P, P);CHKERRQ(ierr);
            ierr = MatDestroy(&P);CHKERRQ(ierr);
            ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
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
