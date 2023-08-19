 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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

#ifndef HPDDM_PETSC_HPP_
#define HPDDM_PETSC_HPP_

#include <petscksp.h>

#include "HPDDM_iterative.hpp"

namespace HPDDM {
class PETScOperator : public EmptyOperator<PetscScalar, PetscInt> {
    public:
        typedef EmptyOperator<PetscScalar, PetscInt> super;
        const KSP ksp_;
    private:
        Vec b_, x_;
        Mat X_[2], C_, Y_;
    public:
        PETScOperator(const PETScOperator&) = delete;
        PETScOperator(const KSP& ksp, PetscInt n) : super(n), ksp_(ksp), b_(), x_(), C_(), Y_() {
            PC pc;
            PetscCallVoid(KSPGetPC(ksp, &pc));
            PetscCallVoid(PCSetFromOptions(pc));
            PetscCallVoid(PCSetUp(pc));
            std::fill_n(X_, 2, nullptr);
        }
        PETScOperator(const KSP& ksp, PetscInt n, PetscInt) : PETScOperator(ksp, n) { }
        ~PETScOperator() {
            PetscCallVoid(MatDestroy(&Y_));
            PetscCallVoid(MatDestroy(&C_));
            PetscCallVoid(MatDestroy(X_));
            PetscCallVoid(MatDestroy(X_ + 1));
            PetscCallVoid(VecDestroy(&x_));
            PetscCallVoid(VecDestroy(&b_));
        }
        template<class K>
        PetscErrorCode GMV(const K* const in, K* const out, const int& mu = 1) const {
            Mat       A, *x = const_cast<Mat*>(X_);
            PetscBool flg;

            PetscFunctionBeginUser;
            PetscCall(KSPGetOperators(ksp_, &A, nullptr));
            PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQKAIJ, MATMPIKAIJ, ""));
            if(flg) {
                PetscBool id;
                PetscCall(MatKAIJGetScaledIdentity(A, &id));
                if(id) {
                    Mat a;
                    const PetscScalar* S, *T;
                    PetscInt bs, n, N;
                    PetscCall(MatGetBlockSize(A, &bs));
                    PetscCall(MatGetLocalSize(A, &n, nullptr));
                    PetscCall(MatGetSize(A, &N, nullptr));
                    const unsigned short eta = mu / bs;
                    PetscCheck(eta * bs == mu, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled case %d != %d", static_cast<int>(eta * bs), static_cast<int>(mu));
                    PetscCall(MatKAIJGetSRead(A, nullptr, nullptr, &S));
                    PetscCall(MatKAIJGetTRead(A, nullptr, nullptr, &T));
                    const PetscBLASInt m = eta * n;
                    PetscScalar* work = new PetscScalar[m]();
                    if(!T)
                        std::copy_n(in, m, work);
                    else if(m)
                        Blas<PetscScalar>::axpy(&m, T, in, &i__1, work, &i__1);
                    PetscInt P = 0, Q = 0;
                    bool reset = false;
                    if(X_[1])
                        PetscCall(MatGetSize(X_[1], &P, &Q));
                    if(P != N / bs || Q != mu) {
                        PetscCall(MatDestroy(x));
                        PetscCall(MatDestroy(x + 1));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp_), n / bs, PETSC_DECIDE, N / bs, mu, work, x + 1));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp_), n / bs, PETSC_DECIDE, N / bs, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : nullptr, x));
                        PetscCall(MatKAIJGetAIJ(A, &a));
                        PetscCall(MatProductCreateWithMat(a, X_[1], nullptr, X_[0]));
                        PetscCall(MatProductSetType(X_[0], MATPRODUCT_AB));
                        PetscCall(MatProductSetFromOptions(X_[0]));
                        PetscCall(MatProductSymbolic(X_[0]));
                    }
                    else {
                        PetscCall(MatDensePlaceArray(X_[1], work));
                        if(std::is_same<PetscScalar, K>::value)
                            PetscCall(MatDensePlaceArray(X_[0], reinterpret_cast<PetscScalar*>(out)));
                        reset = true;
                    }
                    PetscCall(MatProductNumeric(X_[0]));
                    if(m && S)
                        Blas<PetscScalar>::axpy(&m, S, in, &i__1, out, &i__1);
                    delete [] work;
                    PetscCall(MatKAIJRestoreTRead(A, &T));
                    PetscCall(MatKAIJRestoreSRead(A, &S));
                    if(reset) {
                        if(std::is_same<PetscScalar, K>::value)
                            PetscCall(MatDenseResetArray(X_[0]));
                        PetscCall(MatDenseResetArray(X_[1]));
                    }
                    if(!std::is_same<PetscScalar, K>::value) {
                        const PetscScalar* work;
                        PetscCall(MatDenseGetArrayRead(X_[0], &work));
                        HPDDM::copy_n(work, m, out);
                        PetscCall(MatDenseRestoreArrayRead(X_[0], &work));
                    }
                    PetscFunctionReturn(PETSC_SUCCESS);
                }
            }
#if PetscDefined(HAVE_CUDA)
            VecType vtype;
            PetscCall(MatGetVecType(A, &vtype));
            std::initializer_list<std::string> list = { VECCUDA, VECSEQCUDA, VECMPICUDA };
            std::initializer_list<std::string>::const_iterator it = std::find(list.begin(), list.end(), std::string(vtype));
#endif
            if(mu == 1) {
                const PetscScalar *read;
                PetscScalar       *write;
                if(!b_) {
                    PetscInt N;
                    PetscCall(MatGetSize(A, &N, nullptr));
#if PetscDefined(HAVE_CUDA)
                    if(it == list.end()) {
#endif
                        if(std::is_same<PetscScalar, K>::value) {
                            PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ksp_), 1, super::n_, N, nullptr, const_cast<Vec*>(&b_)));
                            PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ksp_), 1, super::n_, N, nullptr, const_cast<Vec*>(&x_)));
                        }
                        else {
                            PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ksp_), super::n_, N, const_cast<Vec*>(&b_)));
                            PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ksp_), super::n_, N, const_cast<Vec*>(&x_)));
                        }
#if PetscDefined(HAVE_CUDA)
                    }
                    else {
                        PetscCall(VecCreateMPICUDA(PetscObjectComm((PetscObject)ksp_), super::n_, N, const_cast<Vec*>(&b_)));
                        PetscCall(VecCreateMPICUDA(PetscObjectComm((PetscObject)ksp_), super::n_, N, const_cast<Vec*>(&x_)));
                    }
#endif
                }
                if(std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                       && it == list.end()
#endif
                                                                          ) {
                    PetscCall(VecPlaceArray(b_, reinterpret_cast<const PetscScalar*>(in)));
                    PetscCall(VecPlaceArray(x_, reinterpret_cast<PetscScalar*>(out)));
                }
                else {
                    PetscCall(VecGetArrayWrite(b_, &write));
                    std::copy_n(in, super::n_, write);
                    PetscCall(VecRestoreArrayWrite(b_, &write));
#if PetscDefined(HAVE_CUDA)
                    if(it != list.end()) {
                        PetscCall(VecCUDAGetArrayRead(b_, &read));
                        PetscCall(VecCUDARestoreArrayRead(b_, &read));
                    }
#endif
                }
#if defined(PETSC_PCHPDDM_MAXLEVELS)
                PetscCall(KSP_MatMult(ksp_, A, b_, x_));
#else
                PetscCall(MatMult(A, b_, x_));
#endif
                if(std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                       && it == list.end()
#endif
                                                                          ) {
                    PetscCall(VecResetArray(x_));
                    PetscCall(VecResetArray(b_));
                }
                else {
                    PetscCall(VecGetArrayRead(x_, &read));
                    HPDDM::copy_n(read, super::n_, out);
                    PetscCall(VecRestoreArrayRead(x_, &read));
                }
            }
            else {
                PC pc;
                Mat *ptr;
                PetscContainer container = nullptr;
                PetscInt M = 0;
                bool reset = false;
                PetscCall(KSPGetPC(ksp_, &pc));
                PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCHPDDM, &flg));
                if(flg) {
                    PetscCall(PetscObjectQuery((PetscObject)A, "_HPDDM_MatProduct", (PetscObject*)&container));
                    if(container) {
                        PetscCall(PetscContainerGetPointer(container, (void**)&ptr));
                        if(ptr[1] != X_[1])
                            for(unsigned short i = 0; i < 2; ++i) {
                                PetscCall(MatDestroy(x + i));
                                x[i] = ptr[i];
                                PetscCall(PetscObjectReference((PetscObject)x[i]));
                            }
                    }
                }
                if(X_[0])
                    PetscCall(MatGetSize(X_[0], nullptr, &M));
                if(M != mu) {
                    PetscInt N;
                    PetscCall(MatGetSize(A, &N, nullptr));
                    PetscCall(MatDestroy(x));
                    PetscCall(MatDestroy(x + 1));
                    if(flg) {
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
                        PCHPDDMCoarseCorrectionType type;
                        PetscCall(PCHPDDMGetCoarseCorrectionType(pc, &type));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, nullptr, x + 1));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, (!std::is_same<PetscScalar, K>::value || type == PC_HPDDM_COARSE_CORRECTION_BALANCED) ? nullptr : reinterpret_cast<PetscScalar*>(out), x));
#endif
                    }
                    else {
#if PetscDefined(HAVE_CUDA)
                        if(it == list.end()) {
#endif
                            PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(const_cast<K*>(in)) : nullptr, x + 1));
                            PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : nullptr, x));
#if PetscDefined(HAVE_CUDA)
                        }
                        else {
                            PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, nullptr, x + 1));
                            PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, nullptr, x));
                        }
#endif
                    }
                    PetscCall(MatProductCreateWithMat(A, X_[1], nullptr, X_[0]));
#if defined(PETSC_PCHPDDM_MAXLEVELS)
                    PetscCall(MatProductSetType(X_[0], !ksp_->transpose_solve ? MATPRODUCT_AB : MATPRODUCT_AtB));
#else
                    PetscCall(MatProductSetType(X_[0], MATPRODUCT_AB));
#endif
                    PetscCall(MatProductSetFromOptions(X_[0]));
                    PetscCall(MatProductSymbolic(X_[0]));
                    if(flg) {
                        reset = true;
                        if(!container) {
                            PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)A), &container));
                            PetscCall(PetscObjectCompose((PetscObject)A, "_HPDDM_MatProduct", (PetscObject)container));
                        }
                        PetscCall(PetscContainerSetPointer(container, x));
                    }
                }
                else {
#if defined(PETSC_PCHPDDM_MAXLEVELS)
                    MatProductType type;
                    bool change = false;
                    PetscCall(MatProductGetType(X_[0], &type));
                    if(type == MATPRODUCT_AB && ksp_->transpose_solve) {
                        change = true;
                        PetscCall(MatProductSetType(X_[0], MATPRODUCT_AtB));
                    }
                    else if(type == MATPRODUCT_AtB && !ksp_->transpose_solve) {
                        change = true;
                        PetscCall(MatProductSetType(X_[0], MATPRODUCT_AB));
                    }
                    if(change) {
                        PetscCall(MatProductSetFromOptions(X_[0]));
                        PetscCall(MatProductSymbolic(X_[0]));
                    }
#endif
                    reset = true;
                    if(container)
                        PetscCall(MatProductReplaceMats(nullptr, X_[1], nullptr, X_[0]));
                }
                if(std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                       && it == list.end()
#endif
                                                                          ) {
                    if (reset) {
                        PetscCall(MatDensePlaceArray(X_[1], reinterpret_cast<PetscScalar*>(const_cast<K*>(in))));
                        PetscCall(MatDensePlaceArray(X_[0], reinterpret_cast<PetscScalar*>(out)));
                    }
                }
                else {
                    PetscScalar* work;
                    PetscCall(MatDenseGetArrayWrite(X_[1], &work));
                    std::copy_n(in, mu * super::n_, work);
                    PetscCall(MatDenseRestoreArrayWrite(X_[1], &work));
#if PetscDefined(HAVE_CUDA)
                    if(it != list.end()) {
                        const PetscScalar* work;
                        PetscCall(MatDenseCUDAGetArrayRead(X_[1], &work));
                        PetscCall(MatDenseCUDARestoreArrayRead(X_[1], &work));
                    }
#endif
                }
                PetscCall(MatProductNumeric(X_[0]));
                if(std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                       && it == list.end()
#endif
                                                                          ) {
                    if(reset) {
                        PetscCall(MatDenseResetArray(X_[0]));
                        PetscCall(MatDenseResetArray(X_[1]));
                    }
                }
                else {
                    const PetscScalar* work;
                    PetscCall(MatDenseGetArrayRead(X_[0], &work));
                    HPDDM::copy_n(work, mu * super::n_, out);
                    PetscCall(MatDenseRestoreArrayRead(X_[0], &work));
                }
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }
#if !defined(PETSC_HAVE_HPDDM) || defined(PETSC_PCHPDDM_MAXLEVELS) || defined(PETSCSUB)
        template<bool = false, class K>
        PetscErrorCode apply(const K* const in, K* const out, const unsigned short& mu = 1, K* = nullptr, const unsigned short& = 0) const {
            PC                pc;
            Mat               A;
            const PetscScalar *read;
            PetscScalar       *write;
            PetscInt          N;
            PetscBool         match;

            PetscFunctionBeginUser;
            PetscCall(KSPGetPC(ksp_, &pc));
            PetscCall(KSPGetOperators(ksp_, &A, nullptr));
            PetscCall(MatGetSize(A, &N, nullptr));
            PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &match, MATSEQKAIJ, MATMPIKAIJ, ""));
            if(match) {
                PetscInt bs, n;
                PetscBool id;
                PetscCall(MatGetLocalSize(A, &n, nullptr));
                PetscCall(MatGetBlockSize(A, &bs));
                PetscCall(MatKAIJGetScaledIdentity(A, &id));
                const unsigned short eta = (id == PETSC_TRUE ? mu / bs : mu);
                PetscCheck(id != PETSC_TRUE || eta * bs == mu, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled case %d != %d", static_cast<int>(eta * bs), static_cast<int>(mu));
                if(!b_) {
                    if(std::is_same<PetscScalar, K>::value) {
                        PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ksp_), 1, n, N, nullptr, const_cast<Vec*>(&b_)));
                        PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ksp_), 1, n, N, nullptr, const_cast<Vec*>(&x_)));
                    }
                    else {
                        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ksp_), n, N, const_cast<Vec*>(&b_)));
                        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ksp_), n, N, const_cast<Vec*>(&x_)));
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    PetscCall(VecGetArrayWrite(b_, &write));
                    PetscCall(VecGetArrayRead(x_, &read));
                }
                for(unsigned short nu = 0; nu < eta; ++nu) {
                    if(id)
                        Wrapper<K>::template imatcopy<'T'>(bs, super::n_, const_cast<K*>(in + nu * n), super::n_, bs);
                    if(std::is_same<PetscScalar, K>::value) {
                        PetscCall(VecPlaceArray(b_, reinterpret_cast<const PetscScalar*>(in + nu * n)));
                        PetscCall(VecPlaceArray(x_, reinterpret_cast<PetscScalar*>(out + nu * n)));
                    }
                    else
                        std::copy_n(in + nu * n, n, write);
#if defined(PETSC_PCHPDDM_MAXLEVELS)
                    PetscCall(KSP_PCApply(ksp_, b_, x_));
#else
                    PetscCall(PCApply(pc, b_, x_));
#endif
                    if(std::is_same<PetscScalar, K>::value) {
                        PetscCall(VecResetArray(x_));
                        PetscCall(VecResetArray(b_));
                    }
                    else
                        HPDDM::copy_n(read, n, out + nu * n);
                    if(id) {
                        Wrapper<K>::template imatcopy<'T'>(super::n_, bs, const_cast<K*>(in + nu * n), bs, super::n_);
                        Wrapper<K>::template imatcopy<'T'>(super::n_, bs, out + nu * n, bs, super::n_);
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    PetscCall(VecRestoreArrayRead(x_, &read));
                    PetscCall(VecRestoreArrayWrite(b_, &write));
                }
                PetscFunctionReturn(PETSC_SUCCESS);
            }
#if PetscDefined(HAVE_CUDA)
            VecType vtype;
            PetscCall(MatGetVecType(A, &vtype));
            std::initializer_list<std::string> list = { VECCUDA, VECSEQCUDA, VECMPICUDA };
            std::initializer_list<std::string>::const_iterator it = std::find(list.begin(), list.end(), std::string(vtype));
#endif
            if(mu > 1) {
                PetscInt M = 0;
                bool reset = false;
                if(Y_)
                    PetscCall(MatGetSize(Y_, nullptr, &M));
                if(M != mu) {
                    PetscCall(MatDestroy(const_cast<Mat*>(&Y_)));
                    PetscCall(MatDestroy(const_cast<Mat*>(&C_)));
#if PetscDefined(HAVE_CUDA)
                    if(it == list.end()) {
#endif
                         PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(const_cast<K*>(in)) : nullptr, const_cast<Mat*>(&C_)));
                         PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : nullptr, const_cast<Mat*>(&Y_)));
#if PetscDefined(HAVE_CUDA)
                    }
                    else {
                        PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, nullptr, const_cast<Mat*>(&C_)));
                        PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)ksp_), super::n_, PETSC_DECIDE, N, mu, nullptr, const_cast<Mat*>(&Y_)));
                    }
#endif
                }
                else if(std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                            && it == list.end()
#endif
                                                                               ) {
                    PetscCall(MatDensePlaceArray(C_, reinterpret_cast<PetscScalar*>(const_cast<K*>(in))));
                    PetscCall(MatDensePlaceArray(Y_, reinterpret_cast<PetscScalar*>(out)));
                    reset = true;
                }
                if(!std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                        || it != list.end()
#endif
                                                                           ) {
                    PetscScalar* work;
                    PetscCall(MatDenseGetArrayWrite(C_, &work));
                    std::copy_n(in, mu * super::n_, work);
                    PetscCall(MatDenseRestoreArrayWrite(C_, &work));
#if PetscDefined(HAVE_CUDA)
                    if(it != list.end()) {
                        PetscCall(MatDenseCUDAGetArrayRead(C_, &read));
                        PetscCall(MatDenseCUDARestoreArrayRead(C_, &read));
                    }
#endif
                }
#if defined(PETSC_PCHPDDM_MAXLEVELS)
                PetscCall(KSP_PCMatApply(ksp_, C_, Y_));
#else
                PetscCall(PCMatApply(pc, C_, Y_));
#endif
                if(reset) {
                    PetscCall(MatDenseResetArray(Y_));
                    PetscCall(MatDenseResetArray(C_));
                }
                else if(!std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                             || it != list.end()
#endif
                                                                                ) {
                    const PetscScalar* work;
                    PetscCall(MatDenseGetArrayRead(Y_, &work));
                    HPDDM::copy_n(work, mu * super::n_, out);
                    PetscCall(MatDenseRestoreArrayRead(Y_, &work));
                }
            }
            else {
                if(!b_) {
#if PetscDefined(HAVE_CUDA)
                    if(it == list.end()) {
#endif
                        if(std::is_same<PetscScalar, K>::value) {
                            PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ksp_), 1, super::n_, N, nullptr, const_cast<Vec*>(&b_)));
                            PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ksp_), 1, super::n_, N, nullptr, const_cast<Vec*>(&x_)));
                        }
                        else {
                            PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ksp_), super::n_, N, const_cast<Vec*>(&b_)));
                            PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ksp_), super::n_, N, const_cast<Vec*>(&x_)));
                        }
#if PetscDefined(HAVE_CUDA)
                    }
                    else {
                        PetscCall(VecCreateMPICUDA(PetscObjectComm((PetscObject)ksp_), super::n_, N, const_cast<Vec*>(&b_)));
                        PetscCall(VecCreateMPICUDA(PetscObjectComm((PetscObject)ksp_), super::n_, N, const_cast<Vec*>(&x_)));
                    }
#endif
                }
                if(std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                       && it == list.end()
#endif
                                                                          ) {
                    PetscCall(VecPlaceArray(b_, reinterpret_cast<const PetscScalar*>(in)));
                    PetscCall(VecPlaceArray(x_, reinterpret_cast<PetscScalar*>(out)));
                }
                else {
                    PetscCall(VecGetArrayWrite(b_, &write));
                    std::copy_n(in, super::n_, write);
                    PetscCall(VecRestoreArrayWrite(b_, &write));
#if PetscDefined(HAVE_CUDA)
                    if(it != list.end()) {
                        PetscCall(VecCUDAGetArrayRead(b_, &read));
                        PetscCall(VecCUDARestoreArrayRead(b_, &read));
                    }
#endif
                }
#if defined(PETSC_PCHPDDM_MAXLEVELS)
                PetscCall(KSP_PCApply(ksp_, b_, x_));
#else
                PetscCall(PCApply(pc, b_, x_));
#endif
                if(std::is_same<PetscScalar, K>::value
#if PetscDefined(HAVE_CUDA)
                                                       && it == list.end()
#endif
                                                                          ) {
                    PetscCall(VecResetArray(x_));
                    PetscCall(VecResetArray(b_));
                }
                else {
                    PetscCall(VecGetArrayRead(x_, &read));
                    HPDDM::copy_n(read, super::n_, out);
                    PetscCall(VecRestoreArrayRead(x_, &read));
                }
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }
#endif
        std::string prefix() const {
            const char* prefix = nullptr;
            if(ksp_)
                PetscCallContinue(KSPGetOptionsPrefix(ksp_, &prefix));
            return prefix ? prefix : "";
        }
};

template<class K>
inline PetscErrorCode convert(MatrixCSR<K>* const& A, Mat& P) {
    PetscFunctionBeginUser;
    PetscCall(MatCreate(PETSC_COMM_SELF, &P));
    PetscCall(MatSetSizes(P, A->n_, A->m_, A->n_, A->m_));
    if(!A->sym_) {
        PetscCall(MatSetType(P, MATSEQAIJ));
        if(std::is_same<int, PetscInt>::value) {
            PetscCall(MatSeqAIJSetPreallocationCSR(P, reinterpret_cast<PetscInt*>(A->ia_), reinterpret_cast<PetscInt*>(A->ja_), A->a_));
        }
        else {
            PetscInt* I = new PetscInt[A->n_ + 1];
            PetscInt* J = new PetscInt[A->nnz_];
            std::copy_n(A->ia_, A->n_ + 1, I);
            std::copy_n(A->ja_, A->nnz_, J);
            PetscCall(MatSeqAIJSetPreallocationCSR(P, I, J, A->a_));
            delete [] J;
            delete [] I;
        }
    }
    else {
        PetscInt* I = new PetscInt[A->n_ + 1];
        PetscInt* J = new PetscInt[A->nnz_];
        PetscScalar* C = new PetscScalar[A->nnz_];
        static_assert(sizeof(int) <= sizeof(PetscInt), "Unsupported PetscInt type");
        Wrapper<K>::template csrcsc<'C', 'C'>(&A->n_, A->a_, A->ja_, A->ia_, C, reinterpret_cast<int*>(J), reinterpret_cast<int*>(I));
        PetscCall(MatSetType(P, MATSEQSBAIJ));
        if(!std::is_same<int, PetscInt>::value) {
            int* ia = reinterpret_cast<int*>(I);
            int* ja = reinterpret_cast<int*>(J);
            for(unsigned int i = A->n_ + 1; i-- > 0; )
                I[i] = ia[i];
            for(unsigned int i = A->nnz_; i-- > 0; )
                J[i] = ja[i];
        }
        PetscCall(MatSeqSBAIJSetPreallocationCSR(P, 1, I, J, C));
        delete [] C;
        delete [] J;
        delete [] I;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSCSUB)
#undef HPDDM_CHECK_COARSEOPERATOR
#define HPDDM_CHECK_SUBDOMAIN
#include "HPDDM_preprocessor_check.hpp"
#define SUBDOMAIN HPDDM::PetscSub
template<class K>
class PetscSub {
    private:
        PETScOperator* op_;
    public:
        PetscSub() : op_() { }
        PetscSub(const PetscSub&) = delete;
        ~PetscSub() { dtor(); }
        static constexpr char numbering_ = 'C';
        PetscErrorCode dtor() {
            PetscFunctionBeginUser;
            if(op_)
                PetscCall(KSPDestroy(const_cast<KSP*>(&op_->ksp_)));
            delete op_;
            op_ = nullptr;
            PetscFunctionReturn(PETSC_SUCCESS);
        }
        template<char N = HPDDM_NUMBERING>
        PetscErrorCode numfact(MatrixCSR<K>* const& A, bool = false, K* const& = nullptr) {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            KSP ksp;
            PC  pc;
            Mat P;
            const Option& opt = *Option::get();
            PetscFunctionBeginUser;
            if(!op_)
                PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
            else
                ksp = op_->ksp_;
            if(N == 'C')
                PetscCall(convert(A, P));
            else {
                P = nullptr;
                std::cerr << "Not implemented" << std::endl;
            }
            std::string prefix("petsc_" + std::string(HPDDM_PREFIX) + opt.getPrefix());
            PetscCall(MatSetOptionsPrefix(P, prefix.c_str()));
            PetscCall(KSPSetOptionsPrefix(ksp, prefix.c_str()));
            PetscCall(KSPSetOperators(ksp, P, P));
            PetscCall(MatDestroy(&P));
            PetscCall(KSPSetType(ksp, KSPPREONLY));
            PetscCall(KSPGetPC(ksp, &pc));
            PetscCall(PCSetType(pc, A->sym_ ? PCCHOLESKY : PCLU));
            PetscCall(KSPSetFromOptions(ksp));
            PetscCall(KSPSetUp(ksp));
            if(opt.val<char>("verbosity", 0) >= 4)
                PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_SELF));
            if(!op_)
                op_ = new PETScOperator(ksp, A->n_);
            PetscFunctionReturn(PETSC_SUCCESS);
        }
        PetscErrorCode solve(K* const x, const unsigned short& n = 1) const {
            PetscFunctionBeginUser;
            if(op_) {
                K* b = new K[n * op_->super::n_];
                PetscCallCXX(std::copy_n(x, n * op_->super::n_, b));
                PetscCall(op_->apply(b, x, n));
                delete [] b;
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }
        PetscErrorCode solve(const K* const b, K* const x, const unsigned short& n = 1) const {
            PetscFunctionBeginUser;
            if(op_)
                PetscCall(op_->apply(b, x, n));
            PetscFunctionReturn(PETSC_SUCCESS);
        }
};
#endif // PETSCSUB
} // HPDDM
#endif // HPDDM_PETSC_HPP_
