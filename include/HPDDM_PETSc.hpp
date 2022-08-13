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
            PetscCallVoid(KSPGetPC(ksp, &pc));
            PetscCallVoid(PCSetFromOptions(pc));
            PetscCallVoid(PCSetUp(pc));
            std::fill_n(_X, 2, nullptr);
        }
        PETScOperator(const KSP& ksp, PetscInt n, PetscInt) : PETScOperator(ksp, n) { }
        ~PETScOperator() {
            PetscCallVoid(MatDestroy(&_Y));
            PetscCallVoid(MatDestroy(&_C));
            PetscCallVoid(MatDestroy(_X));
            PetscCallVoid(MatDestroy(_X + 1));
            PetscCallVoid(VecDestroy(&_x));
            PetscCallVoid(VecDestroy(&_b));
        }
        template<class K>
        PetscErrorCode GMV(const K* const in, K* const out, const int& mu = 1) const {
            Mat       A, *x = const_cast<Mat*>(_X);
            PetscBool flg;

            PetscFunctionBeginUser;
            PetscCall(KSPGetOperators(_ksp, &A, NULL));
            PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQKAIJ, MATMPIKAIJ, ""));
            if(flg) {
                PetscBool id;
                PetscCall(MatKAIJGetScaledIdentity(A, &id));
                if(id) {
                    Mat a;
                    const PetscScalar* S, *T;
                    PetscInt bs, n, N;
                    PetscCall(MatGetBlockSize(A, &bs));
                    PetscCall(MatGetLocalSize(A, &n, NULL));
                    PetscCall(MatGetSize(A, &N, NULL));
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
                    if(_X[1])
                        PetscCall(MatGetSize(_X[1], &P, &Q));
                    if(P != N / bs || Q != mu) {
                        PetscCall(MatDestroy(x));
                        PetscCall(MatDestroy(x + 1));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)_ksp), n / bs, PETSC_DECIDE, N / bs, mu, work, x + 1));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)_ksp), n / bs, PETSC_DECIDE, N / bs, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : NULL, x));
                        PetscCall(MatKAIJGetAIJ(A, &a));
                        PetscCall(MatProductCreateWithMat(a, _X[1], NULL, _X[0]));
                        PetscCall(MatProductSetType(_X[0], MATPRODUCT_AB));
                        PetscCall(MatProductSetFromOptions(_X[0]));
                        PetscCall(MatProductSymbolic(_X[0]));
                    }
                    else {
                        PetscCall(MatDensePlaceArray(_X[1], work));
                        if(std::is_same<PetscScalar, K>::value)
                            PetscCall(MatDensePlaceArray(_X[0], reinterpret_cast<PetscScalar*>(out)));
                        reset = true;
                    }
                    PetscCall(MatProductNumeric(_X[0]));
                    if(m && S)
                        Blas<PetscScalar>::axpy(&m, S, in, &i__1, out, &i__1);
                    delete [] work;
                    PetscCall(MatKAIJRestoreTRead(A, &T));
                    PetscCall(MatKAIJRestoreSRead(A, &S));
                    if(reset) {
                        if(std::is_same<PetscScalar, K>::value)
                            PetscCall(MatDenseResetArray(_X[0]));
                        PetscCall(MatDenseResetArray(_X[1]));
                    }
                    if(!std::is_same<PetscScalar, K>::value) {
                        const PetscScalar* work;
                        PetscCall(MatDenseGetArrayRead(_X[0], &work));
                        HPDDM::copy_n(work, m, out);
                        PetscCall(MatDenseRestoreArrayRead(_X[0], &work));
                    }
                    PetscFunctionReturn(0);
                }
            }
            if(mu == 1) {
                const PetscScalar *read;
                PetscScalar       *write;
                if(!_b) {
                    PetscInt N;
                    PetscCall(MatGetSize(A, &N, NULL));
                    if(std::is_same<PetscScalar, K>::value) {
                        PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, N, NULL, const_cast<Vec*>(&_b)));
                        PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, N, NULL, const_cast<Vec*>(&_x)));
                    }
                    else {
                        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)_ksp), super::_n, N, const_cast<Vec*>(&_b)));
                        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)_ksp), super::_n, N, const_cast<Vec*>(&_x)));
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    PetscCall(VecGetArrayWrite(_b, &write));
                    std::copy_n(in, super::_n, write);
                    PetscCall(VecGetArrayRead(_x, &read));
                }
                else {
                    PetscCall(VecPlaceArray(_b, reinterpret_cast<const PetscScalar*>(in)));
                    PetscCall(VecPlaceArray(_x, reinterpret_cast<PetscScalar*>(out)));
                }
                PetscCall(MatMult(A, _b, _x));
                if(std::is_same<PetscScalar, K>::value) {
                    PetscCall(VecResetArray(_x));
                    PetscCall(VecResetArray(_b));
                }
                else {
                    HPDDM::copy_n(read, super::_n, out);
                    PetscCall(VecRestoreArrayRead(_x, &read));
                    PetscCall(VecRestoreArrayWrite(_b, &write));
                }
            }
            else {
                PC pc;
                Mat *ptr;
                PetscContainer container = NULL;
                PetscInt M = 0;
                bool reset = false;
                PetscCall(KSPGetPC(_ksp, &pc));
                PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCHPDDM, &flg));
                if(flg) {
                    PetscCall(PetscObjectQuery((PetscObject)A, "_HPDDM_MatProduct", (PetscObject*)&container));
                    if(container) {
                        PetscCall(PetscContainerGetPointer(container, (void**)&ptr));
                        if(ptr[1] != _X[1])
                            for(unsigned short i = 0; i < 2; ++i) {
                                PetscCall(MatDestroy(x + i));
                                x[i] = ptr[i];
                                PetscCall(PetscObjectReference((PetscObject)x[i]));
                            }
                    }
                }
                if(_X[0])
                    PetscCall(MatGetSize(_X[0], NULL, &M));
                if(M != mu) {
                    PetscInt N;
                    PetscCall(MatGetSize(A, &N, NULL));
                    PetscCall(MatDestroy(x));
                    PetscCall(MatDestroy(x + 1));
                    if(flg) {
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
                        PCHPDDMCoarseCorrectionType type;
                        PetscCall(PCHPDDMGetCoarseCorrectionType(pc, &type));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, NULL, x + 1));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, (!std::is_same<PetscScalar, K>::value || type == PC_HPDDM_COARSE_CORRECTION_BALANCED) ? NULL : reinterpret_cast<PetscScalar*>(out), x));
#endif
                    }
                    else {
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(const_cast<K*>(in)) : NULL, x + 1));
                        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : NULL, x));
                    }
                    PetscCall(MatProductCreateWithMat(A, _X[1], NULL, _X[0]));
                    PetscCall(MatProductSetType(_X[0], MATPRODUCT_AB));
                    PetscCall(MatProductSetFromOptions(_X[0]));
                    PetscCall(MatProductSymbolic(_X[0]));
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
                    reset = true;
                    if(container)
                        PetscCall(MatProductReplaceMats(NULL, _X[1], NULL, _X[0]));
                }
                if(std::is_same<PetscScalar, K>::value) {
                    if (reset) {
                        PetscCall(MatDensePlaceArray(_X[1], reinterpret_cast<PetscScalar*>(const_cast<K*>(in))));
                        PetscCall(MatDensePlaceArray(_X[0], reinterpret_cast<PetscScalar*>(out)));
                    }
                }
                else {
                    PetscScalar* work;
                    PetscCall(MatDenseGetArrayWrite(_X[1], &work));
                    std::copy_n(in, mu * super::_n, work);
                    PetscCall(MatDenseRestoreArrayWrite(_X[1], &work));
                }
                PetscCall(MatProductNumeric(_X[0]));
                if(std::is_same<PetscScalar, K>::value) {
                    if(reset) {
                        PetscCall(MatDenseResetArray(_X[0]));
                        PetscCall(MatDenseResetArray(_X[1]));
                    }
                }
                else {
                    const PetscScalar* work;
                    PetscCall(MatDenseGetArrayRead(_X[0], &work));
                    HPDDM::copy_n(work, mu * super::_n, out);
                    PetscCall(MatDenseRestoreArrayRead(_X[0], &work));
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

            PetscFunctionBeginUser;
            PetscCall(KSPGetPC(_ksp, &pc));
            PetscCall(KSPGetOperators(_ksp, &A, NULL));
            PetscCall(MatGetSize(A, &N, NULL));
            PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &match, MATSEQKAIJ, MATMPIKAIJ, ""));
            if(match) {
                PetscInt bs, n;
                PetscBool id;
                PetscCall(MatGetLocalSize(A, &n, NULL));
                PetscCall(MatGetBlockSize(A, &bs));
                PetscCall(MatKAIJGetScaledIdentity(A, &id));
                const unsigned short eta = (id == PETSC_TRUE ? mu / bs : mu);
                PetscCheck(id != PETSC_TRUE || eta * bs == mu, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled case %d != %d", static_cast<int>(eta * bs), static_cast<int>(mu));
                if(!_b) {
                    if(std::is_same<PetscScalar, K>::value) {
                        PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, n, N, NULL, const_cast<Vec*>(&_b)));
                        PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, n, N, NULL, const_cast<Vec*>(&_x)));
                    }
                    else {
                        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)_ksp), n, N, const_cast<Vec*>(&_b)));
                        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)_ksp), n, N, const_cast<Vec*>(&_x)));
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    PetscCall(VecGetArrayWrite(_b, &write));
                    PetscCall(VecGetArrayRead(_x, &read));
                }
                for(unsigned short nu = 0; nu < eta; ++nu) {
                    if(id)
                        Wrapper<K>::template imatcopy<'T'>(bs, super::_n, const_cast<K*>(in + nu * n), super::_n, bs);
                    if(std::is_same<PetscScalar, K>::value) {
                        PetscCall(VecPlaceArray(_b, reinterpret_cast<const PetscScalar*>(in + nu * n)));
                        PetscCall(VecPlaceArray(_x, reinterpret_cast<PetscScalar*>(out + nu * n)));
                    }
                    else
                        std::copy_n(in + nu * n, n, write);
                    PetscCall(PCApply(pc, _b, _x));
                    if(std::is_same<PetscScalar, K>::value) {
                        PetscCall(VecResetArray(_x));
                        PetscCall(VecResetArray(_b));
                    }
                    else
                        HPDDM::copy_n(read, n, out + nu * n);
                    if(id) {
                        Wrapper<K>::template imatcopy<'T'>(super::_n, bs, const_cast<K*>(in + nu * n), bs, super::_n);
                        Wrapper<K>::template imatcopy<'T'>(super::_n, bs, out + nu * n, bs, super::_n);
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    PetscCall(VecRestoreArrayRead(_x, &read));
                    PetscCall(VecRestoreArrayWrite(_b, &write));
                }
                PetscFunctionReturn(0);
            }
            if(mu > 1) {
                PetscInt M = 0;
                bool reset = false;
                if(_Y)
                    PetscCall(MatGetSize(_Y, NULL, &M));
                if(M != mu) {
                    PetscCall(MatDestroy(const_cast<Mat*>(&_Y)));
                    PetscCall(MatDestroy(const_cast<Mat*>(&_C)));
                    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(const_cast<K*>(in)) : NULL, const_cast<Mat*>(&_C)));
                    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)_ksp), super::_n, PETSC_DECIDE, N, mu, std::is_same<PetscScalar, K>::value ? reinterpret_cast<PetscScalar*>(out) : NULL, const_cast<Mat*>(&_Y)));
                }
                else if(std::is_same<PetscScalar, K>::value) {
                    PetscCall(MatDensePlaceArray(_C, reinterpret_cast<PetscScalar*>(const_cast<K*>(in))));
                    PetscCall(MatDensePlaceArray(_Y, reinterpret_cast<PetscScalar*>(out)));
                    reset = true;
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    PetscScalar* work;
                    PetscCall(MatDenseGetArrayWrite(_C, &work));
                    std::copy_n(in, mu * super::_n, work);
                    PetscCall(MatDenseRestoreArrayWrite(_C, &work));
                }
                PetscCall(PCMatApply(pc, _C, _Y));
                if(reset) {
                    PetscCall(MatDenseResetArray(_Y));
                    PetscCall(MatDenseResetArray(_C));
                }
                else if(!std::is_same<PetscScalar, K>::value) {
                    const PetscScalar* work;
                    PetscCall(MatDenseGetArrayRead(_Y, &work));
                    HPDDM::copy_n(work, mu * super::_n, out);
                    PetscCall(MatDenseRestoreArrayRead(_Y, &work));
                }
            }
            else {
                if(!_b) {
                    if(std::is_same<PetscScalar, K>::value) {
                        PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, N, NULL, const_cast<Vec*>(&_b)));
                        PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)_ksp), 1, super::_n, N, NULL, const_cast<Vec*>(&_x)));
                    }
                    else {
                        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)_ksp), super::_n, N, const_cast<Vec*>(&_b)));
                        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)_ksp), super::_n, N, const_cast<Vec*>(&_x)));
                    }
                }
                if(!std::is_same<PetscScalar, K>::value) {
                    PetscCall(VecGetArrayWrite(_b, &write));
                    std::copy_n(in, super::_n, write);
                    PetscCall(VecGetArrayRead(_x, &read));
                }
                else {
                    PetscCall(VecPlaceArray(_b, reinterpret_cast<const PetscScalar*>(in)));
                    PetscCall(VecPlaceArray(_x, reinterpret_cast<PetscScalar*>(out)));
                }
                PetscCall(PCApply(pc, _b, _x));
                if(std::is_same<PetscScalar, K>::value) {
                    PetscCall(VecResetArray(_x));
                    PetscCall(VecResetArray(_b));
                }
                else {
                    HPDDM::copy_n(read, super::_n, out);
                    PetscCall(VecRestoreArrayRead(_x, &read));
                    PetscCall(VecRestoreArrayWrite(_b, &write));
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
    PetscFunctionBeginUser;
    PetscCall(MatCreate(PETSC_COMM_SELF, &P));
    PetscCall(MatSetSizes(P, A->_n, A->_m, A->_n, A->_m));
    if(!A->_sym) {
        PetscCall(MatSetType(P, MATSEQAIJ));
        if(std::is_same<int, PetscInt>::value) {
            PetscCall(MatSeqAIJSetPreallocationCSR(P, reinterpret_cast<PetscInt*>(A->_ia), reinterpret_cast<PetscInt*>(A->_ja), A->_a));
        }
        else {
            PetscInt* I = new PetscInt[A->_n + 1];
            PetscInt* J = new PetscInt[A->_nnz];
            std::copy_n(A->_ia, A->_n + 1, I);
            std::copy_n(A->_ja, A->_nnz, J);
            PetscCall(MatSeqAIJSetPreallocationCSR(P, I, J, A->_a));
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
        PetscCall(MatSetType(P, MATSEQSBAIJ));
        if(!std::is_same<int, PetscInt>::value) {
            int* ia = reinterpret_cast<int*>(I);
            int* ja = reinterpret_cast<int*>(J);
            for(unsigned int i = A->_n + 1; i-- > 0; )
                I[i] = ia[i];
            for(unsigned int i = A->_nnz; i-- > 0; )
                J[i] = ja[i];
        }
        PetscCall(MatSeqSBAIJSetPreallocationCSR(P, 1, I, J, C));
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
            PetscFunctionBeginUser;
            if(_op)
                PetscCall(KSPDestroy(const_cast<KSP*>(&_op->_ksp)));
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
            const Option& opt = *Option::get();
            PetscFunctionBeginUser;
            if(!_op)
                PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
            else
                ksp = _op->_ksp;
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
            PetscCall(PCSetType(pc, A->_sym ? PCCHOLESKY : PCLU));
            PetscCall(KSPSetFromOptions(ksp));
            PetscCall(KSPSetUp(ksp));
            if(opt.val<char>("verbosity", 0) >= 4)
                PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_SELF));
            if(!_op)
                _op = new PETScOperator(ksp, A->_n);
            PetscFunctionReturn(0);
        }
        PetscErrorCode solve(K* const x, const unsigned short& n = 1) const {
            PetscFunctionBeginUser;
            if(_op) {
                K* b = new K[n * _op->super::_n];
                PetscCallCXX(std::copy_n(x, n * _op->super::_n, b));
                PetscCall(_op->apply(b, x, n));
                delete [] b;
            }
            PetscFunctionReturn(0);
        }
        PetscErrorCode solve(const K* const b, K* const x, const unsigned short& n = 1) const {
            PetscFunctionBeginUser;
            if(_op)
                PetscCall(_op->apply(b, x, n));
            PetscFunctionReturn(0);
        }
};
#endif // PETSCSUB
} // HPDDM
#endif // _HPDDM_PETSC_
