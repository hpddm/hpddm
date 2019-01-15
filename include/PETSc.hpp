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

#include "petsc/private/kspimpl.h"

namespace HPDDM {
struct PETScOperator : public HPDDM::EmptyOperator<PetscScalar> {
    const KSP     _ksp;
    const PetscInt _bs;
    PetscErrorCode (*const _apply)(PC, Mat, Mat);
    PETScOperator(const PETScOperator&) = delete;
    PETScOperator(const KSP& ksp, int n, PetscInt bs, PetscErrorCode (*apply)(PC, Mat, Mat) = nullptr) : HPDDM::EmptyOperator<PetscScalar>(bs * n), _ksp(ksp), _bs(bs), _apply(apply) { }
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
            VecCreateMPIWithArray(comm, _bs, HPDDM::EmptyOperator<PetscScalar>::_n, N, NULL, &right);
            VecCreateMPIWithArray(comm, _bs, HPDDM::EmptyOperator<PetscScalar>::_n, N, NULL, &left);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                VecPlaceArray(right, in + nu * HPDDM::EmptyOperator<PetscScalar>::_n);
                VecPlaceArray(left, out + nu * HPDDM::EmptyOperator<PetscScalar>::_n);
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
            MatCreateDense(comm, HPDDM::EmptyOperator<PetscScalar>::_n, PETSC_DECIDE, N, mu, const_cast<PetscScalar*>(in), &B);
            MatCreateDense(comm, HPDDM::EmptyOperator<PetscScalar>::_n, PETSC_DECIDE, N, mu, out, &C);
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
            MatCreateDense(comm, HPDDM::EmptyOperator<PetscScalar>::_n, PETSC_DECIDE, N, mu, const_cast<PetscScalar*>(in), &B);
            MatCreateDense(comm, HPDDM::EmptyOperator<PetscScalar>::_n, PETSC_DECIDE, N, mu, out, &C);
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
                MatCreateDense(comm, HPDDM::EmptyOperator<PetscScalar>::_n, PETSC_DECIDE, N, mu, const_cast<PetscScalar*>(in), &B);
                MatCreateDense(comm, HPDDM::EmptyOperator<PetscScalar>::_n, PETSC_DECIDE, N, mu, out, &C);
                _apply(pc, B, C);
                MatDestroy(&C);
                MatDestroy(&B);
            }
            else {
                Vec right, left;
                VecCreateMPIWithArray(comm, _bs, HPDDM::EmptyOperator<PetscScalar>::_n, N, NULL, &right);
                VecCreateMPIWithArray(comm, _bs, HPDDM::EmptyOperator<PetscScalar>::_n, N, NULL, &left);
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    VecPlaceArray(right, in + nu * HPDDM::EmptyOperator<PetscScalar>::_n);
                    VecPlaceArray(left, out + nu * HPDDM::EmptyOperator<PetscScalar>::_n);
                    PCApply(pc, right, left);
                    VecResetArray(left);
                    VecResetArray(right);
                }
                VecDestroy(&right);
                VecDestroy(&left);
            }
        }
    }
};

static PetscErrorCode KSPSetFromOptions_HPDDM(PetscOptionItems* PetscOptionsObject, KSP ksp) {
    PetscErrorCode ierr;
    const char* prefix;
    MPI_Comm comm;
    PetscMPIInt rank;
    PetscFunctionBegin;
    ierr = PetscOptionsHead(PetscOptionsObject, "KSP HPDDM Options, cf. https://github.com/hpddm/hpddm/blob/master/doc/cheatsheet.pdf"); CHKERRQ(ierr);
    PetscOptionsTail();
    HPDDM::Option& opt = *HPDDM::Option::get();
    ierr = KSPGetOptionsPrefix(ksp, &prefix); CHKERRQ(ierr);
    std::string p = prefix ? std::string(prefix) : "";

    if(ksp->pc_side == PC_RIGHT)
        opt[p + "variant"] = 1;
    else
        opt[p + "variant"] = 0;
    opt[p + "tol"] = ksp->rtol;
    opt[p + "max_it"] = ksp->max_it;
#if !(PETSC_VERSION_LT(3,8,0))
    {
        char** names, **values;
        PetscInt N;
        ierr = PetscOptionsLeftGet(NULL, &N, &names, &values); CHKERRQ(ierr);
        std::vector<std::string> optionsLeft;
        optionsLeft.reserve(2 * N);
        for(PetscInt i = 0; i < N; ++i) {
            std::string name(names[i]);
            if(name.find(std::string(HPDDM_PREFIX) + p) != std::string::npos) {
                optionsLeft.emplace_back("-" + name);
                if(values[i])
                    optionsLeft.emplace_back(std::string(values[i]));
            }
        }
        ierr = PetscOptionsLeftRestore(NULL, &N, &names, &values); CHKERRQ(ierr);
        opt.parse<false, true>(optionsLeft, false, { }, p);
    }
#endif
    PetscObjectGetComm((PetscObject)ksp, &comm);
    MPI_Comm_rank(comm, &rank);
    if(rank != 0)
        opt.remove(p + "verbosity");
    ksp->rtol = opt[p + "tol"];
    ksp->max_it = opt[p + "max_it"];
    int variant = opt[p + "variant"];
    if(variant == 0)
        ksp->pc_side = PC_LEFT;
    else
        ksp->pc_side = PC_RIGHT;
    PetscFunctionReturn(0);
}
static PetscErrorCode KSPSetUp_HPDDM(KSP ksp) {
    PetscErrorCode ierr;
    const char* prefix;
    PetscInt n;
    Mat A;
    PetscFunctionBegin;
    ierr = KSPGetOperators(ksp, &A, NULL); CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &n, NULL); CHKERRQ(ierr);
    PETScOperator* op = new PETScOperator(ksp, n, 1);
    ksp->data = reinterpret_cast<void*>(op);
    ierr = KSPGetOptionsPrefix(ksp, &prefix); CHKERRQ(ierr);
    if(prefix)
        op->setPrefix(prefix);
    PetscFunctionReturn(0);
}
static PetscErrorCode KSPDestroy_HPDDM(KSP ksp) {
    PetscErrorCode ierr;
    PetscFunctionBegin;
    delete reinterpret_cast<PETScOperator*>(ksp->data);
    ksp->data = NULL;
    ierr = KSPDestroyDefault(ksp); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
static PetscErrorCode KSPSolve_HPDDM(KSP ksp) {
    PetscErrorCode ierr;
    MPI_Comm comm;
    PetscScalar* x;
    const PetscScalar* b;
    PetscFunctionBegin;
    ierr = VecGetArray(ksp->vec_sol, &x); CHKERRQ(ierr);
    ierr = VecGetArrayRead(ksp->vec_rhs, &b); CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)ksp, &comm); CHKERRQ(ierr);
    const PETScOperator& op = *reinterpret_cast<PETScOperator*>(ksp->data);
    ksp->its = HPDDM::IterativeMethod::solve(op, b, x, 1, comm);
    ierr = VecRestoreArrayRead(ksp->vec_rhs, &b); CHKERRQ(ierr);
    ierr = VecRestoreArray(ksp->vec_sol, &x); CHKERRQ(ierr);
    if(ksp->its < ksp->max_it)
        ksp->reason = KSP_CONVERGED_RTOL;
    else
        ksp->reason = KSP_DIVERGED_ITS;
    PetscFunctionReturn(0);
}
static PetscErrorCode KSPCreate_HPDDM(KSP ksp) {
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2); CHKERRQ(ierr);
    ierr = KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 1); CHKERRQ(ierr);
    ksp->ops->setup          = KSPSetUp_HPDDM;
    ksp->ops->solve          = KSPSolve_HPDDM;
    ksp->ops->destroy        = KSPDestroy_HPDDM;
    ksp->ops->setfromoptions = KSPSetFromOptions_HPDDM;
    PetscFunctionReturn(0);
}
} // HPDDM
#endif // _HPDDM_PETSC_
