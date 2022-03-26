 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-12-21

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
                 2016-     Centre National de la Recherche Scientifique

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

#ifndef _HPDDM_CG_
#define _HPDDM_CG_

#include "HPDDM_iterative.hpp"

namespace HPDDM {
template<bool excluded, class Operator, class K>
inline int IterativeMethod::CG(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
#if !defined(_KSPIMPL_H)
    underlying_type<K> tol;
    unsigned short it;
    char id[2];
    {
        const std::string prefix = A.prefix();
        const Option& opt = *Option::get();
        if((hpddm_method_id<Operator>::value == 1 || hpddm_method_id<Operator>::value == 4) && (!opt.any_of(prefix + "schwarz_method", { HPDDM_SCHWARZ_METHOD_SORAS, HPDDM_SCHWARZ_METHOD_ASM, HPDDM_SCHWARZ_METHOD_NONE }) || opt.any_of(prefix + "schwarz_coarse_correction", { HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED })))
            return GMRES<excluded>(A, b, x, mu, comm);
        options<2>(A, &tol, nullptr, &it, id);
    }
#else
    char* id = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl;
#endif
    const int n = excluded ? 0 : A.getDof();
    const int dim = n * mu;
    underlying_type<K>* res;
    K* trash;
    allocate(res, trash, n, id[1] == HPDDM_VARIANT_FLEXIBLE ? 1 : 0, HPDDM_MAX_IT(it, A), mu);
    short* const hasConverged = new short[mu];
    std::fill_n(hasConverged, mu, -HPDDM_MAX_IT(it, A));
    underlying_type<K>* const dir = res + mu;
    K* const z = trash + dim;
    K* const r = z + dim;
    K* const p = r + dim;
    const underlying_type<K>* const d = reinterpret_cast<const underlying_type<K>*>(A.getScaling());
    bool allocate = A.template start<excluded>(b, x, mu);
    if(!excluded)
        HPDDM_CALL(A.GMV(x, z, mu));
    std::copy_n(b, dim, r);
    Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), z, &i__1, r, &i__1);
    HPDDM_CALL(A.template apply<excluded>(r, p, mu, z));
#if defined(_KSPIMPL_H)
    underlying_type<K>* norm = new underlying_type<K>[mu];
    PetscCall(A.template apply<excluded>(b, z, mu, trash));
    Wrapper<K>::diag(n, d, z, trash, mu);
    for(unsigned short nu = 0; nu < mu; ++nu)
        norm[nu] = std::real(Blas<K>::dot(&n, z + n * nu, &i__1, trash + n * nu, &i__1));
    MPI_Allreduce(MPI_IN_PLACE, norm, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
    std::for_each(norm, norm + mu, [](underlying_type<K>& y) { y = std::sqrt(y); });
#endif
    Wrapper<K>::diag(n, d, p, trash, mu);
    for(unsigned short nu = 0; nu < mu; ++nu)
        dir[nu] = std::real(Blas<K>::dot(&n, trash + n * nu, &i__1, p + n * nu, &i__1));
    MPI_Allreduce(MPI_IN_PLACE, dir, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
    std::transform(dir, dir + mu, res, [](const underlying_type<K>& d) { return std::sqrt(d); });
#if defined(_KSPIMPL_H)
    if(A._ksp->guess_zero) {
        A._ksp->rnorm = *dir / *norm;
        for(unsigned short nu = 1; nu < mu; ++nu)
            A._ksp->rnorm = std::max(A._ksp->rnorm, PetscReal(dir[nu] / norm[nu]));
        PetscCall(KSPLogResidualHistory(A._ksp, A._ksp->rnorm));
        PetscCall(KSPMonitor(A._ksp, 0, A._ksp->rnorm));
    }
    else {
        for(unsigned short nu = 0; nu < mu; ++nu)
            if(res[nu] / norm[nu] < HPDDM_TOL(tol, A))
                dir[nu] = std::numeric_limits<underlying_type<K>>::epsilon() / 1000.0;
        A._ksp->rnorm = *std::max_element(res, res + mu);
        PetscCall(KSPLogResidualHistory(A._ksp, A._ksp->rnorm));
        PetscCall(KSPMonitor(A._ksp, 0, A._ksp->rnorm));
    }
    PetscCall((*A._ksp->converged)(A._ksp, 0, A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP));
    delete [] norm;
    if(A._ksp->reason)
        std::fill_n(dir, mu, underlying_type<K>());
    else if(A._ksp->converged == KSPConvergedSkip)
        dir[0] = 1.0;
#endif
    int i = 0;
    if(std::find_if(dir, dir + mu, [](const underlying_type<K>& v) { return v < std::pow(std::numeric_limits<underlying_type<K>>::epsilon(), 2.0); }) == dir + mu) {
        while(i < HPDDM_MAX_IT(it, A)) {
            for(unsigned short nu = 0; nu < mu; ++nu)
                dir[nu] = std::real(Blas<K>::dot(&n, r + n * nu, &i__1, trash + n * nu, &i__1));
            if(!excluded)
                HPDDM_CALL(A.GMV(p, z, mu));
            if(id[1] != HPDDM_VARIANT_FLEXIBLE)
                Wrapper<K>::diag(n, d, p, trash, mu);
            for(unsigned short nu = 0; nu < mu; ++nu)
                dir[mu + nu] = std::real(Blas<K>::dot(&n, z + n * nu, &i__1, trash + n * nu, &i__1));
            MPI_Allreduce(MPI_IN_PLACE, dir, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
            if(id[1] == HPDDM_VARIANT_FLEXIBLE) {
                std::copy_n(p, dim, p + (i + 1) * dim);
                std::copy_n(dir + mu, mu, dir + (HPDDM_MAX_IT(it, A) + i + 2) * mu);
                std::copy_n(z, dim, p + (HPDDM_MAX_IT(it, A) + i + 1) * dim);
            }
            ++i;
            for(unsigned short nu = 0; nu < mu; ++nu) {
                if(hasConverged[nu] == -HPDDM_MAX_IT(it, A)) {
                    trash[nu] = dir[nu] / dir[mu + nu];
                    Blas<K>::axpy(&n, trash + nu, p + n * nu, &i__1, x + n * nu, &i__1);
                    trash[nu] = -trash[nu];
                    Blas<K>::axpy(&n, trash + nu, z + n * nu, &i__1, r + n * nu, &i__1);
                }
            }
            HPDDM_CALL(A.template apply<excluded>(r, z, mu, trash));
            Wrapper<K>::diag(n, d, z, trash, mu);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                if(id[1] != HPDDM_VARIANT_FLEXIBLE)
                    dir[mu + nu] = std::real(Blas<K>::dot(&n, r + n * nu, &i__1, trash + n * nu, &i__1)) / dir[nu];
                dir[nu] = std::real(Blas<K>::dot(&n, z + n * nu, &i__1, trash + n * nu, &i__1));
            }
            if(id[1] != HPDDM_VARIANT_FLEXIBLE) {
                MPI_Allreduce(MPI_IN_PLACE, dir, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
                for(unsigned short nu = 0; nu < mu; ++nu)
                    Blas<K>::axpby(n, 1.0, z + n * nu, 1, dir[mu + nu], p + n * nu, 1);
            }
            else {
                for(unsigned short k = 0; k < i; ++k)
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        dir[2 * mu + k * mu + nu] = -std::real(Blas<K>::dot(&n, trash + n * nu, &i__1, p + (HPDDM_MAX_IT(it, A) + k + 1) * dim + n * nu, &i__1)) / dir[(HPDDM_MAX_IT(it, A) + k + 2) * mu + nu];
                MPI_Allreduce(MPI_IN_PLACE, dir, (i + 2) * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
                if(!excluded && n) {
                    std::copy_n(z, dim, p);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        for(unsigned short k = 0; k < i; ++k)
                            trash[k] = dir[2 * mu + k * mu + nu];
                        Blas<K>::gemv("N", &n, &i, &(Wrapper<K>::d__1), p + dim + n * nu, &dim, trash, &i__1, &(Wrapper<K>::d__1), p + nu * n, &i__1);
                    }
                }
                Wrapper<K>::diag(n, d, p, trash, mu);
            }
            std::for_each(dir, dir + mu, [](underlying_type<K>& d) { d = std::sqrt(d); });
            checkConvergence<2>(id[0], i, i, HPDDM_TOL(tol, A), mu, res, dir, hasConverged, HPDDM_MAX_IT(it, A));
#if HPDDM_PETSC
            PetscCall(KSPLogResidualHistory(A._ksp, *dir));
            PetscCall(KSPMonitor(A._ksp, i, *dir));
            PetscCall((*A._ksp->converged)(A._ksp, i, *dir, &A._ksp->reason, A._ksp->cnvP));
            if(A._ksp->reason)
                std::for_each(hasConverged, hasConverged + mu, [&](short& c) { if(c == -HPDDM_MAX_IT(it, A)) c = i; });
            else if(A._ksp->converged == KSPConvergedSkip)
                std::fill_n(hasConverged, mu, -HPDDM_MAX_IT(it, A));
#endif
            if(std::find(hasConverged, hasConverged + mu, -HPDDM_MAX_IT(it, A)) == hasConverged + mu) {
                --i;
                break;
            }
        }
    }
    else
        i = -1;
    ++i;
#if !defined(_KSPIMPL_H)
    convergence<2>(id[0], i, HPDDM_MAX_IT(it, A));
#else
    A._ksp->its = i;
#endif
    delete [] res;
    if(Wrapper<K>::is_complex)
        delete [] trash;
    delete [] hasConverged;
    A.end(allocate);
    return HPDDM_RET(std::min(static_cast<unsigned short>(i), HPDDM_MAX_IT(it, A)));
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::BCG(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
#if !defined(_KSPIMPL_H)
    underlying_type<K> tol;
    unsigned short i, m[2];
    char id[2];
    {
        const std::string prefix = A.prefix();
        const Option& opt = *Option::get();
        if((hpddm_method_id<Operator>::value == 1 || hpddm_method_id<Operator>::value == 4) && (!opt.any_of(prefix + "schwarz_method", { HPDDM_SCHWARZ_METHOD_SORAS, HPDDM_SCHWARZ_METHOD_ASM, HPDDM_SCHWARZ_METHOD_NONE }) || opt.any_of(prefix + "schwarz_coarse_correction", { HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED })))
            return GMRES<excluded>(A, b, x, mu, comm);
        options<3>(A, &tol, nullptr, m, id);
        if(opt.val<char>(prefix + "variant", HPDDM_VARIANT_LEFT) == HPDDM_VARIANT_FLEXIBLE)
            return CG<excluded>(A, b, x, mu, comm);
        m[0] = opt.val<unsigned short>(prefix + "enlarge_krylov_subspace", 1);
    }
#else
    unsigned short* m = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->scntl;
    char* id = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl;
#endif
    const int n = excluded ? 0 : A.getDof();
    const int dim = n * mu;
    K* const trash = new K[4 * (dim + mu * mu)];
    K* const p = trash + dim;
    K* const z = p + dim;
    K* const r = z + dim;
    K* const rho = r + dim;
    K* const rhs = rho + 2 * mu * mu;
    K* const gamma = rhs + mu * mu;
    const underlying_type<K>* const d = reinterpret_cast<const underlying_type<K>*>(A.getScaling());
    bool allocate = A.template start<excluded>(b, x, mu);
    if(!excluded)
        HPDDM_CALL(A.GMV(x, z, mu));
    std::copy_n(b, dim, r);
    Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), z, &i__1, r, &i__1);
    HPDDM_CALL(A.template apply<excluded>(r, p, mu, z));
    Wrapper<K>::diag(n, d, p, trash, mu);
    if(!excluded && n) {
        Blas<K>::gemmt("U", &(Wrapper<K>::transc), "N", &mu, &n, &(Wrapper<K>::d__1), r, &n, trash, &n, &(Wrapper<K>::d__0), rho, &mu);
        for(unsigned short nu = 1; nu < mu; ++nu)
            std::copy_n(rho + nu * mu, nu + 1, rho + (nu * (nu + 1)) / 2);
    }
    else
        std::fill_n(rho, (mu * (mu + 1)) / 2, K());
    MPI_Allreduce(MPI_IN_PLACE, rho, (mu * (mu + 1)) / 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
    for(unsigned short nu = mu; nu > 0; --nu)
        std::copy_backward(rho + (nu * (nu - 1)) / 2, rho + (nu * (nu + 1)) / 2, rho + nu * mu - (mu - nu));
    for(unsigned short i = 0; i < mu; ++i)
        for(unsigned short j = 0; j < i; ++j)
            rho[i + j * mu] = Wrapper<K>::conj(rho[j + i * mu]);
    std::copy_n(rho, mu * mu, rho + mu * mu);
    int info = -1;
    for(unsigned short i = 0; i < mu && info == -1; ++i)
        if(std::abs(rho[(i + 1) * mu]) > 10.0 * std::numeric_limits<underlying_type<K>>::epsilon())
            info = 0;
    if(info == 0)
        info = QR<excluded>(id[1], n, mu, p, gamma, mu, d, trash, comm);
    if(info != mu) {
        delete [] trash;
        A.end(allocate);
#if defined(_KSPIMPL_H)
        A._ksp->reason = (info != -1 ? KSP_DIVERGED_BREAKDOWN : KSP_CONVERGED_HAPPY_BREAKDOWN);
#endif
        return HPDDM_RET(CG<excluded>(A, b, x, mu, comm));
    }
#if !defined(_KSPIMPL_H)
    diagonal<3>(id[0], gamma, mu);
#endif
    underlying_type<K>* const norm = new underlying_type<K>[mu];
    if(m[0] <= 1)
        for(unsigned short nu = 0; nu < mu; ++nu)
            norm[nu] = Blas<K>::nrm2(&(info = nu + 1), gamma + mu * nu, &i__1);
    else {
        std::fill_n(z, m[0], K());
        for(unsigned short nu = 0; nu < m[0]; ++nu)
            Blas<K>::axpy(&(info = nu + 1), &(Wrapper<K>::d__1), gamma + mu * nu, &i__1, z, &i__1);
        *norm = Blas<K>::nrm2(&(info = m[0]), z, &i__1);
    }
#if defined(_KSPIMPL_H)
    A._ksp->rnorm = *std::max_element(norm, norm + (m[0] <= 1 ? mu : 1));
    PetscCall(KSPLogResidualHistory(A._ksp, A._ksp->rnorm));
    PetscCall(KSPMonitor(A._ksp, 0, A._ksp->rnorm));
    PetscCall((*A._ksp->converged)(A._ksp, 0, A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP));
    if(A._ksp->reason) {
        delete [] norm;
        delete [] trash;
        A.end(allocate);
        return 0;
    }
#endif
    HPDDM_IT(i, A) = 1;
    while(HPDDM_IT(i, A) <= HPDDM_MAX_IT(m[1], A)) {
        if(!excluded) {
            HPDDM_CALL(A.GMV(p, z, mu));
            Blas<K>::trsm("L", "U", &(Wrapper<K>::transc), "N", &mu, &mu, &(Wrapper<K>::d__1), gamma, &mu, rho + mu * mu, &mu);
        }
        Wrapper<K>::diag(n, d, z, trash, mu);
        if(!excluded && n) {
            Blas<K>::gemmt("U", &(Wrapper<K>::transc), "N", &mu, &n, &(Wrapper<K>::d__1), p, &n, trash, &n, &(Wrapper<K>::d__0), rhs, &mu);
            for(unsigned short nu = 1; nu < mu; ++nu)
                std::copy_n(rhs + nu * mu, nu + 1, rhs + (nu * (nu + 1)) / 2);
        }
        else
            std::fill_n(rhs, (mu * (mu + 1)) / 2, K());
        MPI_Allreduce(MPI_IN_PLACE, rhs, (mu * (mu + 1)) / 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
        Lapack<K>::ppsv("U", &mu, &mu, rhs, rho + mu * mu, &mu, &info);
        if(info) {
            delete [] norm;
            delete [] trash;
            A.end(allocate);
#if defined(_KSPIMPL_H)
            A._ksp->reason = KSP_DIVERGED_BREAKDOWN;
#endif
            return HPDDM_RET(CG<excluded>(A, b, x, mu, comm));
        }
        if(!excluded && n) {
            Blas<K>::gemm("N", "N", &n, &mu, &mu, &(Wrapper<K>::d__1), p, &n, rho + mu * mu, &mu, &(Wrapper<K>::d__1), x, &n);
            Blas<K>::gemm("N", "N", &n, &mu, &mu, &(Wrapper<K>::d__2), z, &n, rho + mu * mu, &mu, &(Wrapper<K>::d__1), r, &n);
        }
        HPDDM_CALL(A.template apply<excluded>(r, z, mu, trash));
        Wrapper<K>::diag(n, d, z, trash, mu);
        if(!excluded && n) {
            Blas<K>::gemmt("U", &(Wrapper<K>::transc), "N", &mu, &n, &(Wrapper<K>::d__1), r, &n, trash, &n, &(Wrapper<K>::d__0), rhs, &mu);
            for(unsigned short nu = 1; nu < mu; ++nu)
                std::copy_n(rhs + nu * mu, nu + 1, rhs + (nu * (nu + 1)) / 2);
            if(m[0] <= 1)
                for(unsigned short nu = 0; nu < mu; ++nu)
                    rho[(2 * mu - 1) * mu + nu] = std::real(Blas<K>::dot(&n, z + n * nu, &i__1, trash + n * nu, &i__1));
            else {
                for(unsigned short nu = 1; nu < m[0]; ++nu)
                    Blas<K>::axpy(&n, &(Wrapper<K>::d__1), trash + nu * n, &i__1, trash, &i__1);
                std::copy_n(z, n, trash + n);
                for(unsigned short nu = 1; nu < m[0]; ++nu)
                    Blas<K>::axpy(&n, &(Wrapper<K>::d__1), z + nu * n, &i__1, trash + n, &i__1);
                rho[2 * mu * mu - 1] = std::real(Blas<K>::dot(&n, trash, &i__1, trash + n, &i__1));
            }
        }
        else
            std::fill_n(rho + (2 * mu - 1) * mu, mu + (mu * (mu + 1)) / 2, K());
        MPI_Allreduce(MPI_IN_PLACE, rhs - mu / (m[0] <= 1 ? mu : 1), mu / (m[0] <= 1 ? mu : 1) + (mu * (mu + 1)) / 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
        bool converged = (mu == checkBlockConvergence<3>(id[0], HPDDM_IT(i, A), HPDDM_TOL(tol, A), mu, mu, norm, rho + 2 * mu * mu - mu / (m[0] <= 1 ? mu : 1), 0, trash, (m[0] <= 1 ? mu : 1)));
#if defined(_KSPIMPL_H)
        A._ksp->rnorm = *std::max_element(reinterpret_cast<underlying_type<K>*>(trash), reinterpret_cast<underlying_type<K>*>(trash) + (m[0] <= 1 ? mu : 1));
        PetscCall(KSPLogResidualHistory(A._ksp, A._ksp->rnorm));
        PetscCall(KSPMonitor(A._ksp, HPDDM_IT(j, A), A._ksp->rnorm));
        PetscCall((*A._ksp->converged)(A._ksp, HPDDM_IT(j, A), A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP));
        if(A._ksp->reason)
            converged = true;
        else if(A._ksp->converged == KSPConvergedSkip)
            converged = false;
#endif
        if(converged)
            break;
        else if(++HPDDM_IT(i, A) <= HPDDM_MAX_IT(m[1], A)) {
            for(unsigned short nu = mu; nu > 0; --nu)
                std::copy_backward(rhs + (nu * (nu - 1)) / 2, rhs + (nu * (nu + 1)) / 2, rhs + nu * mu - (mu - nu));
            for(unsigned short i = 0; i < mu; ++i)
                for(unsigned short j = 0; j < i; ++j)
                    rhs[i + j * mu] = Wrapper<K>::conj(rhs[j + i * mu]);
            std::copy_n(rhs, mu * mu, rho + mu * mu);
            Lapack<K>::posv("U", &mu, &mu, rho, &mu, rhs, &mu, &info);
            if(info) {
                delete [] norm;
                delete [] trash;
                A.end(allocate);
#if defined(_KSPIMPL_H)
                A._ksp->reason = KSP_DIVERGED_BREAKDOWN;
#endif
                return HPDDM_RET(CG<excluded>(A, b, x, mu, comm));
            }
            if(!excluded && n) {
                Blas<K>::trmm("L", "U", "N", "N", &mu, &mu, &(Wrapper<K>::d__1), gamma, &mu, rhs, &mu);
                std::copy(p, r, trash);
                Blas<K>::gemm("N", "N", &n, &mu, &mu, &(Wrapper<K>::d__1), trash, &n, rhs, &mu, &(Wrapper<K>::d__1), p, &n);
            }
            if(QR<excluded>(id[1], n, mu, p, gamma, mu, d, trash, comm) != mu) {
                delete [] norm;
                delete [] trash;
                A.end(allocate);
#if defined(_KSPIMPL_H)
                A._ksp->reason = KSP_DIVERGED_BREAKDOWN;
#endif
                return HPDDM_RET(CG<excluded>(A, b, x, mu, comm));
            }
            std::copy_n(rho + mu * mu, mu * mu, rho);
        }
    }
#if !defined(_KSPIMPL_H)
    convergence<3>(id[0], HPDDM_IT(i, A), HPDDM_MAX_IT(m[1], A));
#endif
    delete [] norm;
    delete [] trash;
    A.end(allocate);
    return HPDDM_RET(std::min(HPDDM_IT(i, A), HPDDM_MAX_IT(m[1], A)));
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::BFBCG(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
#if !defined(_KSPIMPL_H)
    underlying_type<K> tol[2];
    unsigned short i, m[2];
    char id[2];
    {
        const std::string prefix = A.prefix();
        const Option& opt = *Option::get();
        if((hpddm_method_id<Operator>::value == 1 || hpddm_method_id<Operator>::value == 4) && (!opt.any_of(prefix + "schwarz_method", { HPDDM_SCHWARZ_METHOD_SORAS, HPDDM_SCHWARZ_METHOD_ASM, HPDDM_SCHWARZ_METHOD_NONE }) || opt.any_of(prefix + "schwarz_coarse_correction", { HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED })))
            return GMRES<excluded>(A, b, x, mu, comm);
        options<6>(A, tol, nullptr, m, id);
        if(opt.val<char>(prefix + "variant", HPDDM_VARIANT_LEFT) == HPDDM_VARIANT_FLEXIBLE)
            return CG<excluded>(A, b, x, mu, comm);
    }
#else
    PetscReal* tol = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->rcntl;
    unsigned short* m = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->scntl;
    char* id = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl;
#endif
    const int n = excluded ? 0 : A.getDof();
    const int dim = n * mu;
    K* const trash = new K[5 * dim + (mu * (3 * mu + 1)) / 2 + mu / m[0]];
    K* const q = trash + dim;
    K* const r = q + dim;
    K* const p = r + dim;
    K* const z = p + dim;
    K* const gamma = z + dim;
    const underlying_type<K>* const d = reinterpret_cast<const underlying_type<K>*>(A.getScaling());
    bool allocate = A.template start<excluded>(b, x, mu);
    int* const piv = new int[mu];
    int deflated = -1;
    int info;
    if(!excluded)
        HPDDM_CALL(A.GMV(x, trash, mu));
    std::copy_n(b, dim, r);
    Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), trash, &i__1, r, &i__1);
    HPDDM_CALL(A.template apply<excluded>(r, p, mu, trash));
    RRQR<excluded>(id[1], n, mu, p, gamma, tol[0], deflated, piv, d, trash, comm);
#if !defined(_KSPIMPL_H)
    diagonal<6>(id[0], gamma, mu, tol[0], piv);
#endif
    underlying_type<K>* const norm = new underlying_type<K>[mu];
    if(m[0] <= 1)
        for(unsigned short nu = 0; nu < mu; ++nu)
            norm[nu] = Blas<K>::nrm2(&(info = nu + 1), gamma + mu * nu, &i__1);
    else {
        std::fill_n(trash, m[0], K());
        for(unsigned short nu = 0; nu < m[0]; ++nu)
            Blas<K>::axpy(&(info = nu + 1), &(Wrapper<K>::d__1), gamma + mu * nu, &i__1, trash, &i__1);
        *norm = Blas<K>::nrm2(&(info = m[0]), trash, &i__1);
    }
    if(tol[0] > -0.9) {
        Lapack<K>::lapmt(&i__1, &n, &mu, x, &n, piv);
        Lapack<K>::lapmt(&i__1, &n, &mu, r, &n, piv);
        if(m[0] <= 1)
            Lapack<underlying_type<K>>::lapmt(&i__1, &i__1, &mu, norm, &i__1, piv);
    }
#if defined(_KSPIMPL_H)
    A._ksp->rnorm = *std::max_element(norm, norm + (m[0] <= 1 ? deflated : 1));
    PetscCall(KSPLogResidualHistory(A._ksp, A._ksp->rnorm));
    PetscCall(KSPMonitor(A._ksp, 0, A._ksp->rnorm));
    if(A._ksp->rnorm > 10.0 * std::numeric_limits<underlying_type<K>>::epsilon()) {
        if(tol[0] <= -0.9 && deflated != mu)
            A._ksp->reason = KSP_DIVERGED_BREAKDOWN;
        else
            PetscCall((*A._ksp->converged)(A._ksp, 0, A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP));
    }
    else
        A._ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
    if(A._ksp->reason) {
        delete [] norm;
        delete [] piv;
        delete [] trash;
        A.end(allocate);
        return 0;
    }
#endif
    HPDDM_IT(i, A) = (deflated != 0 ? 1 : 0);
    while(HPDDM_IT(i, A) <= HPDDM_MAX_IT(m[1], A) && deflated != 0) {
        if(!excluded)
            HPDDM_CALL(A.GMV(p, q, deflated));
        K* const alpha = gamma + (deflated * (deflated + 1)) / 2;
        if(!excluded && n) {
            Wrapper<K>::diag(n, d, p, trash, deflated);
            Blas<K>::gemmt("U", &(Wrapper<K>::transc), "N", &deflated, &n, &(Wrapper<K>::d__1), trash, &n, q, &n, &(Wrapper<K>::d__0), gamma, &deflated);
            for(unsigned short nu = 1; nu < deflated; ++nu)
                std::copy_n(gamma + nu * deflated, nu + 1, gamma + (nu * (nu + 1)) / 2);
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &deflated, &mu, &n, &(Wrapper<K>::d__1), trash, &n, r, &n, &(Wrapper<K>::d__0), alpha, &deflated);
        }
        else
            std::fill_n(gamma, (deflated * (deflated + 1)) / 2 + deflated * mu, K());
        MPI_Allreduce(MPI_IN_PLACE, gamma, (deflated * (deflated + 1)) / 2 + deflated * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
        Lapack<K>::pptrf("U", &deflated, gamma, &info);
        Lapack<K>::pptrs("U", &deflated, &mu, gamma, alpha, &deflated, &info);
        if(!excluded && n) {
            Blas<K>::gemm("N", "N", &n, &mu, &deflated, &(Wrapper<K>::d__1), p, &n, alpha, &deflated, &(Wrapper<K>::d__1), x, &n);
            Blas<K>::gemm("N", "N", &n, &mu, &deflated, &(Wrapper<K>::d__2), q, &n, alpha, &deflated, &(Wrapper<K>::d__1), r, &n);
        }
        HPDDM_CALL(A.template apply<excluded>(r, z, mu, trash));
        K* const res = alpha + deflated * mu;
        if(!excluded && n) {
            Wrapper<K>::diag(n, d, z, trash, mu);
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &deflated, &mu, &n, &(Wrapper<K>::d__1), q, &n, trash, &n, &(Wrapper<K>::d__0), alpha, &deflated);
            if(m[0] <= 1)
                for(unsigned short nu = 0; nu < mu; ++nu)
                    res[nu] = std::real(Blas<K>::dot(&n, z + n * nu, &i__1, trash + n * nu, &i__1));
            else {
                for(unsigned short nu = 1; nu < m[0]; ++nu)
                    Blas<K>::axpy(&n, &(Wrapper<K>::d__1), trash + nu * n, &i__1, trash, &i__1);
                std::copy_n(z, n, q);
                for(unsigned short nu = 1; nu < m[0]; ++nu)
                    Blas<K>::axpy(&n, &(Wrapper<K>::d__1), z + nu * n, &i__1, q, &i__1);
                res[0] = std::real(Blas<K>::dot(&n, trash, &i__1, q, &i__1));
            }
        }
        else
             std::fill_n(alpha, deflated * mu + mu / m[0], K());
        MPI_Allreduce(MPI_IN_PLACE, alpha, deflated * mu + mu / m[0], Wrapper<K>::mpi_type(), MPI_SUM, comm);
        bool converged = (mu == checkBlockConvergence<6>(id[0], HPDDM_IT(i, A), HPDDM_TOL(tol[1], A), mu, deflated, norm, res, 0, trash, m[0]));
#if defined(_KSPIMPL_H)
        A._ksp->rnorm = *std::max_element(reinterpret_cast<underlying_type<K>*>(trash), reinterpret_cast<underlying_type<K>*>(trash) + deflated);
        PetscCall(KSPLogResidualHistory(A._ksp, A._ksp->rnorm));
        PetscCall(KSPMonitor(A._ksp, HPDDM_IT(j, A), A._ksp->rnorm));
        PetscCall((*A._ksp->converged)(A._ksp, HPDDM_IT(j, A), A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP));
        if(A._ksp->reason)
            converged = true;
        else if(A._ksp->converged == KSPConvergedSkip)
            converged = false;
#endif
        if(converged)
            break;
        else if(++HPDDM_IT(i, A) <= HPDDM_MAX_IT(m[1], A)) {
            if(!excluded && n) {
                Lapack<K>::pptrs("U", &deflated, &mu, gamma, alpha, &deflated, &info);
                std::swap_ranges(p, p + dim, z);
                Blas<K>::gemm("N", "N", &n, &mu, &deflated, &(Wrapper<K>::d__2), z, &n, alpha, &deflated, &(Wrapper<K>::d__1), p, &n);
            }
            if(tol[0] > -0.9) {
                Lapack<K>::lapmt(&i__0, &n, &mu, x, &n, piv);
                Lapack<K>::lapmt(&i__0, &n, &mu, p, &n, piv);
                Lapack<K>::lapmt(&i__0, &n, &mu, r, &n, piv);
                if(m[0] <= 1)
                    Lapack<underlying_type<K>>::lapmt(&i__0, &i__1, &mu, norm, &i__1, piv);
            }
            RRQR<excluded>(id[1], n, mu, p, gamma, tol[0], deflated, piv, d, trash, comm);
            if(tol[0] > -0.9) {
                Lapack<K>::lapmt(&i__1, &n, &mu, x, &n, piv);
                Lapack<K>::lapmt(&i__1, &n, &mu, r, &n, piv);
                if(m[0] <= 1)
                    Lapack<underlying_type<K>>::lapmt(&i__1, &i__1, &mu, norm, &i__1, piv);
            }
        }
    }
    if(tol[0] > -0.9)
        Lapack<K>::lapmt(&i__0, &n, &mu, x, &n, piv);
#if !defined(_KSPIMPL_H)
    convergence<6>(id[0], HPDDM_IT(i, A), HPDDM_MAX_IT(m[1], A));
#endif
    delete [] norm;
    delete [] piv;
    delete [] trash;
    A.end(allocate);
    return HPDDM_RET(std::min(HPDDM_IT(i, A), HPDDM_MAX_IT(m[1], A)));
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::PCG(const Operator& A, const K* const f, K* const x, const MPI_Comm& comm) {
    underlying_type<K> tol;
    unsigned short it;
    char verbosity;
    options<8>(A, &tol, nullptr, &it, &verbosity);
    typedef typename std::conditional<std::is_pointer<typename std::remove_reference<decltype(*A.getScaling())>::type>::value, K**, K*>::type ptr_type;
    const int n = std::is_same<ptr_type, K*>::value ? A.getDof() : A.getMult();
    const int offset = std::is_same<ptr_type, K*>::value ? A.getEliminated() : 0;
    ptr_type storage[std::is_same<ptr_type, K*>::value ? 1 : 2];
    // storage[0] = r
    // storage[1] = lambda
    A.allocateArray(storage);
    auto m = A.getScaling();
    bool allocate = std::is_same<ptr_type, K*>::value ? A.template start<excluded>(f, x + offset, nullptr, storage[0]) : A.template start<excluded>(f, x, storage[1], storage[0]);
    std::vector<ptr_type> z;
    z.reserve(it);
    ptr_type zCurr;
    A.allocateSingle(zCurr);
    z.emplace_back(zCurr);
    if(!excluded)
        A.precond(storage[0], zCurr);                                                              //     z_0 = M r_0

    underlying_type<K> resInit;
    A.template computeDot<excluded>(&resInit, zCurr, zCurr, comm);
    resInit = std::sqrt(resInit);

    std::vector<ptr_type> p;
    p.reserve(it);
    ptr_type pCurr;
    A.allocateSingle(pCurr);
    p.emplace_back(pCurr);

    K* alpha = new K[excluded ? std::max(static_cast<unsigned short>(2), it) : 2 * it];
    underlying_type<K> resRel = std::numeric_limits<underlying_type<K>>::max();
    unsigned short i = 1;
    while(i <= it) {
        if(!excluded) {
            A.template project<excluded, 'N'>(zCurr, pCurr);                                       //     p_i = P z_i
            for(unsigned short k = 0; k < i - 1; ++k)
                alpha[it + k] = dot(&n, z[k], &i__1, pCurr, &i__1);
            MPI_Allreduce(MPI_IN_PLACE, alpha + it, i - 1, Wrapper<K>::mpi_type(), MPI_SUM, comm); // alpha_k = < z_k, p_i >
            for(unsigned short k = 0; k < i - 1; ++k) {
                alpha[it + k] /= -alpha[k];
                axpy(&n, alpha + it + k, p[k], &i__1, pCurr, &i__1);                               //     p_i = p_i - sum < z_k, p_i > / < z_k, p_k > p_k
            }
            A.apply(pCurr, zCurr);                                                                 //     z_i = F p_i

            A.allocateSingle(zCurr);
            if(std::is_same<ptr_type, K*>::value) {
                diag(n, m, pCurr, zCurr);
                alpha[i - 1] = dot(&n, z.back(), &i__1, zCurr, &i__1);
                alpha[i]     = dot(&n, storage[0], &i__1, zCurr, &i__1);
            }
            else {
                alpha[i - 1] = dot(&n, z.back(), &i__1, pCurr, &i__1);
                alpha[i]     = dot(&n, storage[0], &i__1, pCurr, &i__1);
            }
            MPI_Allreduce(MPI_IN_PLACE, alpha + i - 1, 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            alpha[it] = alpha[i] / alpha[i - 1];
            if(std::is_same<ptr_type, K*>::value)
                axpy(&n, alpha + it, pCurr, &i__1, x + offset, &i__1);
            else
                axpy(&n, alpha + it, pCurr, &i__1, storage[1], &i__1);                             // l_i + 1 = l_i + < r_i, p_i > / < z_i, p_i > p_i
            alpha[it] = -alpha[it];
            axpy(&n, alpha + it, z.back(), &i__1, storage[0], &i__1);                              // r_i + 1 = r_i - < r_i, p_i > / < z_i, p_i > z_i
            A.template project<excluded, 'T'>(storage[0]);                                         // r_i + 1 = P^T r_i + 1

            z.emplace_back(zCurr);
            A.precond(storage[0], zCurr);                                                          // z_i + 1 = M r_i
        }
        else {
            A.template project<excluded, 'N'>(zCurr, pCurr);
            std::fill_n(alpha, i - 1, K());
            MPI_Allreduce(MPI_IN_PLACE, alpha, i - 1, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            std::fill_n(alpha, 2, K());
            MPI_Allreduce(MPI_IN_PLACE, alpha, 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            A.template project<excluded, 'T'>(storage[0]);
        }
        A.template computeDot<excluded>(&resRel, zCurr, zCurr, comm);
        resRel = std::sqrt(resRel);
        if(verbosity > 2)
            std::cout << "PCG: " << std::setw(3) << i << " " << resRel << " " << resInit << " " << resRel / resInit << " < " << tol << std::endl;
        if(resRel / resInit <= tol)
            break;
        else
            ++i;
        if(!excluded) {
            A.allocateSingle(pCurr);
            p.emplace_back(pCurr);
            diag(n, m, z[i - 2]);
        }
    }
    convergence<7>(verbosity, i, it);
    if(std::is_same<ptr_type, K*>::value)
        A.template computeSolution<excluded>(f, x);
    else
        A.template computeSolution<excluded>(storage[1], x);
    delete [] alpha;
    for(auto zCurr : z)
        clean(zCurr);
    for(auto pCurr : p)
        clean(pCurr);
    clean(storage[0]);
    A.end(allocate);
    return std::min(i, it);
}
} // HPDDM
#endif // _HPDDM_CG_
