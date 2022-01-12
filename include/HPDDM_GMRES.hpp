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

#ifndef _HPDDM_GMRES_
#define _HPDDM_GMRES_

#include "HPDDM_iterative.hpp"

namespace HPDDM {
template<bool excluded, class Operator, class K>
inline int IterativeMethod::GMRES(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
#if !defined(_KSPIMPL_H)
    underlying_type<K> tol;
    unsigned short j, m[2];
    char id[3];
    options<0>(A, &tol, nullptr, m, id);
#else
    unsigned short* m = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->scntl;
    char* id = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl;
#endif
    int ierr;
    const int n = excluded ? 0 : A.getDof();
    K** const H = new K*[m[0] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 3 : 2) + 1];
    K** const v = H + m[0];
    K* const s = new K[mu * ((m[0] + 1) * (m[0] + 1) + n * (2 + m[0] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1)) + (!Wrapper<K>::is_complex ? m[0] + 1 : (m[0] + 2) / 2))];
    K* const Ax = s + mu * (m[0] + 1);
    *H = Ax + mu * n;
    for(unsigned short i = 1; i < m[0]; ++i)
        H[i] = *H + i * mu * (m[0] + 1);
    *v = *H + m[0] * mu * (m[0] + 1);
    for(unsigned short i = 1; i < m[0] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1; ++i)
        v[i] = *v + i * mu * n;
    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(*v + (m[0] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1) * mu * n);
    underlying_type<K>* const sn = norm + mu;
    const underlying_type<K>* const d = A.getScaling();
    short* const hasConverged = new short[mu];
    std::fill_n(hasConverged, mu, -m[0]);
    bool allocate;
    ierr = initializeNorm<excluded>(A, id[1], b, x, *v, n, Ax, norm, mu, 1, allocate);HPDDM_CHKERRQ(ierr);
    HPDDM_IT(j, A) = 1;
    while(HPDDM_IT(j, A) <= HPDDM_MAX_IT(m[1], A)) {
        if(!excluded) {
            ierr = A.GMV(x, id[1] == HPDDM_VARIANT_LEFT ? Ax : *v, mu);HPDDM_CHKERRQ(ierr);
        }
        Blas<K>::axpby(mu * n, 1.0, b, 1, -1.0, id[1] == HPDDM_VARIANT_LEFT ? Ax : *v, 1);
        if(id[1] == HPDDM_VARIANT_LEFT) {
            ierr = A.template apply<excluded>(Ax, *v, mu);HPDDM_CHKERRQ(ierr);
        }
        if(d)
            for(unsigned short nu = 0; nu < mu; ++nu) {
                sn[nu] = 0.0;
                for(int j = 0; j < n; ++j)
                    sn[nu] += d[j] * std::norm(v[0][nu * n + j]);
            }
        else
            for(unsigned short nu = 0; nu < mu; ++nu)
                sn[nu] = std::real(Blas<K>::dot(&n, *v + nu * n, &i__1, *v + nu * n, &i__1));
        if(HPDDM_IT(j, A) == 1) {
            MPI_Allreduce(MPI_IN_PLACE, norm, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                norm[nu] = std::sqrt(norm[nu]);
                if(norm[nu] < HPDDM_EPS)
                    norm[nu] = 1.0;
                if(sn[nu] < std::pow(std::numeric_limits<underlying_type<K>>::epsilon(), 2.0)) {
                    HPDDM_IT(j, A) = 0;
                    break;
                }
            }
        }
        else
            MPI_Allreduce(MPI_IN_PLACE, sn, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
        if(HPDDM_IT(j, A) == 0) {
#if HPDDM_PETSC
            ierr = KSPLogResidualHistory(A._ksp, underlying_type<K>());CHKERRQ(ierr);
            ierr = KSPMonitor(A._ksp, 0, underlying_type<K>());CHKERRQ(ierr);
            A._ksp->reason = KSP_DIVERGED_BREAKDOWN;
#endif
            std::fill_n(hasConverged, mu, 0);
            break;
        }
        for(unsigned short nu = 0; nu < mu; ++nu) {
            if(hasConverged[nu] > 0)
                hasConverged[nu] = 0;
            s[nu] = std::sqrt(sn[nu]);
            std::for_each(*v + nu * n, *v + (nu + 1) * n, [&](K& y) { y /= s[nu]; });
        }
#if HPDDM_PETSC
        if(HPDDM_IT(j, A) == 1) {
            A._ksp->rnorm = std::abs(*std::max_element(s, s + mu, [](const K& lhs, const K& rhs) { return std::abs(lhs) < std::abs(rhs); }));
            ierr = KSPLogResidualHistory(A._ksp, A._ksp->rnorm);CHKERRQ(ierr);
            ierr = KSPMonitor(A._ksp, 0, A._ksp->rnorm);CHKERRQ(ierr);
            ierr = (*A._ksp->converged)(A._ksp, 0, A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP);CHKERRQ(ierr);
            if(A._ksp->reason) {
                delete [] hasConverged;
                A.end(allocate);
                delete [] s;
                delete [] H;
                return 0;
            }
        }
#endif
        unsigned short i = 0;
        while(i < m[0] && HPDDM_IT(j, A) <= HPDDM_MAX_IT(m[1], A)) {
            if(id[1] == HPDDM_VARIANT_LEFT) {
                if(!excluded) {
                    ierr = A.GMV(v[i], Ax, mu);HPDDM_CHKERRQ(ierr);
                }
                ierr = A.template apply<excluded>(Ax, v[i + 1], mu);HPDDM_CHKERRQ(ierr);
            }
            else {
                ierr = A.template apply<excluded>(v[i], id[1] == HPDDM_VARIANT_FLEXIBLE ? v[i + m[0] + 1] : Ax, mu, v[i + 1]);HPDDM_CHKERRQ(ierr);
                if(!excluded) {
                    ierr = A.GMV(id[1] == HPDDM_VARIANT_FLEXIBLE ? v[i + m[0] + 1] : Ax, v[i + 1], mu);HPDDM_CHKERRQ(ierr);
                }
            }
            Arnoldi<excluded>(id[2], m[0], H, v, s, sn, n, i++, mu, d, Ax, comm);
            checkConvergence<0>(id[0], HPDDM_IT(j, A), i, HPDDM_TOL(tol, A), mu, norm, s + i * mu, hasConverged, m[0]);
#if HPDDM_PETSC
            A._ksp->rnorm = std::abs(*std::max_element(s + i * mu, s + (i + 1) * mu, [](const K& lhs, const K& rhs) { return std::abs(lhs) < std::abs(rhs); }));
            ierr = KSPLogResidualHistory(A._ksp, A._ksp->rnorm);CHKERRQ(ierr);
            ierr = KSPMonitor(A._ksp, HPDDM_IT(j, A), A._ksp->rnorm);CHKERRQ(ierr);
            ierr = (*A._ksp->converged)(A._ksp, HPDDM_IT(j, A), A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP);CHKERRQ(ierr);
            if(A._ksp->reason)
                std::for_each(hasConverged, hasConverged + mu, [&](short& c) { if(c == -m[0]) c = i; });
            else if(A._ksp->converged == KSPConvergedSkip)
                std::fill_n(hasConverged, mu, -m[0]);
#endif
            if(std::find(hasConverged, hasConverged + mu, -m[0]) == hasConverged + mu) {
                i = 0;
                break;
            }
            ++HPDDM_IT(j, A);
        }
        if(HPDDM_IT(j, A) != HPDDM_MAX_IT(m[1], A) + 1 && i == m[0]) {
            ierr = updateSol<excluded>(A, id[1], n, x, H, s, v + (id[1] == HPDDM_VARIANT_FLEXIBLE ? m[0] + 1 : 0), hasConverged, mu, Ax);HPDDM_CHKERRQ(ierr);
#if !defined(_KSPIMPL_H)
            if(id[0] > 1)
                std::cout << "GMRES restart(" << m[0] << ")" << std::endl;
#endif
        }
        else
            break;
    }
    if(!excluded && HPDDM_IT(j, A) == HPDDM_MAX_IT(m[1], A) + 1 && m[0] > 0) {
        const int rem = HPDDM_MAX_IT(m[1], A) % m[0];
        std::for_each(hasConverged, hasConverged + mu, [&rem](short& d) { if(d < 0) d = rem > 0 ? rem : -d; });
    }
    ierr = updateSol<excluded>(A, id[1], n, x, H, s, v + (id[1] == HPDDM_VARIANT_FLEXIBLE ? m[0] + 1 : 0), hasConverged, mu, Ax);HPDDM_CHKERRQ(ierr);
    convergence<0>(id[0], HPDDM_IT(j, A), HPDDM_MAX_IT(m[1], A));
    delete [] hasConverged;
    A.end(allocate);
    delete [] s;
    delete [] H;
    return HPDDM_RET(std::min(HPDDM_IT(j, A), HPDDM_MAX_IT(m[1], A)));
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::BGMRES(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
#if !defined(_KSPIMPL_H)
    underlying_type<K> tol[2];
    unsigned short j, m[3];
    char id[3];
    options<1>(A, tol, nullptr, m, id);
#else
    underlying_type<K>* tol = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->rcntl;
    unsigned short* m = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->scntl;
    char* id = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl;
#endif
    int ierr;
    const int n = excluded ? 0 : A.getDof();
    K** const H = new K*[m[0] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 3 : 2) + 1];
    K** const v = H + m[0];
    int ldh = mu * (m[0] + 1);
    int info;
    int N = 2 * mu;
    const underlying_type<K>* const d = A.getScaling();
    int lwork = mu * (d ? n + ldh : std::max(n, ldh));
    *H = new K[lwork + mu * ((m[0] + 1) * ldh + n * (m[0] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1) + 2 * m[0]) + (Wrapper<K>::is_complex ? (mu + 1) / 2 : mu)];
    *v = *H + m[0] * mu * ldh;
    K* const s = *v + mu * n * (m[0] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1);
    K* const tau = s + mu * ldh;
    K* const Ax = tau + m[0] * N;
    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(Ax + lwork);
    bool allocate;
    ierr = initializeNorm<excluded>(A, id[1], b, x, *v, n, Ax, norm, mu, m[1], allocate);HPDDM_CHKERRQ(ierr);
    MPI_Allreduce(MPI_IN_PLACE, norm, mu / m[1], Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
    for(unsigned short nu = 0; nu < mu / m[1]; ++nu) {
        norm[nu] = std::sqrt(norm[nu]);
        if(norm[nu] < HPDDM_EPS)
            norm[nu] = 1.0;
    }
    HPDDM_IT(j, A) = 1;
    short dim = mu * m[0];
    int* const piv = new int[mu];
    int deflated = -1;
    while(HPDDM_IT(j, A) <= HPDDM_MAX_IT(m[2], A)) {
        if(!excluded) {
            ierr = A.GMV(x, id[1] == HPDDM_VARIANT_LEFT ? Ax : *v, mu);HPDDM_CHKERRQ(ierr);
        }
        Blas<K>::axpby(mu * n, 1.0, b, 1, -1.0, id[1] == HPDDM_VARIANT_LEFT ? Ax : *v, 1);
        if(id[1] == HPDDM_VARIANT_LEFT) {
            ierr = A.template apply<excluded>(Ax, *v, mu);HPDDM_CHKERRQ(ierr);
        }
        RRQR<excluded>((id[2] >> 2) & 7, n, mu, *v, s, tol[0], N, piv, d, Ax, comm);
#if !defined(_KSPIMPL_H)
        diagonal<1>(id[0], s, mu, tol[0], piv);
#endif
        if(tol[0] > -0.9 && m[1] <= 1)
            Lapack<underlying_type<K>>::lapmt(&i__1, &i__1, &mu, norm, &i__1, piv);
        if(N == 0) {
#if HPDDM_PETSC
            ierr = KSPLogResidualHistory(A._ksp, underlying_type<K>());CHKERRQ(ierr);
            ierr = KSPMonitor(A._ksp, 0, underlying_type<K>());CHKERRQ(ierr);
            A._ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
#endif
            HPDDM_IT(j, A) = 0;
            break;
        }
#if HPDDM_PETSC
        else if(HPDDM_IT(j, A) == 1) {
            A._ksp->rnorm = std::abs(s[0]);
            for(unsigned short nu = 1; nu < mu; ++nu)
                A._ksp->rnorm = std::max(A._ksp->rnorm, std::abs(s[nu * (mu + 1)]));
            ierr = KSPLogResidualHistory(A._ksp, A._ksp->rnorm);CHKERRQ(ierr);
            ierr = KSPMonitor(A._ksp, 0, A._ksp->rnorm);CHKERRQ(ierr);
            if(tol[0] <= -0.9 && N != mu)
                A._ksp->reason = KSP_DIVERGED_BREAKDOWN;
            else {
                ierr = (*A._ksp->converged)(A._ksp, 0, A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP);CHKERRQ(ierr);
            }
            if(A._ksp->reason) {
                HPDDM_IT(j, A) = 0;
                break;
            }
        }
#endif
        if(N != mu) {
            int nrhs = mu - N;
            Lapack<K>::trtrs("U", "N", "N", &N, &nrhs, s, &mu, s + N * mu, &mu, &info);
#if HPDDM_PETSC
            ierr = PetscInfo(A._ksp, "HPDDM: Deflating %d out of %d RHS\n", mu - N, mu);CHKERRQ(ierr);
#endif
        }
        if(N != deflated) {
            deflated = N;
            dim = deflated * (HPDDM_IT(j, A) - 1 + m[0] > HPDDM_MAX_IT(m[2], A) ? HPDDM_MAX_IT(m[2], A) - HPDDM_IT(j, A) + 1 : m[0]);
            ldh = deflated * (m[0] + 1);
            for(unsigned short i = 1; i < m[0]; ++i)
                H[i] = *H + i * deflated * ldh;
            for(unsigned short i = 1; i < m[0] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1; ++i)
                v[i] = *v + i * deflated * n;
        }
        N *= 2;
        std::fill_n(tau, m[0] * N, K());
        Wrapper<K>::template imatcopy<'N'>(mu, mu, s, mu, ldh);
        for(unsigned short nu = 0; nu < deflated; ++nu)
            std::fill(s + nu * (ldh + 1) + 1, s + (nu + 1) * ldh, K());
        std::fill(*H, *v, K());
        unsigned short i = 0;
        while(i < m[0] && HPDDM_IT(j, A) <= HPDDM_MAX_IT(m[2], A)) {
            if(id[1] == HPDDM_VARIANT_LEFT) {
                if(!excluded) {
                    ierr = A.GMV(v[i], Ax, deflated);HPDDM_CHKERRQ(ierr);
                }
                ierr = A.template apply<excluded>(Ax, v[i + 1], deflated);HPDDM_CHKERRQ(ierr);
            }
            else {
                ierr = A.template apply<excluded>(v[i], id[1] == HPDDM_VARIANT_FLEXIBLE ? v[i + m[0] + 1] : Ax, deflated, v[i + 1]);HPDDM_CHKERRQ(ierr);
                if(!excluded) {
                    ierr = A.GMV(id[1] == HPDDM_VARIANT_FLEXIBLE ? v[i + m[0] + 1] : Ax, v[i + 1], deflated);HPDDM_CHKERRQ(ierr);
                }
            }
            if(BlockArnoldi<excluded>(id[2], m[0], H, v, tau, s, lwork, n, i++, deflated, d, Ax, comm)) {
                dim = deflated * (i - 1);
                i = HPDDM_IT(j, A) = 0;
                break;
            }
            bool converged = (mu == checkBlockConvergence<1>(id[0], HPDDM_IT(j, A), HPDDM_TOL(tol[1], A), mu, deflated, norm, s + deflated * i, ldh, Ax, m[1]));
#if HPDDM_PETSC
            A._ksp->rnorm = *std::max_element(reinterpret_cast<underlying_type<K>*>(Ax), reinterpret_cast<underlying_type<K>*>(Ax) + deflated);
            ierr = KSPLogResidualHistory(A._ksp, A._ksp->rnorm);CHKERRQ(ierr);
            ierr = KSPMonitor(A._ksp, HPDDM_IT(j, A), A._ksp->rnorm);CHKERRQ(ierr);
            ierr = (*A._ksp->converged)(A._ksp, HPDDM_IT(j, A), A._ksp->rnorm, &A._ksp->reason, A._ksp->cnvP);CHKERRQ(ierr);
            if(A._ksp->reason)
                converged = true;
            else if(A._ksp->converged == KSPConvergedSkip)
                converged = false;
#endif
            if(converged) {
                dim = deflated * i;
                i = 0;
                break;
            }
            ++HPDDM_IT(j, A);
        }
        if(tol[0] > -0.9)
            Lapack<K>::lapmt(&i__1, &n, &mu, x, &n, piv);
        if(HPDDM_IT(j, A) != HPDDM_MAX_IT(m[2], A) + 1 && i == m[0]) {
            ierr = updateSol<excluded>(A, id[1], n, x, H, s, v + (id[1] == HPDDM_VARIANT_FLEXIBLE ? m[0] + 1 : 0), &dim, mu, Ax, deflated);HPDDM_CHKERRQ(ierr);
            if(tol[0] > -0.9) {
                Lapack<K>::lapmt(&i__0, &n, &mu, x, &n, piv);
                if(m[1] <= 1)
                    Lapack<underlying_type<K>>::lapmt(&i__0, &i__1, &mu, norm, &i__1, piv);
            }
#if !defined(_KSPIMPL_H)
            if(id[0] > 1)
                std::cout << "BGMRES restart(" << m[0] << ")" << std::endl;
#endif
        }
        else
            break;
    }
    if(!excluded && HPDDM_IT(j, A) != 0 && HPDDM_IT(j, A) == HPDDM_MAX_IT(m[2], A) + 1 && m[0] > 0) {
        const int rem = HPDDM_MAX_IT(m[2], A) % m[0];
        if(rem != 0)
            dim = deflated * rem;
    }
    if(HPDDM_IT(j, A) != 0 && deflated != -1) {
        ierr = updateSol<excluded>(A, id[1], n, x, H, s, v + (id[1] == HPDDM_VARIANT_FLEXIBLE ? m[0] + 1 : 0), &dim, mu, Ax, deflated);HPDDM_CHKERRQ(ierr);
        if(tol[0] > -0.9)
            Lapack<K>::lapmt(&i__0, &n, &mu, x, &n, piv);
    }
    delete [] piv;
    A.end(allocate);
    delete [] *H;
    delete [] H;
    if(HPDDM_IT(j, A) != 0 || deflated == -1) {
#if !defined(_KSPIMPL_H)
        convergence<1>(id[0], HPDDM_IT(j, A), HPDDM_MAX_IT(m[2], A));
#endif
        return HPDDM_RET(std::min(HPDDM_IT(j, A), HPDDM_MAX_IT(m[2], A)));
    }
    else {
#if defined(_KSPIMPL_H)
        A._ksp->reason = KSP_DIVERGED_BREAKDOWN;
#endif
        return HPDDM_RET(GMRES<excluded>(A, b, x, mu, comm));
    }
}
} // HPDDM
#endif // _HPDDM_GMRES_
