 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2016-01-06

   Copyright (C) 2016-     Centre National de la Recherche Scientifique

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

#ifndef _HPDDM_GCRODR_
#define _HPDDM_GCRODR_

#if defined(PETSC_HAVE_SLEPC) && defined(PETSC_USE_SHARED_LIBRARIES)
static PetscErrorCode (*loadedKSPSym)(const char*, const MPI_Comm&, PetscMPIInt, PetscInt, PetscScalar*, int, PetscScalar*, int, PetscInt, PetscScalar*) = nullptr;
#endif

#include "HPDDM_GMRES.hpp"

namespace HPDDM {
template<class K>
inline void selectNu(unsigned short target, std::vector<std::pair<unsigned short, std::complex<underlying_type<K>>>>& q, unsigned short n, const K* const alphar, const K* const alphai, const K* const beta = nullptr) {
    for(unsigned short i = 0; i < n; ++i) {
        std::complex<underlying_type<K>> tmp(Wrapper<K>::is_complex ? alphar[i] : std::complex<underlying_type<K>>(std::real(alphar[i]), std::real(alphai[i])));
        if(beta)
             tmp /= beta[i];
        q.emplace_back(i, tmp);
    }
    using type = typename std::vector<std::pair<unsigned short, std::complex<underlying_type<K>>>>::const_reference;
    switch(target) {
        case HPDDM_RECYCLE_TARGET_LM: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return std::norm(lhs.second) > std::norm(rhs.second); }); break;
        case HPDDM_RECYCLE_TARGET_SR: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return std::real(lhs.second) < std::real(rhs.second); }); break;
        case HPDDM_RECYCLE_TARGET_LR: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return std::real(lhs.second) > std::real(rhs.second); }); break;
        case HPDDM_RECYCLE_TARGET_SI: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return std::imag(lhs.second) < std::imag(rhs.second); }); break;
        case HPDDM_RECYCLE_TARGET_LI: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return std::imag(lhs.second) > std::imag(rhs.second); }); break;
        default:                      std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return std::norm(lhs.second) < std::norm(rhs.second); });
    }
}

template<bool excluded, class Operator, class K>
inline int IterativeMethod::GCRODR(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
#if !defined(_KSPIMPL_H)
    underlying_type<K> tol;
    int k;
    unsigned short m[2];
    char id[5];
    options<4>(A, &tol, &k, m, id);
#else
    underlying_type<K> tol = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->rcntl[0];
    int k = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->icntl[0];
    unsigned short* m = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->scntl;
    char* id = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl;
#endif
    if(k <= 0) {
        if(id[0])
            std::cout << "WARNING -- please choose a positive number of Ritz vectors to compute, now switching to GMRES" << std::endl;
        return GMRES<excluded>(A, b, x, mu, comm);
    }
    int ierr;
    const int n = excluded ? 0 : A.getDof();
    const int ldh = mu * (m[1] + 1);
    K** const H = new K*[m[1] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 4 : 3) + 1];
    K** const save = H + m[1];
    *save = new K[ldh * m[1]];
    K** const v = save + m[1];
    const underlying_type<K>* const d = A.getScaling();
    const int ldv = mu * n;
    K* U = A.storage(), *C = nullptr;
    if(U) {
        std::pair<unsigned short, unsigned short> storage = A.k();
        k = (storage.first >= mu ? storage.second : (storage.second * storage.first) / mu);
        C = U + storage.first * storage.second * n;
    }
    K* const s = new K[mu * ((m[1] + 1) * (m[1] + 1) + n * ((id[1] == HPDDM_VARIANT_RIGHT ? 3 : 2) + m[1] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1)) + (!Wrapper<K>::is_complex ? m[1] + 1 : (m[1] + 2) / 2)) + (d && U && id[1] == HPDDM_VARIANT_RIGHT && id[4] / 4 == 0 ? n * std::max(k - mu * (m[1] - k + 2), 0) : 0)];
    *H = s + ldh;
    for(unsigned short i = 1; i < m[1]; ++i) {
        H[i] = *H + i * ldh;
        save[i] = *save + i * ldh;
    }
    *v = *H + m[1] * ldh;
    for(unsigned short i = 1; i < m[1] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1; ++i)
        v[i] = *v + i * ldv;
    K* const Ax = *v + (m[1] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1) * ldv;
    short* const hasConverged = new short[mu];
    std::fill_n(hasConverged, mu, -m[1]);
    int info;
    unsigned short j = 1;
    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(Ax + (id[1] == HPDDM_VARIANT_RIGHT ? 2 : 1) * ldv + (d && U && id[1] == HPDDM_VARIANT_RIGHT && id[4] / 4 == 0 ? n * std::max(k - mu * (m[1] - k + 2), 0) : 0));
    underlying_type<K>* const sn = norm + mu;
    ierr = initializeNorm<excluded>(A, id[1], b, x, *v, n, Ax, norm, mu, 1);
    const bool allocate = static_cast<bool>(ierr);
    if(ierr < 0)
        return ierr;
    while(j <= m[0]) {
        unsigned short i = (U ? k : 0);
        if(!excluded) {
            ierr = A.GMV(x, id[1] == HPDDM_VARIANT_LEFT ? Ax : v[i], mu);HPDDM_CHKERRQ(ierr)
            Blas<K>::axpby(ldv, 1.0, b, 1, -1.0, id[1] == HPDDM_VARIANT_LEFT ? Ax : v[i], 1);
        }
        if(id[1] == HPDDM_VARIANT_LEFT) {
            ierr = A.template apply<excluded>(Ax, v[i], mu);HPDDM_CHKERRQ(ierr)
        }
        if(j == 1 && U) {
            K* pt;
            if(id[1] == HPDDM_VARIANT_RIGHT) {
                pt = *v;
                if(id[4] / 4 == 0) {
                    ierr = A.template apply<excluded>(U, pt, mu * k, C);HPDDM_CHKERRQ(ierr)
                }
            }
            else
                pt = U;
            if(id[4] / 4 == 0) {
                if(id[1] == HPDDM_VARIANT_LEFT) {
                    if(!excluded) {
                        ierr = A.GMV(pt, *v, mu * k);HPDDM_CHKERRQ(ierr)
                    }
                    ierr = A.template apply<excluded>(*v, C, mu * k);HPDDM_CHKERRQ(ierr)
                }
                else if(!excluded) {
                    ierr = A.GMV(pt, C, mu * k);HPDDM_CHKERRQ(ierr)
                }
                QR<excluded>((id[2] >> 2) & 7, n, k, C, *save, k, d, *v + (id[1] == HPDDM_VARIANT_RIGHT ? ldv * (i + 1) : 0), comm, true, mu);
                if(!excluded && n) {
                    if(id[1] == HPDDM_VARIANT_RIGHT)
                        for(unsigned short nu = 0; nu < mu; ++nu)
                            Blas<K>::trsm("R", "U", "N", "N", &n, &k, &(Wrapper<K>::d__1), *save + nu * k * k, &k, pt + nu * n, &ldv);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::trsm("R", "U", "N", "N", &n, &k, &(Wrapper<K>::d__1), *save + nu * k * k, &k, U + nu * n, &ldv);
                }
            }
#ifdef PETSCHPDDM_H
            ierr = PetscLogEventBegin(KSP_GMRESOrthogonalization, A._ksp, 0, 0, 0);HPDDM_CHKERRQ(ierr)
#endif
            orthogonalization<excluded>(id[2] & 3, n, k, mu, C, v[i], H[i], d, Ax, comm);
#ifdef PETSCHPDDM_H
            ierr = PetscLogEventEnd(KSP_GMRESOrthogonalization, A._ksp, 0, 0, 0);HPDDM_CHKERRQ(ierr)
#endif
            if(id[1] != 1 || id[4] / 4 == 0) {
                if(!excluded && n)
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::gemv("N", &n, &k, &(Wrapper<K>::d__1), pt + nu * n, &ldv, H[i] + nu, &mu, &(Wrapper<K>::d__1), x + nu * n, &i__1);
            }
            else {
                if(!excluded && n)
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::gemv("N", &n, &k, &(Wrapper<K>::d__1), U + nu * n, &ldv, H[i] + nu, &mu, &(Wrapper<K>::d__0), *v + nu * n, &i__1);
                ierr = A.template apply<excluded>(*v, Ax, mu);HPDDM_CHKERRQ(ierr)
                int tmp = mu * n;
                Blas<K>::axpy(&tmp, &(Wrapper<K>::d__1), Ax, &i__1, x, &i__1);
            }
            std::copy_n(C, k * ldv, *v);
        }
        if(d)
            for(unsigned short nu = 0; nu < mu; ++nu) {
                sn[nu] = 0.0;
                for(int j = 0; j < n; ++j)
                    sn[nu] += d[j] * std::norm(v[i][nu * n + j]);
            }
        else
            for(unsigned short nu = 0; nu < mu; ++nu)
                sn[nu] = std::real(Blas<K>::dot(&n, v[i] + nu * n, &i__1, v[i] + nu * n, &i__1));
        if(j == 1) {
            MPI_Allreduce(MPI_IN_PLACE, norm, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                norm[nu] = std::sqrt(norm[nu]);
                if(norm[nu] < HPDDM_EPS)
                    norm[nu] = 1.0;
                if(100 * sn[nu] < std::numeric_limits<underlying_type<K>>::epsilon()) {
                    j = 0;
                    break;
                }
            }
            if(j == 0) {
#if HPDDM_PETSC
                KSPMonitor(A._ksp, 0, underlying_type<K>());
#endif
                std::fill_n(hasConverged, mu, 0);
                break;
            }
        }
        else
            MPI_Allreduce(MPI_IN_PLACE, sn, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
        for(unsigned short nu = 0; nu < mu; ++nu) {
            if(hasConverged[nu] > 0)
                hasConverged[nu] = 0;
            s[mu * i + nu] = std::sqrt(sn[nu]);
            if(U) {
                sn[mu * i + nu] = sn[nu];
                sn[nu] = std::real(s[mu * i + nu]);
            }
            std::for_each(v[i] + nu * n, v[i] + (nu + 1) * n, [&](K& y) { y /= s[mu * i + nu]; });
        }
#if HPDDM_PETSC
        if(j == 1)
            KSPMonitor(A._ksp, 0, std::abs(*std::max_element(s + i * mu, s + (i + 1) * mu, [](const K& lhs, const K& rhs) { return std::abs(lhs) < std::abs(rhs); })));
#endif
        while(i < m[1] && j <= m[0]) {
            if(id[1] == HPDDM_VARIANT_LEFT) {
                if(!excluded) {
                    ierr = A.GMV(v[i], Ax, mu);HPDDM_CHKERRQ(ierr)
                }
                ierr = A.template apply<excluded>(Ax, v[i + 1], mu);HPDDM_CHKERRQ(ierr)
            }
            else {
                ierr = A.template apply<excluded>(v[i], id[1] == HPDDM_VARIANT_FLEXIBLE ? v[i + m[1] + 1] : Ax, mu, v[i + 1]);HPDDM_CHKERRQ(ierr)
                if(!excluded) {
                    ierr = A.GMV(id[1] == HPDDM_VARIANT_FLEXIBLE ? v[i + m[1] + 1] : Ax, v[i + 1], mu);HPDDM_CHKERRQ(ierr)
                }
            }
            if(U) {
#ifdef PETSCHPDDM_H
                ierr = PetscLogEventBegin(KSP_GMRESOrthogonalization, A._ksp, 0, 0, 0);HPDDM_CHKERRQ(ierr)
#endif
                orthogonalization<excluded>(id[2] & 3, n, k, mu, C, v[i + 1], H[i], d, Ax, comm);
#ifdef PETSCHPDDM_H
                ierr = PetscLogEventEnd(KSP_GMRESOrthogonalization, A._ksp, 0, 0, 0);HPDDM_CHKERRQ(ierr)
#endif
            }
            Arnoldi<excluded>(id[2], m[1], H, v, s, sn, n, i++, mu, d, Ax, comm, save, U ? k : 0);
            checkConvergence<4>(id[0], j, i, tol, mu, norm, s + i * mu, hasConverged, m[1]);
#if HPDDM_PETSC
            KSPMonitor(A._ksp, j, std::abs(*std::max_element(s + i * mu, s + (i + 1) * mu, [](const K& lhs, const K& rhs) { return std::abs(lhs) < std::abs(rhs); })));
#endif
            if(std::find(hasConverged, hasConverged + mu, -m[1]) == hasConverged + mu) {
                i += (U ? m[1] - k : m[1]);
                break;
            }
            ++j;
        }
        bool converged;
        if(j != m[0] + 1 && i == m[1]) {
            converged = false;
            if(id[0] > 1)
                std::cout << "GCRODR restart(" << m[1] << ", " << k << ")" << std::endl;
        }
        else {
            converged = true;
            if(!excluded && j == m[0] + 1) {
                int rem = (U ? (m[0] - m[1]) % (m[1] - k) : m[0] % m[1]);
                if(rem) {
                    if(U)
                        rem += k;
                    std::for_each(hasConverged, hasConverged + mu, [&](short& dim) { if(dim < 0) dim = rem; });
                }
            }
        }
        ierr = updateSolRecycling<excluded>(A, id[1], n, x, H, s, v, sn, C, U, hasConverged, k, mu, Ax, comm);HPDDM_CHKERRQ(ierr)
        if(i == m[1]) {
            if(U)
                i -= k;
            for(unsigned short nu = 0; nu < mu; ++nu)
                std::for_each(v[m[1]] + nu * n, v[m[1]] + (nu + 1) * n, [&](K& y) { y /= save[i - 1][i + nu * (m[1] + 1)]; });
        }
#if !HPDDM_PETSC
        if(id[4] / 4 <= 1) {
#else
        {
#endif
            if(!U) {
                int dim = std::abs(*std::min_element(hasConverged, hasConverged + mu, [](const short& lhs, const short& rhs) { return lhs == 0 ? false : rhs == 0 ? true : lhs < rhs; }));
                if(j < k || dim < k)
                    k = dim;
                U = const_cast<Operator&>(A).allocate(n, mu, k);
                C = U + k * ldv;
                if(!excluded && n) {
                    std::fill_n(s, dim * mu, K());
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        K h = H[dim - 1][(m[1] + 1) * nu + dim] / H[dim - 1][(m[1] + 1) * nu + dim - 1];
                        for(i = dim; i-- > 1; ) {
                            s[i + dim * nu] = H[i - 1][(m[1] + 1) * nu + i] * h;
                            h *= -sn[(i - 1) * mu + nu];
                        }
                        s[dim * nu] = h;
                        for(i = 0; i < dim; ++i) {
                            std::fill_n(save[i] + i + 2 + nu * (m[1] + 1), m[1] - i - 1, K());
                            std::copy_n(save[i] + nu * (m[1] + 1), m[1] + 1, H[i] + nu * (m[1] + 1));
                        }
                        h = save[dim - 1][dim + nu * (m[1] + 1)] * save[dim - 1][dim + nu * (m[1] + 1)];
                        Blas<K>::axpy(&dim, &h, s + dim * nu, &i__1, H[dim - 1] + nu * (m[1] + 1), &i__1);
                        int* select = new int[dim]();
                        int row = dim + 1;
                        int lwork = -1;
                        Lapack<K>::hseqr("E", "N", &dim, &i__1, &dim, nullptr, &ldh, nullptr, nullptr, nullptr, &i__1, &h, &lwork, &info);
                        lwork = std::max(static_cast<int>(std::real(h)), Wrapper<K>::is_complex ? dim * dim : (dim * (dim + 2)));
                        *select = -1;
                        Lapack<K>::geqrf(&row, &k, nullptr, &ldh, nullptr, &h, select, &info);
                        lwork = std::max(static_cast<int>(std::real(h)), lwork);
                        Lapack<K>::mqr("R", "N", &n, &row, &k, nullptr, &ldh, nullptr, nullptr, &ldv, &h, select, &info);
                        *select = 0;
                        lwork = std::max(static_cast<int>(std::real(h)), lwork);
                        K* work = new K[lwork];
                        K* w = new K[Wrapper<K>::is_complex ? dim : (2 * dim)];
                        K* backup = new K[dim * dim]();
                        Wrapper<K>::template omatcopy<'N'>(dim, dim, *H + nu * (m[1] + 1), ldh, backup, dim);
                        Lapack<K>::hseqr("E", "N", &dim, &i__1, &dim, backup, &dim, w, w + dim, nullptr, &i__1, work, &lwork, &info);
                        delete [] backup;
                        std::vector<std::pair<unsigned short, std::complex<underlying_type<K>>>> q;
                        q.reserve(dim);
                        selectNu(id[3], q, dim, w, w + dim);
                        q.resize(k);
                        int mm = Wrapper<K>::is_complex ? k : 0;
                        for(typename decltype(q)::const_iterator it = q.cbegin(); it < q.cend(); ++it) {
                            if(Wrapper<K>::is_complex)
                                select[it->first] = 1;
                            else {
                                if(std::abs(w[dim + it->first]) < HPDDM_EPS) {
                                    select[it->first] = 1;
                                    ++mm;
                                }
                                else if(mm < k + 1) {
                                    select[it->first] = 1;
                                    mm += 2;
                                    ++it;
                                }
                                else
                                    break;
                            }
                        }
                        decltype(q)().swap(q);
                        underlying_type<K>* rwork = Wrapper<K>::is_complex ? new underlying_type<K>[dim] : nullptr;
                        K* vr = new K[mm * dim];
                        int* ifailr = new int[mm];
                        int col;
                        Lapack<K>::hsein("R", "Q", "N", select, &dim, *H + nu * (m[1] + 1), &ldh, w, w + dim, nullptr, &i__1, vr, &dim, &mm, &col, work, rwork, nullptr, ifailr, &info);
                        delete [] ifailr;
                        delete [] select;
                        delete [] rwork;
                        delete [] w;
                        Blas<K>::gemm("N", "N", &n, &k, &dim, &(Wrapper<K>::d__1), v[id[1] == HPDDM_VARIANT_FLEXIBLE ? m[1] + 1 : 0] + nu * n, &ldv, vr, &dim, &(Wrapper<K>::d__0), U + nu * n, &ldv);
                        Blas<K>::gemm("N", "N", &row, &k, &dim, &(Wrapper<K>::d__1), *save + nu * (m[1] + 1), &ldh, vr, &dim, &(Wrapper<K>::d__0), *H + nu * (m[1] + 1), &ldh);
                        Lapack<K>::geqrf(&row, &k, *H + nu * (m[1] + 1), &ldh, vr, work, &lwork, &info);
                        Lapack<K>::mqr("R", "N", &n, &row, &k, *H + nu * (m[1] + 1), &ldh, vr, *v + nu * n, &ldv, work, &lwork, &info);
                        Wrapper<K>::template omatcopy<'N'>(k, n, *v + nu * n, ldv, C + nu * n, ldv);
                        Blas<K>::trsm("R", "U", "N", "N", &n, &k, &(Wrapper<K>::d__1), *H + nu * (m[1] + 1), &ldh, U + nu * n, &ldv);
                        delete [] vr;
                        delete [] work;
                    }
                }
            }
            else if(j > m[1] - k) {
                const unsigned short active = std::count_if(hasConverged, hasConverged + mu, [](short nu) { return nu != 0; });
                unsigned short* const activeSet = new unsigned short[active];
                for(unsigned short nu = 0, curr = 0; nu < active; ++curr)
                    if(hasConverged[curr])
                        activeSet[nu++] = curr;
                K* prod = (id[4] % 4 == HPDDM_RECYCLE_STRATEGY_B ? nullptr : new K[k * active * (m[1] + 2)]);
                if(excluded || !n) {
                    if(id[4] % 4 != HPDDM_RECYCLE_STRATEGY_B) {
                        std::fill_n(prod, k * active * (m[1] + 2), K());
                        MPI_Allreduce(MPI_IN_PLACE, prod, k * active * (m[1] + 2), Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    }
                }
                else {
                    std::copy_n(C, k * ldv, *v);
                    if(id[1] == HPDDM_VARIANT_FLEXIBLE)
                        std::copy_n(v[m[1] + 1], k * ldv, U);
                    if(id[4] % 4 != HPDDM_RECYCLE_STRATEGY_B) {
                        info = m[1] + 1;
                        for(unsigned short nu = 0; nu < active; ++nu) {
                            if(d) {
                                for(i = 0; i < k; ++i)
                                    Wrapper<K>::diag(n, d, U + activeSet[nu] * n + i * ldv, C + i * n);
                                Blas<K>::gemm(&(Wrapper<K>::transc), "N", &info, &k, &n, &(Wrapper<K>::d__1), *v + activeSet[nu] * n, &ldv, C, &n, &(Wrapper<K>::d__0), prod + k * nu * info, &info);
                                for(i = 0; i < k; ++i)
                                    prod[k * active * info + k * nu + i] = Blas<K>::dot(&n, U + activeSet[nu] * n + i * ldv, &i__1, C + i * n, &i__1);
                            }
                            else {
                                Blas<K>::gemm(&(Wrapper<K>::transc), "N", &info, &k, &n, &(Wrapper<K>::d__1), *v + activeSet[nu] * n, &ldv, U + activeSet[nu] * n, &ldv, &(Wrapper<K>::d__0), prod + k * nu * info, &info);
                                for(i = 0; i < k; ++i)
                                    prod[k * active * info + k * nu + i] = Blas<K>::dot(&n, U + activeSet[nu] * n + i * ldv, &i__1, U + activeSet[nu] * n + i * ldv, &i__1);
                            }
                        }
                        MPI_Allreduce(MPI_IN_PLACE, prod, k * active * (m[1] + 2), Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        std::for_each(prod + k * active * (m[1] + 1), prod + k * active * (m[1] + 2), [](K& u) { u = 1.0 / std::sqrt(std::real(u)); });
                    }
                    for(unsigned short nu = 0; nu < active; ++nu) {
                        int dim = std::abs(hasConverged[activeSet[nu]]);
                        for(i = 0; i < dim; ++i)
                            std::fill_n(save[i] + i + 2 + activeSet[nu] * (m[1] + 1), m[1] - i - 1, K());
                        K* A = new K[dim * (dim + 2 + !Wrapper<K>::is_complex)];
                        if(id[4] % 4 != HPDDM_RECYCLE_STRATEGY_B)
                            for(i = 0; i < k; ++i) {
                                Blas<K>::scal(&n, prod + k * active * (m[1] + 1) + k * nu + i, U + activeSet[nu] * n + i * ldv, &i__1);
                                for(unsigned short j = 0; j < k; ++j)
                                    A[j + i * dim] = (i == j ? prod[k * active * (m[1] + 1) + k * nu + i] * prod[k * active * (m[1] + 1) + k * nu + i] : Wrapper<K>::d__0);
                            }
                        else {
                            std::fill_n(A, dim * k, K());
                            for(i = 0; i < k; ++i)
                                A[i * (dim + 1)] = 1.0;
                        }
                        int diff = dim - k;
                        Wrapper<K>::template omatcopy<'N'>(diff, k, H[k] + activeSet[nu] * (m[1] + 1), ldh, A + k * dim, dim);
                        if(id[4] % 4 != HPDDM_RECYCLE_STRATEGY_B)
                            for(i = 0; i < k; ++i)
                                Blas<K>::scal(&diff, prod + k * active * (m[1] + 1) + k * nu + i, A + k * dim + i, &dim);
                        Wrapper<K>::template omatcopy<'C'>(diff, k, A + k * dim, dim, A + k, dim);
                        int row = diff + 1;
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &diff, &k, &(Wrapper<K>::d__1), H[k] + activeSet[nu] * (m[1] + 1), &ldh, H[k] + activeSet[nu] * (m[1] + 1), &ldh, &(Wrapper<K>::d__0), A + k * dim + k, &dim);
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &diff, &row, &(Wrapper<K>::d__1), *save + activeSet[nu] * (m[1] + 1), &ldh, *save + activeSet[nu] * (m[1] + 1), &ldh, &(Wrapper<K>::d__1), A + k * dim + k, &dim);
                        K* B = new K[dim * (dim + 1)]();
                        if(id[4] % 4 != HPDDM_RECYCLE_STRATEGY_B) {
                            row = dim + 1;
                            for(i = 0; i < k; ++i)
                                std::transform(prod + k * nu * (m[1] + 1) + i * (m[1] + 1), prod + k * nu * (m[1] + 1) + i * (m[1] + 1) + dim + 1, B + i * (dim + 1), [&](const K& u) { return prod[k * active * (m[1] + 1) + k * nu + i] * u; });
                            Wrapper<K>::template omatcopy<'C'>(diff, diff, *save + activeSet[nu] * (m[1] + 1), ldh, B + k + k * (dim + 1), dim + 1);
                            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &k, &row, &(Wrapper<K>::d__1), *save + activeSet[nu] * (m[1] + 1), &ldh, B + k, &row, &(Wrapper<K>::d__0), *H + k + 1 + activeSet[nu] * (m[1] + 1), &ldh);
                            Wrapper<K>::template omatcopy<'N'>(k, diff, *H + k + 1 + activeSet[nu] * (m[1] + 1), ldh, B + k, dim + 1);
                            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &k, &k, &(Wrapper<K>::d__1), H[k] + activeSet[nu] * (m[1] + 1), &ldh, B, &row, &(Wrapper<K>::d__1), B + k, &row);
                            for(i = 0; i < k; ++i)
                                Blas<K>::scal(&k, prod + k * active * (m[1] + 1) + k * nu + i, B + i, &row);
                        }
                        else {
                            row = dim;
                            for(i = 0; i < k; ++i)
                                B[i * (dim + 1)] = 1.0;
                            int diff = dim - k;
                            Wrapper<K>::template omatcopy<'C'>(diff, k, H[k] + activeSet[nu] * (m[1] + 1), ldh, B + k, dim);
                            Wrapper<K>::template omatcopy<'C'>(diff, diff, *save + activeSet[nu] * (m[1] + 1), ldh, B + k + k * dim, dim);
                        }
                        K* alpha = A + dim * dim;
                        int lwork = -1;
                        K* vr = new K[dim * dim];
                        Lapack<K>::ggev("N", "V", &dim, A, &dim, B, &row, alpha, alpha + 2 * dim, alpha + dim, nullptr, &i__1, nullptr, &dim, alpha, &lwork, nullptr, &info);
                        lwork = std::real(*alpha);
                        K* work = new K[Wrapper<K>::is_complex ? (lwork + 4 * dim) : lwork];
                        underlying_type<K>* rwork = reinterpret_cast<underlying_type<K>*>(work + lwork);
                        Lapack<K>::ggev("N", "V", &dim, A, &dim, B, &row, alpha, alpha + 2 * dim, alpha + dim, nullptr, &i__1, vr, &dim, work, &lwork, rwork, &info);
                        std::vector<std::pair<unsigned short, std::complex<underlying_type<K>>>> q;
                        q.reserve(dim);
                        selectNu(id[3], q, dim, alpha, alpha + 2 * dim, alpha + dim);
                        delete [] B;
                        delete [] A;
                        info = std::accumulate(q.cbegin(), q.cbegin() + k, 0, [](int a, typename decltype(q)::const_reference b) { return a + b.first; });
                        for(i = k; info != (k * (k - 1)) / 2 && i < dim; ++i)
                            info += q[i].first;
                        int* perm = new int[i];
                        for(unsigned short j = 0; j < i; ++j)
                            perm[j] = q[j].first + 1;
                        decltype(q)().swap(q);
                        Lapack<K>::lapmt(&i__1, &dim, &(info = i), vr, &dim, perm);
                        row = diff + 1;
                        Blas<K>::gemm("N", "N", &row, &k, &diff, &(Wrapper<K>::d__1), *save + activeSet[nu] * (m[1] + 1), &ldh, vr + k, &dim, &(Wrapper<K>::d__0), *H + k + activeSet[nu] * (m[1] + 1), &ldh);
                        Wrapper<K>::template omatcopy<'N'>(k, k, vr, dim, *H + activeSet[nu] * (m[1] + 1), ldh);
                        if(id[4] % 4 != HPDDM_RECYCLE_STRATEGY_B)
                            for(i = 0; i < k; ++i)
                                Blas<K>::scal(&k, prod + k * active * (m[1] + 1) + k * nu + i, *H + activeSet[nu] * (m[1] + 1) + i, &ldh);
                        Blas<K>::gemm("N", "N", &k, &k, &diff, &(Wrapper<K>::d__1), H[k] + activeSet[nu] * (m[1] + 1), &ldh, vr + k, &dim, &(Wrapper<K>::d__1), *H + activeSet[nu] * (m[1] + 1), &ldh);
                        row = dim + 1;
                        *perm = -1;
                        Lapack<K>::geqrf(&row, &k, nullptr, &ldh, nullptr, work, perm, &info);
                        Lapack<K>::mqr("R", "N", &n, &row, &k, nullptr, &ldh, nullptr, nullptr, &ldv, work + 1, perm, &info);
                        delete [] perm;
                        lwork = std::max(std::real(work[0]), std::real(work[1]));
                        delete [] work;
                        work = new K[lwork];
                        Lapack<K>::geqrf(&row, &k, *H + activeSet[nu] * (m[1] + 1), &ldh, Ax, work, &lwork, &info);
                        if(d)
                            Wrapper<K>::template omatcopy<'N'>(k, n, *v + activeSet[nu] * n, ldv, C + activeSet[nu] * n, ldv);
                        Wrapper<K>::template omatcopy<'N'>(k, n, U + activeSet[nu] * n, ldv, v[id[1] == HPDDM_VARIANT_FLEXIBLE ? m[1] + 1 : 0] + activeSet[nu] * n, ldv);
                        Blas<K>::gemm("N", "N", &n, &k, &dim, &(Wrapper<K>::d__1), v[id[1] == HPDDM_VARIANT_FLEXIBLE ? m[1] + 1 : 0] + activeSet[nu] * n, &ldv, vr, &dim, &(Wrapper<K>::d__0), U + activeSet[nu] * n, &ldv);
                        Blas<K>::trsm("R", "U", "N", "N", &n, &k, &(Wrapper<K>::d__1), *H + activeSet[nu] * (m[1] + 1), &ldh, U + activeSet[nu] * n, &ldv);
                        Wrapper<K>::template omatcopy<'N'>(k, n, C + activeSet[nu] * n, ldv, *v + activeSet[nu] * n, ldv);
                        Lapack<K>::mqr("R", "N", &n, &row, &k, *H + activeSet[nu] * (m[1] + 1), &ldh, Ax, *v + activeSet[nu] * n, &ldv, work, &lwork, &info);
                        Wrapper<K>::template omatcopy<'N'>(k, n, *v + activeSet[nu] * n, ldv, C + activeSet[nu] * n, ldv);
                        delete [] work;
                        delete [] vr;
                    }
                }
                delete [] prod;
                delete [] activeSet;
            }
        }
        if(converged)
            break;
    }
#if !HPDDM_PETSC
    if(j != 0 && j != m[0] + 1 && id[4] / 4)
        (*Option::get())[A.prefix("recycle_same_system")] += 1;
#endif
    convergence<4>(id[0], j, m[0]);
    delete [] hasConverged;
    A.end(allocate);
    delete [] s;
    delete [] *save;
    delete [] H;
    return std::min(j, m[0]);
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::BGCRODR(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
#if !defined(_KSPIMPL_H)
    underlying_type<K> tol[2];
    int k;
    unsigned short m[3];
    char id[5];
    options<5>(A, tol, &k, m, id);
#else
    underlying_type<K>* tol = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->rcntl;
    int k = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->icntl[0];
    unsigned short* m = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->scntl;
    char* id = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl;
#endif
    if(k <= 0) {
        if(id[0])
            std::cout << "WARNING -- please choose a positive number of Ritz vectors to compute, now switching to BGMRES" << std::endl;
        return BGMRES<excluded>(A, b, x, mu, comm);
    }
    int ierr;
    const int n = excluded ? 0 : A.getDof();
    int ldh = mu * (m[1] + 1);
    K** const H = new K*[m[1] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 4 : 3) + 1];
    K** const save = H + m[1];
    *save = new K[ldh * mu * m[1]]();
    K** const v = save + m[1];
    int info;
    int N = 2 * mu;
    const underlying_type<K>* const d = A.getScaling();
    int ldv = mu * n;
    K* U = A.storage(), *C = nullptr;
    if(U) {
        std::pair<unsigned short, unsigned short> storage = A.k();
        if(storage.second * storage.first < mu) {
            const_cast<Operator&>(A).destroy();
            U = nullptr;
        }
        else {
            k = (storage.first >= mu ? storage.second : (storage.second * storage.first) / mu);
            C = U + storage.first * storage.second * n;
        }
    }
    int lwork = mu * (d ? (n + (id[1] == HPDDM_VARIANT_RIGHT ? std::max(n, ldh) : ldh)) : std::max((id[1] == HPDDM_VARIANT_RIGHT ? 2 : 1) * n, ldh));
    *H = new K[lwork + (d && U && id[1] == HPDDM_VARIANT_RIGHT && id[4] / 4 == 0 ? mu * n * std::max(2 * k - m[1] - 2, 0) : 0) + mu * ((m[1] + 1) * ldh + n * (m[1] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1) + 2 * m[1]) + (Wrapper<K>::is_complex ? (mu + 1) / 2 : mu)];
    *v = *H + m[1] * mu * ldh;
    K* const Ax = *v + ldv * (m[1] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1);
    K* const s = Ax + lwork + (d && U && id[1] == HPDDM_VARIANT_RIGHT && id[4] / 4 == 0 ? mu * n * std::max(2 * k - m[1] - 2, 0) : 0);
    K* const tau = s + mu * ldh;
    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(tau + m[1] * N);
    ierr = initializeNorm<excluded>(A, id[1], b, x, *v, n, Ax, norm, mu, m[2]);
    const bool allocate = static_cast<bool>(ierr);
    if(ierr < 0)
        return ierr;
    MPI_Allreduce(MPI_IN_PLACE, norm, mu / m[2], Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
    for(unsigned short nu = 0; nu < mu / m[2]; ++nu) {
        norm[nu] = std::sqrt(norm[nu]);
        if(norm[nu] < HPDDM_EPS)
            norm[nu] = 1.0;
    }
    unsigned short j = 1;
    short dim = mu * m[1];
    int* const piv = new int[mu];
    int deflated = -1;
    while(j <= m[0]) {
        if(!excluded) {
            ierr = A.GMV(x, id[1] == HPDDM_VARIANT_LEFT ? Ax : *v, mu);HPDDM_CHKERRQ(ierr)
            Blas<K>::axpby(mu * n, 1.0, b, 1, -1.0, id[1] == HPDDM_VARIANT_LEFT ? Ax : *v, 1);
        }
        if(id[1] == HPDDM_VARIANT_LEFT) {
            ierr = A.template apply<excluded>(Ax, *v, mu);HPDDM_CHKERRQ(ierr)
        }
        if(j == 1 && U) {
            K* pt;
            const int bK = mu * k;
            if(id[1] == HPDDM_VARIANT_RIGHT) {
                pt = *v + ldv;
                if(id[4] / 4 == 0) {
                    ierr = A.template apply<excluded>(U, pt, bK, C);HPDDM_CHKERRQ(ierr)
                }
            }
            else
                pt = U;
            if(id[4] / 4 == 0) {
                if(id[1] == HPDDM_VARIANT_LEFT) {
                    if(!excluded) {
                        ierr = A.GMV(pt, *v + ldv, bK);HPDDM_CHKERRQ(ierr)
                    }
                    ierr = A.template apply<excluded>(*v + ldv, C, bK);HPDDM_CHKERRQ(ierr)
                }
                else if(!excluded) {
                    ierr = A.GMV(pt, C, bK);HPDDM_CHKERRQ(ierr)
                }
                QR<excluded>((id[2] >> 2) & 7, n, bK, C, *save, bK, d, *v + ldv * (id[1] == HPDDM_VARIANT_RIGHT ? (k + 1) : 1), comm);
                if(!excluded && n) {
                    if(id[1] == HPDDM_VARIANT_RIGHT)
                        Blas<K>::trsm("R", "U", "N", "N", &n, &bK, &(Wrapper<K>::d__1), *save, &bK, pt, &n);
                    Blas<K>::trsm("R", "U", "N", "N", &n, &bK, &(Wrapper<K>::d__1), *save, &bK, U, &n);
                }
                std::fill_n(*save, bK * bK, K());
            }
#ifdef PETSCHPDDM_H
            ierr = PetscLogEventBegin(KSP_GMRESOrthogonalization, A._ksp, 0, 0, 0);HPDDM_CHKERRQ(ierr)
#endif
            blockOrthogonalization<excluded>(id[2] & 3, n, k, mu, C, *v, *H, ldh, d, Ax, comm);
#ifdef PETSCHPDDM_H
            ierr = PetscLogEventEnd(KSP_GMRESOrthogonalization, A._ksp, 0, 0, 0);HPDDM_CHKERRQ(ierr)
#endif
            if(id[1] != HPDDM_VARIANT_RIGHT || id[4] / 4 == 0) {
                if(!excluded && n)
                    Blas<K>::gemm("N", "N", &n, &mu, &bK, &(Wrapper<K>::d__1), pt, &n, *H, &ldh, &(Wrapper<K>::d__1), x, &n);
            }
            else {
                if(!excluded && n)
                    Blas<K>::gemm("N", "N", &n, &mu, &bK, &(Wrapper<K>::d__1), U, &n, *H, &ldh, &(Wrapper<K>::d__0), Ax, &n);
                ierr = A.template apply<excluded>(Ax, pt, mu);HPDDM_CHKERRQ(ierr)
                Blas<K>::axpy(&ldv, &(Wrapper<K>::d__1), pt, &i__1, x, &i__1);
            }
        }
        RRQR<excluded>((id[2] >> 2) & 7, n, mu, *v, s, tol[1], N, piv, d, Ax, comm);
        if(N == 0) {
#if HPDDM_PETSC
            KSPMonitor(A._ksp, 0, underlying_type<K>());
#endif
            j = 0;
            break;
        }
        diagonal<5>(id[0], s, mu, tol[1], piv);
#if HPDDM_PETSC
        if(j == 1) {
            underlying_type<K> max = std::abs(s[0]);
            for(unsigned short nu = 1; nu < mu; ++nu)
                max = std::max(max, std::abs(s[nu * (mu + 1)]));
            KSPMonitor(A._ksp, 0, max);
        }
#endif
        if(tol[1] > -0.9 && m[2] <= 1)
            Lapack<underlying_type<K>>::lapmt(&i__1, &i__1, &mu, norm, &i__1, piv);
        if(N != mu) {
            int nrhs = mu - N;
            Lapack<K>::trtrs("U", "N", "N", &N, &nrhs, s, &mu, s + N * mu, &mu, &info);
        }
        if(N != deflated) {
            deflated = N;
            dim = deflated * (j - 1 + m[1] > m[0] ? m[0] - j + 1 : m[1]);
            ldh = deflated * (m[1] + 1);
            ldv = deflated * n;
            for(unsigned short i = 1; i < m[1]; ++i) {
                H[i] = *H + i * deflated * ldh;
                save[i] = *save + i * deflated * ldh;
            }
            for(unsigned short i = 1; i < m[1] * (id[1] == HPDDM_VARIANT_FLEXIBLE ? 2 : 1) + 1; ++i)
                v[i] = *v + i * ldv;
        }
        N *= 2;
        std::fill_n(tau, m[1] * N, K());
        Wrapper<K>::template imatcopy<'N'>(mu, mu, s, mu, ldh);
        std::fill(*H, *v, K());
        if(U) {
            for(unsigned short i = 0; i < mu; ++i)
                std::copy_n(s + i * ldh, deflated, s + i * ldh + deflated * k);
            std::copy_n(*v, ldv, v[k]);
        }
        unsigned short i = (U ? k : 0);
        for(unsigned short nu = 0; nu < deflated; ++nu)
            std::fill(s + i * deflated + nu * (ldh + 1) + 1, s + (nu + 1) * ldh, K());
        if(j == 1 && U)
            std::copy_n(C, k * ldv, *v);
        while(i < m[1] && j <= m[0]) {
            if(id[1] == HPDDM_VARIANT_LEFT) {
                if(!excluded) {
                    ierr = A.GMV(v[i], Ax, deflated);HPDDM_CHKERRQ(ierr)
                }
                ierr = A.template apply<excluded>(Ax, v[i + 1], deflated);HPDDM_CHKERRQ(ierr)
            }
            else {
                ierr = A.template apply<excluded>(v[i], id[1] == HPDDM_VARIANT_FLEXIBLE ? v[i + m[1] + 1] : Ax, deflated, v[i + 1]);HPDDM_CHKERRQ(ierr)
                if(!excluded) {
                    ierr = A.GMV(id[1] == HPDDM_VARIANT_FLEXIBLE ? v[i + m[1] + 1] : Ax, v[i + 1], deflated);HPDDM_CHKERRQ(ierr)
                }
            }
            if(U) {
#ifdef PETSCHPDDM_H
                ierr = PetscLogEventBegin(KSP_GMRESOrthogonalization, A._ksp, 0, 0, 0);HPDDM_CHKERRQ(ierr)
#endif
                blockOrthogonalization<excluded>(id[2] & 3, n, k, deflated, C, v[i + 1], H[i], ldh, d, Ax, comm);
#ifdef PETSCHPDDM_H
                ierr = PetscLogEventEnd(KSP_GMRESOrthogonalization, A._ksp, 0, 0, 0);HPDDM_CHKERRQ(ierr)
#endif
            }
            if(BlockArnoldi<excluded>(id[2], m[1], H, v, tau, s, lwork, n, i++, deflated, d, Ax, comm, save, U ? k : 0)) {
                dim = deflated * (i - 1);
                i = j = 0;
                break;
            }
            const bool converged = (mu == checkBlockConvergence<5>(id[0], j, tol[0], mu, deflated, norm, s + deflated * i, ldh, Ax, m[2]));
#if HPDDM_PETSC
            {
                underlying_type<K>* norm = reinterpret_cast<underlying_type<K>*>(Ax);
                underlying_type<K> max = std::abs(norm[0]);
                for(unsigned short nu = 1; nu < deflated; ++nu)
                    max = std::max(max, std::abs(norm[nu]));
                KSPMonitor(A._ksp, j, max);
            }
#endif
            if(converged) {
                dim = deflated * i;
                i = 0;
                break;
            }
            ++j;
        }
        bool converged;
        if(tol[1] > -0.9)
            Lapack<K>::lapmt(&i__1, &n, &mu, x, &n, piv);
        if(j != m[0] + 1 && i == m[1]) {
            converged = false;
            if(tol[1] > -0.9 && m[2] <= 1)
                Lapack<underlying_type<K>>::lapmt(&i__0, &i__1, &mu, norm, &i__1, piv);
            if(id[0] > 1)
                std::cout << "BGCRODR restart(" << m[1] << ", " << k << ")" << std::endl;
        }
        else {
            if(i == 0 && j == 0)
                break;
            converged = true;
            if(!excluded && j != 0 && j == m[0] + 1) {
                const int rem = (U ? (m[0] - m[1]) % (m[1] - k) : m[0] % m[1]);
                if(rem)
                    dim = deflated * (rem + (U ? k : 0));
            }
        }
        ierr = updateSolRecycling<excluded>(A, id[1], n, x, H, s, v, s, C, U, &dim, k, mu, Ax, comm, deflated);HPDDM_CHKERRQ(ierr)
        if(tol[1] > -0.9)
            Lapack<K>::lapmt(&i__0, &n, &mu, x, &n, piv);
        if(i == m[1] && ((id[2] >> 2) & 7) == 0) {
            if(U)
                i -= k;
            if(!excluded && n)
                Blas<K>::trsm("R", "U", "N", "N", &n, &deflated, &(Wrapper<K>::d__1), save[i - 1] + i * deflated, &ldh, v[m[1]], &n);
        }
#if !HPDDM_PETSC
        if(id[4] / 4 <= 1) {
#else
        {
#endif
            if(!U) {
                int dim = std::min(j, m[1]);
                if(dim < k)
                    k = dim;
                U = const_cast<Operator&>(A).allocate(n, mu, k);
                C = U + k * ldv;
                if(!excluded && n) {
                    std::fill_n(s, deflated * ldh, K());
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &deflated, &deflated, &deflated, &(Wrapper<K>::d__1), save[dim - 1] + dim * deflated, &ldh, save[m[1] - 1] + dim * deflated, &ldh, &(Wrapper<K>::d__0), s + (dim - 1) * deflated, &ldh);
                    dim *= deflated;
                    Lapack<K>::trtrs("U", &(Wrapper<K>::transc), "N", &dim, &deflated, *H, &ldh, s, &ldh, &info);
                    for(i = dim / deflated; i-- > 0; )
                        Lapack<K>::mqr("L", "N", &N, &deflated, &N, H[i] + i * deflated, &ldh, tau + i * N, s + i * deflated, &ldh, Ax, &lwork, &info);
                    for(i = 0; i < dim / deflated; ++i)
                        for(unsigned short nu = 0; nu < deflated; ++nu)
                            std::fill(save[i] + nu * ldh + (i + 1) * deflated + nu + 1, save[i] + (nu + 1) * ldh, K());
                    std::copy_n(*save, deflated * ldh * m[1], *H);
                    for(i = 0; i < deflated; ++i)
                        Blas<K>::axpy(&dim, &(Wrapper<K>::d__1), s + i * ldh, &i__1, H[dim / deflated - 1] + i * ldh, &i__1);
                    int lwork = -1;
                    int row = dim + deflated;
                    int bK = deflated * k;
#if !defined(PETSC_HAVE_SLEPC) || !defined(PETSC_USE_SHARED_LIBRARIES)
                    K* w = new K[Wrapper<K>::is_complex ? dim : (2 * dim)];
                    K* vr = new K[std::max(2, dim * dim)];
                    underlying_type<K>* rwork = Wrapper<K>::is_complex ? new underlying_type<K>[2 * dim] : nullptr;
                    Lapack<K>::geev("N", "V", &dim, nullptr, &ldh, nullptr, nullptr, nullptr, &i__1, nullptr, &dim, vr, &lwork, rwork, &info);
                    lwork = std::real(*vr);
                    K* work = new K[std::max(2, lwork)];
                    Lapack<K>::geev("N", "V", &dim, *H, &ldh, w, w + dim, nullptr, &i__1, vr, &dim, work, &lwork, rwork, &info);
                    delete [] rwork;
                    lwork = -1;
#else
                    K* work = new K[2];
#endif
                    Lapack<K>::geqrf(&row, &bK, nullptr, &ldh, nullptr, work, &lwork, &info);
                    Lapack<K>::mqr("R", "N", &n, &row, &bK, nullptr, &ldh, nullptr, nullptr, &n, work + 1, &lwork, &info);
                    lwork = std::max(std::real(work[0]), std::real(work[1]));
                    delete [] work;
#if !defined(PETSC_HAVE_SLEPC) || !defined(PETSC_USE_SHARED_LIBRARIES)
                    std::vector<std::pair<unsigned short, std::complex<underlying_type<K>>>> q;
                    q.reserve(dim);
                    selectNu(id[3], q, dim, w, w + dim);
                    delete [] w;
                    info = std::accumulate(q.cbegin(), q.cbegin() + bK, 0, [](int a, typename decltype(q)::const_reference b) { return a + b.first; });
                    for(i = bK; info != (bK * (bK - 1)) / 2 && i < dim; ++i)
                        info += q[i].first;
                    int* perm = new int[i];
                    for(unsigned short j = 0; j < i; ++j)
                        perm[j] = q[j].first + 1;
                    decltype(q)().swap(q);
                    Lapack<K>::lapmt(&i__1, &dim, &(info = i), vr, &dim, perm);
                    delete [] perm;
#else
                    K* vr = new K[bK * dim]();
                    if(!loadedKSPSym) {
                        ierr = PetscDLLibrarySym(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, NULL, "KSPHPDDM_Internal", (void**)&loadedKSPSym);CHKERRQ(ierr);
                        if(!loadedKSPSym)
                            return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, -PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, "KSPHPDDM_Internal symbol not found in loaded libhpddm_petsc");
                    }
                    ierr = (*loadedKSPSym)(std::string(A.prefix() + "ksp_hpddm_recycle_").c_str(), comm,
#if !defined(_KSPIMPL_H)
                            1
#else
                            static_cast<PetscMPIInt>(reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl[3])
#endif
                             , static_cast<PetscInt>(dim), *H, ldh, nullptr, 0, static_cast<PetscInt>(bK), vr);HPDDM_CHKERRQ(ierr)
#endif
                    Blas<K>::gemm("N", "N", &n, &bK, &dim, &(Wrapper<K>::d__1), v[id[1] == HPDDM_VARIANT_FLEXIBLE ? m[1] + 1 : 0], &n, vr, &dim, &(Wrapper<K>::d__0), U, &n);
                    Blas<K>::gemm("N", "N", &row, &bK, &dim, &(Wrapper<K>::d__1), *save, &ldh, vr, &dim, &(Wrapper<K>::d__0), *H, &ldh);
                    delete [] vr;
                    work = new K[lwork];
                    Lapack<K>::geqrf(&row, &bK, *H, &ldh, Ax, work, &lwork, &info);
                    Lapack<K>::mqr("R", "N", &n, &row, &bK, *H, &ldh, Ax, *v, &n, work, &lwork, &info);
                    std::copy_n(*v, k * ldv, C);
                    Blas<K>::trsm("R", "U", "N", "N", &n, &bK, &(Wrapper<K>::d__1), *H, &ldh, U, &n);
                    delete [] work;
                }
            }
            else if(j > m[1] - k) {
                int bK = deflated * k;
                int diff = dim - bK;
                K* prod = (id[4] % 4 == 1 ? nullptr : new K[bK * (dim + deflated + 1)]);
                if(excluded || !n) {
                    if(id[4] % 4 != 1) {
                        std::fill_n(prod, bK * (dim + deflated + 1), K());
                        MPI_Allreduce(MPI_IN_PLACE, prod, bK * (dim + deflated + 1), Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    }
                }
                else {
                    std::copy_n(C, k * ldv, *v);
                    if(id[1] == HPDDM_VARIANT_FLEXIBLE)
                        std::copy_n(v[m[1] + 1], k * ldv, U);
                    for(i = 0; i < m[1] - k; ++i)
                        for(unsigned short nu = 0; nu < deflated; ++nu)
                            std::fill(save[i] + nu + 2 + nu * ldh + (i + 1) * deflated, save[i] + (nu + 1) * ldh, K());
                    K* a = new K[dim * (dim + 2 + !Wrapper<K>::is_complex)];
                    if(id[4] % 4 != 1) {
                        info = dim + deflated;
                        if(d) {
                            Wrapper<K>::diag(n, d, U, C, bK);
                            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &info, &bK, &n, &(Wrapper<K>::d__1), *v, &n, C, &n, &(Wrapper<K>::d__0), prod, &info);
                            for(unsigned short nu = 0; nu < bK; ++nu)
                                prod[bK * (dim + deflated) + nu] = Blas<K>::dot(&n, U + nu * n, &i__1, C + nu * n, &i__1);
                        }
                        else {
                            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &info, &bK, &n, &(Wrapper<K>::d__1), *v, &n, U, &n, &(Wrapper<K>::d__0), prod, &info);
                            for(unsigned short nu = 0; nu < bK; ++nu)
                                prod[bK * (dim + deflated) + nu] = Blas<K>::dot(&n, U + nu * n, &i__1, U + nu * n, &i__1);
                        }
                        MPI_Allreduce(MPI_IN_PLACE, prod, bK * (dim + deflated + 1), Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        for(unsigned short nu = 0; nu < bK; ++nu) {
                            prod[bK * (dim + deflated) + nu] = 1.0 / std::sqrt(std::real(prod[bK * (dim + deflated) + nu]));
                            Blas<K>::scal(&n, prod + bK * (dim + deflated) + nu, U + nu * n, &i__1);
                        }
                        for(i = 0; i < bK; ++i)
                            for(unsigned short j = 0; j < bK; ++j)
                                a[j + i * dim] = (i == j ? prod[bK * (dim + deflated) + i] * prod[bK * (dim + deflated) + i] : Wrapper<K>::d__0);
                    }
                    else {
                        std::fill_n(a, dim * bK, K());
                        for(i = 0; i < bK; ++i)
                            a[i * (dim + 1)] = Wrapper<K>::d__1;
                    }
                    Wrapper<K>::template omatcopy<'N'>(diff, bK, H[k], ldh, a + bK * dim, dim);
                    info = dim;
                    if(id[4] % 4 != 1)
                        for(unsigned short nu = 0; nu < bK; ++nu)
                            Blas<K>::scal(&diff, prod + bK * (dim + deflated) + nu, a + bK * dim + nu, &info);
                    Wrapper<K>::template omatcopy<'C'>(diff, bK, a + bK * dim, info, a + bK, info);
                    int row = diff + deflated;
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &diff, &bK, &(Wrapper<K>::d__1), H[k], &ldh, H[k], &ldh, &(Wrapper<K>::d__0), a + bK * dim + bK, &info);
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &diff, &row, &(Wrapper<K>::d__1), *save, &ldh, *save, &ldh, &(Wrapper<K>::d__1), a + bK * dim + bK, &info);
                    K* B = new K[deflated * m[1] * (dim + deflated)]();
                    if(id[4] % 4 != 1) {
                        row = dim + deflated;
                        for(i = 0; i < bK; ++i)
                            std::transform(prod + i * (dim + deflated), prod + (i + 1) * (dim + deflated), B + i * (dim + deflated), [&](const K& u) { return prod[bK * (dim + deflated) + i] * u; });
                        Wrapper<K>::template omatcopy<'C'>(diff, diff, *save, ldh, B + bK + bK * row, row);
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &bK, &row, &(Wrapper<K>::d__1), *save, &ldh, B + bK, &row, &(Wrapper<K>::d__0), *H + deflated * (k + 1), &ldh);
                        Wrapper<K>::template omatcopy<'N'>(bK, diff, *H + deflated * (k + 1), ldh, B + bK, row);
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &bK, &bK, &(Wrapper<K>::d__1), H[k], &ldh, B, &row, &(Wrapper<K>::d__1), B + bK, &row);
                        for(i = 0; i < bK; ++i)
                            Blas<K>::scal(&bK, prod + bK * (dim + deflated) + i, B + i, &row);
                    }
                    else {
                        for(i = 0; i < bK; ++i)
                            B[i * (dim + 1)] = 1.0;
                        Wrapper<K>::template omatcopy<'C'>(diff, bK, H[k], ldh, B + bK, dim);
                        Wrapper<K>::template omatcopy<'C'>(diff, diff, *save, ldh, B + bK + bK * dim, dim);
                        row = dim;
                    }
                    int lwork = -1;
                    int bDim = dim;
#if !defined(PETSC_HAVE_SLEPC) || !defined(PETSC_USE_SHARED_LIBRARIES)
                    K* alpha = a + dim * dim;
                    K* vr = new K[dim * dim];
                    Lapack<K>::ggev("N", "V", &bDim, a, &bDim, B, &row, alpha, alpha + 2 * bDim, alpha + bDim, nullptr, &i__1, nullptr, &bDim, alpha, &lwork, nullptr, &info);
                    lwork = std::real(*alpha);
                    K* work = new K[Wrapper<K>::is_complex ? (lwork + 4 * bDim) : lwork];
                    underlying_type<K>* rwork = reinterpret_cast<underlying_type<K>*>(work + lwork);
                    Lapack<K>::ggev("N", "V", &bDim, a, &bDim, B, &row, alpha, alpha + 2 * bDim, alpha + bDim, nullptr, &i__1, vr, &bDim, work, &lwork, rwork, &info);
                    std::vector<std::pair<unsigned short, std::complex<underlying_type<K>>>> q;
                    q.reserve(bDim);
                    selectNu(id[3], q, bDim, alpha, alpha + 2 * bDim, alpha + bDim);
                    info = std::accumulate(q.cbegin(), q.cbegin() + bK, 0, [](int a, typename decltype(q)::const_reference b) { return a + b.first; });
                    for(i = bK; info != (bK * (bK - 1)) / 2 && i < bDim; ++i)
                        info += q[i].first;
                    int* perm = new int[i];
                    for(unsigned short j = 0; j < i; ++j)
                        perm[j] = q[j].first + 1;
                    decltype(q)().swap(q);
                    Lapack<K>::lapmt(&i__1, &bDim, &(info = i), vr, &bDim, perm);
#else
                    K* vr = new K[bK * dim]();
                    if(!loadedKSPSym) {
                        ierr = PetscDLLibrarySym(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, NULL, "KSPHPDDM_Internal", (void**)&loadedKSPSym);CHKERRQ(ierr);
                        if(!loadedKSPSym)
                            return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, -PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, "KSPHPDDM_Internal symbol not found in loaded libhpddm_petsc");
                    }
                    ierr = (*loadedKSPSym)(std::string(A.prefix() + "ksp_hpddm_recycle_").c_str(), comm,
#if !defined(_KSPIMPL_H)
                            1
#else
                            static_cast<PetscMPIInt>(reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl[3])
#endif
                             , static_cast<PetscInt>(bDim), a, bDim, B, row, static_cast<PetscInt>(bK), vr);HPDDM_CHKERRQ(ierr)
                    int* perm = new int[1];
                    K* work = new K[2];
#endif
                    delete [] B;
                    delete [] a;
                    row = diff + deflated;
                    Blas<K>::gemm("N", "N", &row, &bK, &diff, &(Wrapper<K>::d__1), *save, &ldh, vr + bK, &bDim, &(Wrapper<K>::d__0), *H + bK, &ldh);
                    Wrapper<K>::template omatcopy<'N'>(bK, bK, vr, bDim, *H, ldh);
                    if(id[4] % 4 != 1)
                        for(i = 0; i < bK; ++i)
                            Blas<K>::scal(&bK, prod + bK * (dim + deflated) + i, *H + i, &ldh);
                    Blas<K>::gemm("N", "N", &bK, &bK, &diff, &(Wrapper<K>::d__1), H[k], &ldh, vr + bK, &bDim, &(Wrapper<K>::d__1), *H, &ldh);
                    row = dim + deflated;
                    *perm = -1;
                    Lapack<K>::geqrf(&row, &bK, nullptr, &ldh, nullptr, work, perm, &info);
                    Lapack<K>::mqr("R", "N", &n, &row, &bK, nullptr, &ldh, nullptr, nullptr, &n, work + 1, perm, &info);
                    delete [] perm;
                    lwork = std::max(std::real(work[0]), std::real(work[1]));
                    delete [] work;
                    work = new K[lwork];
                    Lapack<K>::geqrf(&row, &bK, *H, &ldh, Ax, work, &lwork, &info);
                    if(d)
                        std::copy_n(*v, k * ldv, C);
                    std::copy_n(U, k * ldv, v[id[1] == HPDDM_VARIANT_FLEXIBLE ? m[1] + 1 : 0]);
                    Blas<K>::gemm("N", "N", &n, &bK, &bDim, &(Wrapper<K>::d__1), v[id[1] == HPDDM_VARIANT_FLEXIBLE ? m[1] + 1 : 0], &n, vr, &bDim, &(Wrapper<K>::d__0), U, &n);
                    Blas<K>::trsm("R", "U", "N", "N", &n, &bK, &(Wrapper<K>::d__1), *H, &ldh, U, &n);
                    std::copy_n(C, k * ldv, *v);
                    Lapack<K>::mqr("R", "N", &n, &row, &bK, *H, &ldh, Ax, *v, &n, work, &lwork, &info);
                    std::copy_n(*v, k * ldv, C);
                    delete [] work;
                    delete [] vr;
                }
                delete [] prod;
            }
        }
        if(converged)
            break;
    }
#if !HPDDM_PETSC
    if(j != 0 && j != m[0] + 1 && id[4] / 4)
        (*Option::get())[A.prefix("recycle_same_system")] += 1;
#endif
    delete [] piv;
    A.end(allocate);
    delete [] *H;
    delete [] *save;
    delete [] H;
    if(j != 0 || deflated == -1) {
        convergence<5>(id[0], j, m[0]);
        return std::min(j, m[0]);
    }
    else
        return GCRODR<excluded>(A, b, x, mu, comm);
}
} // HPDDM
#endif // _HPDDM_GCRODR_
