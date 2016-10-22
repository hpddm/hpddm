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

#include "iterative.hpp"

namespace HPDDM {
template<bool excluded, class Operator, class K>
inline int IterativeMethod::GMRES(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
    underlying_type<K> tol;
    unsigned short m[2];
    char id[3];
    options<0>(A.prefix(), &tol, nullptr, m, id);
    const int n = excluded ? 0 : A.getDof();
    K** const H = new K*[m[1] * (id[1] == 2 ? 3 : 2) + 1];
    K** const v = H + m[1];
    K* const s = new K[mu * ((m[1] + 1) * (m[1] + 1) + n * (2 + m[1] * (id[1] == 2 ? 2 : 1)) + (!Wrapper<K>::is_complex ? m[1] + 1 : (m[1] + 2) / 2))];
    K* const Ax = s + mu * (m[1] + 1);
    *H = Ax + mu * n;
    for(unsigned short i = 1; i < m[1]; ++i)
        H[i] = *H + i * mu * (m[1] + 1);
    *v = *H + m[1] * mu * (m[1] + 1);
    for(unsigned short i = 1; i < m[1] * (id[1] == 2 ? 2 : 1) + 1; ++i)
        v[i] = *v + i * mu * n;
    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(*v + (m[1] * (id[1] == 2 ? 2 : 1) + 1) * mu * n);
    underlying_type<K>* const sn = norm + mu;
    short* const hasConverged = new short[mu];
    std::fill_n(hasConverged, mu, -m[1]);
    bool allocate = initializeNorm<excluded>(A, id[1], b, x, *v, n, Ax, norm, mu, 1);
    unsigned short j = 1;
    while(j <= m[0]) {
        if(!excluded)
            A.GMV(x, !id[1] ? Ax : *v, mu);
        Blas<K>::axpby(mu * n, 1.0, b, 1, -1.0, !id[1] ? Ax : *v, 1);
        if(!id[1])
            A.template apply<excluded>(Ax, *v, mu);
        for(unsigned short nu = 0; nu < mu; ++nu)
            sn[nu] = std::real(Blas<K>::dot(&n, *v + nu * n, &i__1, *v + nu * n, &i__1));
        if(j == 1) {
            MPI_Allreduce(MPI_IN_PLACE, norm, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                norm[nu] = std::sqrt(norm[nu]);
                if(norm[nu] < HPDDM_EPS)
                    norm[nu] = 1.0;
            }
        }
        else
            MPI_Allreduce(MPI_IN_PLACE, sn, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
        for(unsigned short nu = 0; nu < mu; ++nu) {
            if(hasConverged[nu] > 0)
                hasConverged[nu] = 0;
            s[nu] = std::sqrt(sn[nu]);
            std::for_each(*v + nu * n, *v + (nu + 1) * n, [&](K& y) { y /= s[nu]; });
        }
        unsigned short i = 0;
        while(i < m[1] && j <= m[0]) {
            if(!id[1]) {
                if(!excluded)
                    A.GMV(v[i], Ax, mu);
                A.template apply<excluded>(Ax, v[i + 1], mu);
            }
            else {
                A.template apply<excluded>(v[i], id[1] == 2 ? v[i + m[1] + 1] : Ax, mu, v[i + 1]);
                if(!excluded)
                    A.GMV(id[1] == 2 ? v[i + m[1] + 1] : Ax, v[i + 1], mu);
            }
            Arnoldi<excluded>(id[2], m[1], H, v, s, sn, n, i++, mu, comm);
            checkConvergence<0>(id[0], j, i, tol, mu, norm, s + i * mu, hasConverged, m[1]);
            if(std::find(hasConverged, hasConverged + mu, -m[1]) == hasConverged + mu) {
                i = 0;
                break;
            }
            else
                ++j;
        }
        if(j != m[0] + 1 && i == m[1]) {
            updateSol<excluded>(A, id[1], n, x, H, s, v + (id[1] == 2 ? m[1] + 1 : 0), hasConverged, mu, Ax);
            if(id[0] > 1)
                std::cout << "GMRES restart(" << m[1] << ")" << std::endl;
        }
        else
            break;
    }
    if(!excluded && j == m[0] + 1) {
        const int rem = m[0] % m[1];
        std::for_each(hasConverged, hasConverged + mu, [&](short& d) { if(d < 0) d = rem > 0 ? rem : -d; });
    }
    updateSol<excluded>(A, id[1], n, x, H, s, v + (id[1] == 2 ? m[1] + 1 : 0), hasConverged, mu, Ax);
    convergence<0>(id[0], j, m[0]);
    delete [] hasConverged;
    A.end(allocate);
    delete [] s;
    delete [] H;
    return std::min(j, m[0]);
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::BGMRES(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
    underlying_type<K> tol[2];
    unsigned short m[3];
    char id[3];
    options<1>(A.prefix(), tol, nullptr, m, id);
    const int n = excluded ? 0 : A.getDof();
    K** const H = new K*[m[1] * (id[1] == 2 ? 3 : 2) + 1];
    K** const v = H + m[1];
    int ldh = mu * (m[1] + 1);
    int info;
    int N = 2 * mu;
    int lwork = mu * std::max(n, ((id[2] >> 2) & 7) == 1 ? mu : ldh);
    *H = new K[lwork + mu * ((m[1] + 1) * ldh + n * (m[1] * (id[1] == 2 ? 2 : 1) + 1) + 2 * m[1]) + (Wrapper<K>::is_complex ? (mu + 1) / 2 : mu)];
    *v = *H + m[1] * mu * ldh;
    K* const s = *v + mu * n * (m[1] * (1 + (id[1] == 2)) + 1);
    K* const tau = s + mu * ldh;
    K* const Ax = tau + m[1] * N;
    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(Ax + lwork);
    bool allocate = initializeNorm<excluded>(A, id[1], b, x, *v, n, Ax, norm, mu, m[2]);
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
        if(!excluded)
            A.GMV(x, !id[1] ? Ax : *v, mu);
        Blas<K>::axpby(mu * n, 1.0, b, 1, -1.0, !id[1] ? Ax : *v, 1);
        if(!id[1])
            A.template apply<excluded>(Ax, *v, mu);
        RRQR<excluded>((id[2] >> 2) & 7, n, mu, *v, s, mu, tol[1], N, piv, Ax, comm);
        diagonal<1>(id[0], s, mu, tol[1], piv);
        if(tol[1] > -0.9)
            Lapack<underlying_type<K>>::lapmt(&i__1, &i__1, &mu, norm, &i__1, piv);
        if(N != mu) {
            int nrhs = mu - N;
            Lapack<K>::trtrs("U", "N", "N", &N, &nrhs, s, &mu, s + N * mu, &mu, &info);
        }
        if(N != deflated) {
            deflated = N;
            dim = deflated * (j - 1 + m[1] > m[0] ? m[0] - j + 1 : m[1]);
            ldh = deflated * (m[1] + 1);
            for(unsigned short i = 1; i < m[1]; ++i)
                H[i] = *H + i * deflated * ldh;
            for(unsigned short i = 1; i < m[1] * (id[1] == 2 ? 2 : 1) + 1; ++i)
                v[i] = *v + i * deflated * n;
        }
        N *= 2;
        std::fill_n(tau, m[1] * N, K());
        Wrapper<K>::template imatcopy<'N'>(mu, mu, s, mu, ldh);
        for(unsigned short nu = 0; nu < deflated; ++nu)
            std::fill(s + nu * (ldh + 1) + 1, s + (nu + 1) * ldh, K());
        std::fill(*H, *v, K());
        unsigned short i = 0;
        while(i < m[1] && j <= m[0]) {
            if(!id[1]) {
                if(!excluded)
                    A.GMV(v[i], Ax, deflated);
                A.template apply<excluded>(Ax, v[i + 1], deflated);
            }
            else {
                A.template apply<excluded>(v[i], id[1] == 2 ? v[i + m[1] + 1] : Ax, deflated, v[i + 1]);
                if(!excluded)
                    A.GMV(id[1] == 2 ? v[i + m[1] + 1] : Ax, v[i + 1], deflated);
            }
            if(BlockArnoldi<excluded>(id[2], m[1], H, v, tau, s, lwork, n, i++, deflated, Ax, comm)) {
                dim = deflated * (i - 1);
                i = j = 0;
                break;
            }
            if(deflated == checkBlockConvergence<1>(id[0], j, tol[0], mu, deflated, norm, s + deflated * i, ldh, Ax, m[2])) {
                dim = deflated * i;
                i = 0;
                break;
            }
            else
                ++j;
        }
        if(tol[1] > -0.9)
            Lapack<K>::lapmt(&i__1, &n, &mu, x, &n, piv);
        if(j != m[0] + 1 && i == m[1]) {
            updateSol<excluded>(A, id[1], n, x, H, s, v + (id[1] == 2 ? m[1] + 1 : 0), &dim, mu, Ax, deflated);
            if(tol[1] > -0.9) {
                Lapack<K>::lapmt(&i__0, &n, &mu, x, &n, piv);
                Lapack<underlying_type<K>>::lapmt(&i__0, &i__1, &mu, norm, &i__1, piv);
            }
            if(id[0] > 1)
                std::cout << "BGMRES restart(" << m[1] << ")" << std::endl;
        }
        else
            break;
    }
    if(!excluded && j != 0 && j == m[0] + 1) {
        const int rem = m[0] % m[1];
        if(rem != 0)
            dim = deflated * rem;
    }
    updateSol<excluded>(A, id[1], n, x, H, s, v + (id[1] == 2 ? m[1] + 1 : 0), &dim, mu, Ax, deflated);
    if(tol[1] > -0.9)
        Lapack<K>::lapmt(&i__0, &n, &mu, x, &n, piv);
    delete [] piv;
    A.end(allocate);
    delete [] *H;
    delete [] H;
    if(j != 0) {
        convergence<1>(id[0], j, m[0]);
        return std::min(j, m[0]);
    }
    else
        return GMRES<excluded>(A, b, x, mu, comm);
}
} // HPDDM
#endif // _HPDDM_GMRES_
