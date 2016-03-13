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

#include "GMRES.hpp"

namespace HPDDM {
template<class K>
class Recycling : private Singleton {
    private:
        K*              _storage;
        unsigned short       _mu;
        unsigned short        _k;
    public:
        template<int N>
        Recycling(Singleton::construct_key<N>, unsigned short mu) : _storage(), _mu(mu) { }
        ~Recycling() {
            delete [] _storage;
            _storage = nullptr;
        }
        void setMu(const unsigned short mu) { _mu = mu; };
        bool recycling() const {
            return _storage != nullptr;
        }
        void allocate(int n, unsigned short k) {
            _k = k;
            _storage = new K[2 * _mu * _k * n];
        }
        K* storage() const {
            return _storage;
        }
        unsigned short k() const {
            return _k;
        }
        template<int N = 0>
        static std::shared_ptr<Recycling> get(unsigned short mu) {
            return Singleton::get<Recycling, N>(mu);
        }
};

template<bool excluded, class Operator, class K>
inline int IterativeMethod::GCRODR(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
    const Option& opt = *Option::get();
    int k = opt.val<int>("recycle", 0);
    const unsigned char verbosity = opt.val<unsigned char>("verbosity");
    if(k <= 0) {
        if(verbosity)
            std::cout << "WARNING -- please choose a positive number of Ritz vectors to compute, now switching to GMRES" << std::endl;
        return GMRES(A, b, x, mu, comm);
    }
    const int n = excluded ? 0 : A.getDof();
    const unsigned short it = opt["max_it"];
    underlying_type<K> tol = opt["tol"];
    std::cout << std::scientific;
    epsilon(tol, verbosity);
    const int m = std::min(static_cast<unsigned short>(std::numeric_limits<short>::max()), std::min(static_cast<unsigned short>(opt["gmres_restart"]), it));
    k = std::min(m - 1, k);
    const char variant = (opt["variant"] == 0 ? 'L' : opt["variant"] == 1 ? 'R' : 'F');

    const int ldh = mu * (m + 1);
    K** const H = new K*[m * (3 + (variant == 'F')) + 1];
    K** const save = H + m;
    *save = new K[ldh * m];
    K** const v = save + m;
    K* const s = new K[mu * ((m + 1) * (m + 1) + n * (2 + (variant == 'R') + m * (1 + (variant == 'F'))) + (!Wrapper<K>::is_complex ? m + 1 : (m + 2) / 2))];
    K* const Ax = s + ldh;
    const int ldv = mu * n;
    *H = Ax + (1 + (variant == 'R')) * ldv;
    for(unsigned short i = 1; i < m; ++i) {
        H[i] = *H + i * ldh;
        save[i] = *save + i * ldh;
    }
    *v = *H + m * ldh;
    for(unsigned short i = 1; i < m * (1 + (variant == 'F')) + 1; ++i)
        v[i] = *v + i * ldv;
    bool alloc = A.setBuffer(mu);
    short* const hasConverged = new short[mu];
    std::fill_n(hasConverged, mu, -m);

    int info;
    unsigned short j = 1;
    bool recycling;
    K* U, *C = nullptr;
    Recycling<K>& recycled = *Recycling<K>::get(mu);
    if(recycled.recycling()) {
        recycling = true;
        k = recycled.k();
        U = recycled.storage();
        C = U + k * ldv;
    }
    else
        recycling = false;

    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(*v + (m * (1 + (variant == 'F')) + 1) * ldv);
    underlying_type<K>* const sn = norm + mu;
    A.template start<excluded>(b, x, mu);
    if(variant == 'L') {
        A.template apply<excluded>(b, *v, mu, Ax);
        for(unsigned short nu = 0; nu < mu; ++nu)
            norm[nu] = std::real(Blas<K>::dot(&n, *v + nu * n, &i__1, *v + nu * n, &i__1));
    }
    else
        localSquaredNorm(b, n, norm, mu);

    char id = opt.val<char>("orthogonalization", 0) + 4 * opt.val<char>("qr", 0);
    while(j <= it) {
        unsigned short i = (recycling ? k : 0);
        if(!excluded)
            A.GMV(x, variant == 'L' ? Ax : v[i], mu);
        Blas<K>::axpby(ldv, 1.0, b, 1, -1.0, variant == 'L' ? Ax : v[i], 1);
        if(variant == 'L')
            A.template apply<excluded>(Ax, v[i], mu);
        if(j == 1 && recycling) {
            K* pt;
            switch(variant) {
                case 'L': pt = U; break;
                case 'R': pt = *v;
                          for(unsigned short nu = 0; nu < k; ++nu)
                              A.template apply<excluded>(U + nu * ldv, pt + nu * ldv, mu, Ax);
                          break;
                default: std::copy_n(U, k * ldv, pt = v[m + 1]);
            }
            if(!opt.val<unsigned short>("recycle_same_system")) {
                for(unsigned short nu = 0; nu < k; ++nu) {
                    if(variant == 'L') {
                        A.GMV(pt + nu * ldv, Ax, mu);
                        A.template apply<excluded>(Ax, C + nu * ldv, mu);
                    }
                    else
                        A.GMV(pt + nu * ldv, C + nu * ldv, mu);
                }
                K* work = new K[k * k * mu];
                QR<excluded>(id / 4, n, k, mu, C, work, k, comm);
                delete [] work;
            }
            orthogonalization<excluded>(id % 4, n, k, mu, C, v[i], H[i], comm);
            for(unsigned short nu = 0; nu < mu; ++nu)
                Blas<K>::gemv("N", &n, &k, &(Wrapper<K>::d__1), pt + nu * n, &ldv, H[i] + nu, &mu, &(Wrapper<K>::d__1), x + nu * n, &i__1);
            std::copy_n(C, k * ldv, *v);
        }
        for(unsigned short nu = 0; nu < mu; ++nu)
            sn[nu] = std::real(Blas<K>::dot(&n, v[i] + nu * n, &i__1, v[i] + nu * n, &i__1));
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
            s[mu * i + nu] = std::sqrt(sn[nu]);
            if(recycling) {
                sn[mu * i + nu] = sn[nu];
                sn[nu] = std::real(s[mu * i + nu]);
            }
            std::for_each(v[i] + nu * n, v[i] + (nu + 1) * n, [&](K& y) { y /= s[mu * i + nu]; });
        }
        while(i < m && j <= it) {
            if(variant == 'L') {
                if(!excluded)
                    A.GMV(v[i], Ax, mu);
                A.template apply<excluded>(Ax, v[i + 1], mu);
            }
            else {
                A.template apply<excluded>(v[i], variant == 'F' ? v[i + m + 1] : Ax, mu, v[i + 1]);
                if(!excluded)
                    A.GMV(variant == 'F' ? v[i + m + 1] : Ax, v[i + 1], mu);
            }
            if(recycling)
                orthogonalization<excluded>(id % 4, n, k, mu, C, v[i + 1], H[i], comm);
            Arnoldi<excluded>(id % 4, m, H, v, s, sn, n, i++, mu, comm, save, recycling ? k : 0);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                if(hasConverged[nu] == -m && ((tol > 0 && std::abs(s[i * mu + nu]) / norm[nu] <= tol) || (tol < 0 && std::abs(s[i * mu + nu]) <= -tol)))
                    hasConverged[nu] = i;
            }
            if(verbosity > 0) {
                int tmp[2] { 0, 0 };
                underlying_type<K> beta = std::abs(s[i * mu]);
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    if(hasConverged[nu] != -m)
                        ++tmp[0];
                    else if(std::abs(s[i * mu + nu]) > beta) {
                        beta = std::abs(s[i * mu + nu]);
                        tmp[1] = nu;
                    }
                }
                if(tol > 0)
                    std::cout << "GCRODR: " << std::setw(3) << j << " " << beta << " " <<  norm[tmp[1]] << " " <<  beta / norm[tmp[1]] << " < " << tol;
                else
                    std::cout << "GCRODR: " << std::setw(3) << j << " " << beta << " < " << -tol;
                if(mu > 1) {
                    std::cout << " (rhs #" << tmp[1] + 1;
                    if(tmp[0] > 0)
                        std::cout << ", " << tmp[0] << " converged rhs";
                    std::cout << ")";
                }
                std::cout << std::endl;
            }
            if(std::find(hasConverged, hasConverged + mu, -m) == hasConverged + mu) {
                i += (recycling ? m - k : m);
                break;
            }
            else
                ++j;
        }
        bool converged;
        if(j != it + 1 && i == m) {
            converged = false;
            if(verbosity > 0)
                std::cout << "GCRODR restart(" << m << ", " << k << ")" << std::endl;
        }
        else {
            converged = true;
            if(!excluded && j == it + 1) {
                int rem = (recycling ? (it - m) % (m - k) : it % m);
                if(rem != 0) {
                    if(recycling)
                        rem += k;
                    std::for_each(hasConverged, hasConverged + mu, [&](short& dim) { if(dim < 0) dim = rem; });
                }
            }
        }
        if(!excluded)
            updateSolRecycling(A, variant, n, x, H, s, v, sn, C, U, hasConverged, k, mu, Ax, comm);
        else
            addSol(A, variant, n, x, std::distance(H[0], H[1]), s, v + (m + 1) * (variant == 'F'), hasConverged, mu, Ax);
        if(i == m) {
            if(recycling)
                i -= k;
            for(unsigned short nu = 0; nu < mu; ++nu)
                std::for_each(v[m] + nu * n, v[m] + (nu + 1) * n, [&](K& y) { y /= save[i - 1][i + nu * (m + 1)]; });
        }
        if(!recycling) {
            recycling = true;
            int dim = std::abs(*std::min_element(hasConverged, hasConverged + mu, [](const short& lhs, const short& rhs) { return lhs == 0 ? false : rhs == 0 ? true : lhs < rhs; }));
            if(j < k || dim < k)
                k = dim;
            recycled.allocate(n, k);
            U = recycled.storage();
            C = U + k * ldv;
            std::fill_n(s, dim * mu, K());
            for(unsigned short nu = 0; nu < mu; ++nu) {
                K h = H[dim - 1][(m + 1) * nu + dim] / H[dim - 1][(m + 1) * nu + dim - 1];
                for(i = dim; i-- > 1; ) {
                    s[i + dim * nu] = H[i - 1][(m + 1) * nu + i] * h;
                    h *= -sn[(i - 1) * mu + nu];
                }
                s[dim * nu] = h;
                for(i = 0; i < dim; ++i) {
                    std::fill_n(save[i] + i + 2 + nu * (m + 1), m - i - 1, K());
                    std::copy_n(save[i] + nu * (m + 1), m + 1, H[i] + nu * (m + 1));
                }
                h = save[dim - 1][dim + nu * (m + 1)] * save[dim - 1][dim + nu * (m + 1)];
                Blas<K>::axpy(&dim, &h, s + dim * nu, &i__1, H[dim - 1] + nu * (m + 1), &i__1);
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
                Wrapper<K>::template omatcopy<'N'>(dim, dim, *H + nu * (m + 1), ldh, backup, dim);
                Lapack<K>::hseqr("E", "N", &dim, &i__1, &dim, backup, &dim, w, w + dim, nullptr, &i__1, work, &lwork, &info);
                delete [] backup;
                std::vector<std::pair<unsigned short, underlying_type<K>>> p;
                p.reserve(k + 1);
                for(i = 0; i < dim; ++i) {
                    underlying_type<K> magnitude = Wrapper<K>::is_complex ? std::norm(w[i]) : std::real(w[i] * w[i] + w[dim + i] * w[dim + i]);
                    typename decltype(p)::iterator it = std::lower_bound(p.begin(), p.end(), std::make_pair(i, magnitude), [](const std::pair<unsigned short, underlying_type<K>>& lhs, const std::pair<unsigned short, underlying_type<K>>& rhs) { return lhs.second < rhs.second; });
                    if(p.size() < k || it != p.end())
                        p.insert(it, std::make_pair(i, magnitude));
                    if(p.size() == k + 1)
                        p.pop_back();
                }
                int mm = Wrapper<K>::is_complex ? k : 0;
                for(typename decltype(p)::const_iterator it = p.cbegin(); it < p.cend(); ++it) {
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
                decltype(p)().swap(p);
                underlying_type<K>* rwork = Wrapper<K>::is_complex ? new underlying_type<K>[dim] : nullptr;
                K* vr = new K[mm * dim];
                int* ifailr = new int[mm];
                int col;
                Lapack<K>::hsein("R", "Q", "N", select, &dim, *H + nu * (m + 1), &ldh, w, w + dim, nullptr, &i__1, vr, &dim, &mm, &col, work, rwork, nullptr, ifailr, &info);
                delete [] ifailr;
                delete [] select;
                delete [] rwork;
                delete [] w;
                Blas<K>::gemm("N", "N", &n, &k, &dim, &(Wrapper<K>::d__1), v[(m + 1) * (variant == 'F')] + nu * n, &ldv, vr, &dim, &(Wrapper<K>::d__0), U + nu * n, &ldv);
                Blas<K>::gemm("N", "N", &row, &k, &dim, &(Wrapper<K>::d__1), *save + nu * (m + 1), &ldh, vr, &dim, &(Wrapper<K>::d__0), *H + nu * (m + 1), &ldh);
                delete [] vr;
                K* tau = new K[k];
                Lapack<K>::geqrf(&row, &k, *H + nu * (m + 1), &ldh, tau, work, &lwork, &info);
                Lapack<K>::mqr("R", "N", &n, &row, &k, *H + nu * (m + 1), &ldh, tau, *v + nu * n, &ldv, work, &lwork, &info);
                Wrapper<K>::template omatcopy<'N'>(k, n, *v + nu * n, ldv, C + nu * n, ldv);
                Blas<K>::trsm("R", "U", "N", "N", &n, &k, &(Wrapper<K>::d__1), *H + nu * (m + 1), &ldh, U + nu * n, &ldv);
                delete [] tau;
                delete [] work;
            }
        }
        else if(!opt.val<unsigned short>("recycle_same_system")) {
            std::copy_n(C, k * ldv, *v);
            if(variant == 'F')
                std::copy_n(v[m + 1], k * ldv, U);
            const unsigned short active = std::count_if(hasConverged, hasConverged + mu, [](short nu) { return nu != 0; });
            K* prod = new K[k * active * (m + 2) + (active * sizeof(K)) / sizeof(unsigned short) + 1];
            unsigned short* const activeSet = reinterpret_cast<unsigned short*>(prod + k * active * (m + 2));
            for(unsigned short nu = 0, curr = 0; nu < active; ++curr)
                if(hasConverged[curr] != 0)
                    activeSet[nu++] = curr;
            info = m + 1;
            for(unsigned short nu = 0; nu < active; ++nu) {
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", &info, &k, &n, &(Wrapper<K>::d__1), *v + activeSet[nu] * n, &ldv, U + activeSet[nu] * n, &ldv, &(Wrapper<K>::d__0), prod + k * nu * info, &info);
                for(i = 0; i < k; ++i)
                    prod[k * active * info + k * nu + i] = Blas<K>::dot(&n, U + activeSet[nu] * n + i * ldv, &i__1, U + activeSet[nu] * n + i * ldv, &i__1);
            }
            MPI_Allreduce(MPI_IN_PLACE, prod, k * active * (m + 2), Wrapper<K>::mpi_type(), MPI_SUM, comm);
            std::for_each(prod + k * active * (m + 1), prod + k * active * (m + 2), [](K& u) { u = 1.0 / std::sqrt(std::real(u)); });
            for(unsigned short nu = 0; nu < active; ++nu) {
                int dim = std::abs(hasConverged[activeSet[nu]]);
                for(i = 0; i < k; ++i)
                    Blas<K>::scal(&n, prod + k * active * (m + 1) + k * nu + i, U + activeSet[nu] * n + i * ldv, &i__1);
                for(i = 0; i < dim; ++i)
                    std::fill_n(save[i] + i + 2 + activeSet[nu] * (m + 1), m - i - 1, K());
                K* A = new K[dim * dim];
                for(i = 0; i < k; ++i)
                    for(unsigned short j = 0; j < k; ++j)
                        A[j + i * dim] = (i == j ? prod[k * active * (m + 1) + k * nu + i] * prod[k * active * (m + 1) + k * nu + i] : Wrapper<K>::d__0);
                int diff = dim - k;
                Wrapper<K>::template omatcopy<'N'>(diff, k, H[k] + activeSet[nu] * (m + 1), ldh, A + k * dim, dim);
                for(i = 0; i < k; ++i)
                    Blas<K>::scal(&diff, prod + k * active * (m + 1) + k * nu + i, A + k * dim + i, &dim);
                Wrapper<K>::template omatcopy<'C'>(diff, k, A + k * dim, dim, A + k, dim);
                int row = diff + 1;
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &diff, &k, &(Wrapper<K>::d__1), H[k] + activeSet[nu] * (m + 1), &ldh, H[k] + activeSet[nu] * (m + 1), &ldh, &(Wrapper<K>::d__0), A + k * dim + k, &dim);
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &diff, &row, &(Wrapper<K>::d__1), *save + activeSet[nu] * (m + 1), &ldh, *save + activeSet[nu] * (m + 1), &ldh, &(Wrapper<K>::d__1), A + k * dim + k, &dim);
                K* B = new K[dim * (dim + 1)]();
                row = dim + 1;
                for(i = 0; i < k; ++i)
                    std::transform(prod + k * nu * (m + 1) + i * (m + 1), prod + k * nu * (m + 1) + i * (m + 1) + dim + 1, B + i * (dim + 1), [&](const K& u) { return prod[k * active * (m + 1) + k * nu + i] * u; });
                Wrapper<K>::template omatcopy<'C'>(diff, diff, *save + activeSet[nu] * (m + 1), ldh, B + k + k * (dim + 1), dim + 1);
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &k, &row, &(Wrapper<K>::d__1), *save + activeSet[nu] * (m + 1), &ldh, B + k, &row, &(Wrapper<K>::d__0), *H + k + 1 + activeSet[nu] * (m + 1), &ldh);
                Wrapper<K>::template omatcopy<'N'>(k, diff, *H + k + 1 + activeSet[nu] * (m + 1), ldh, B + k, dim + 1);
                Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &k, &k, &(Wrapper<K>::d__1), H[k] + activeSet[nu] * (m + 1), &ldh, B, &row, &(Wrapper<K>::d__1), B + k, &row);
                for(i = 0; i < k; ++i)
                    Blas<K>::scal(&k, prod + k * active * (m + 1) + k * nu + i, B + i, &row);
                K* alpha = new K[(2 + !Wrapper<K>::is_complex) * dim];
                int lwork = -1;
                K* vr = new K[dim * dim];
                Lapack<K>::ggev("N", "V", &dim, A, &dim, B, &row, alpha, alpha + 2 * dim, alpha + dim, nullptr, &i__1, nullptr, &dim, alpha, &lwork, nullptr, &info);
                lwork = std::real(*alpha);
                K* work = new K[Wrapper<K>::is_complex ? (lwork + 4 * dim) : lwork];
                underlying_type<K>* rwork = reinterpret_cast<underlying_type<K>*>(work + lwork);
                Lapack<K>::ggev("N", "V", &dim, A, &dim, B, &row, alpha, alpha + 2 * dim, alpha + dim, nullptr, &i__1, vr, &dim, work, &lwork, rwork, &info);
                std::vector<std::pair<unsigned short, underlying_type<K>>> q;
                q.reserve(dim);
                for(i = 0; i < dim; ++i) {
                    underlying_type<K> magnitude = Wrapper<K>::is_complex ? std::norm(alpha[i] / alpha[dim + i]) : std::real((alpha[i] * alpha[i] + alpha[2 * dim + i] * alpha[2 * dim + i]) / (alpha[dim + i] * alpha[dim + i]));
                    q.emplace_back(i, magnitude);
                }
                std::sort(q.begin(), q.end(), [](const std::pair<unsigned short, underlying_type<K>>& lhs, const std::pair<unsigned short, underlying_type<K>>& rhs) { return lhs.second < rhs.second; });
                info = std::accumulate(q.cbegin(), q.cbegin() + k, 0, [](int a, const std::pair<unsigned short, underlying_type<K>>& b) { return a + b.first; });
                for(i = k; info != (k * (k - 1)) / 2 && i < dim; ++i)
                    info += q[i].first;
                int* perm = new int[i];
                std::transform(q.cbegin(), q.cbegin() + i, perm, [](const std::pair<unsigned short, underlying_type<K>>& u) { return u.first + 1; });
                decltype(q)().swap(q);
                Lapack<K>::lapmt(&i__1, &dim, &(info = i), vr, &dim, perm);
                row = diff + 1;
                Blas<K>::gemm("N", "N", &row, &k, &diff, &(Wrapper<K>::d__1), *save + activeSet[nu] * (m + 1), &ldh, vr + k, &dim, &(Wrapper<K>::d__0), *H + k + activeSet[nu] * (m + 1), &ldh);
                Wrapper<K>::template omatcopy<'N'>(k, k, vr, dim, *H + activeSet[nu] * (m + 1), ldh);
                for(i = 0; i < k; ++i)
                    Blas<K>::scal(&k, prod + k * active * (m + 1) + k * nu + i, *H + activeSet[nu] * (m + 1) + i, &ldh);
                Blas<K>::gemm("N", "N", &k, &k, &diff, &(Wrapper<K>::d__1), H[k] + activeSet[nu] * (m + 1), &ldh, vr + k, &dim, &(Wrapper<K>::d__1), *H + activeSet[nu] * (m + 1), &ldh);
                row = dim + 1;
                K* tau = new K[k];
                *perm = -1;
                Lapack<K>::geqrf(&row, &k, nullptr, &ldh, nullptr, work, perm, &info);
                Lapack<K>::mqr("R", "N", &n, &row, &k, nullptr, &ldh, nullptr, nullptr, &ldv, work + 1, perm, &info);
                delete [] perm;
                if(std::real(work[0]) > (Wrapper<K>::is_complex ? (lwork + 4 * dim) : lwork) || std::real(work[1]) > (Wrapper<K>::is_complex ? (lwork + 4 * dim) : lwork)) {
                    lwork = std::max(std::real(work[0]), std::real(work[1]));
                    delete [] work;
                    work = new K[lwork];
                }
                Lapack<K>::geqrf(&row, &k, *H + activeSet[nu] * (m + 1), &ldh, tau, work, &lwork, &info);
                Wrapper<K>::template omatcopy<'N'>(k, n, U + activeSet[nu] * n, ldv, v[(m + 1) * (variant == 'F')] + activeSet[nu] * n, ldv);
                Blas<K>::gemm("N", "N", &n, &k, &dim, &(Wrapper<K>::d__1), v[(m + 1) * (variant == 'F')] + activeSet[nu] * n, &ldv, vr, &dim, &(Wrapper<K>::d__0), U + activeSet[nu] * n, &ldv);
                Blas<K>::trsm("R", "U", "N", "N", &n, &k, &(Wrapper<K>::d__1), *H + activeSet[nu] * (m + 1), &ldh, U + activeSet[nu] * n, &ldv);
                Wrapper<K>::template omatcopy<'N'>(k, n, C + activeSet[nu] * n, ldv, *v + activeSet[nu] * n, ldv);
                Lapack<K>::mqr("R", "N", &n, &row, &k, *H + activeSet[nu] * (m + 1), &ldh, tau, *v + activeSet[nu] * n, &ldv, work, &lwork, &info);
                Wrapper<K>::template omatcopy<'N'>(k, n, *v + activeSet[nu] * n, ldv, C + activeSet[nu] * n, ldv);
                delete [] tau;
                delete [] work;
                delete [] vr;
                delete [] alpha;
                delete [] B;
                delete [] A;
            }
            delete [] prod;
        }
        if(converged)
            break;
    }
    if(verbosity > 0) {
        if(j != it + 1)
            std::cout << "GCRODR converges after " << j << " iteration" << (j > 1 ? "s" : "") << std::endl;
        else
            std::cout << "GCRODR does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
    }
    delete [] hasConverged;
    A.clearBuffer(alloc);
    delete [] s;
    delete [] *save;
    delete [] H;
    std::cout.unsetf(std::ios_base::scientific);
    return std::min(j, it);
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::BGCRODR(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
    const Option& opt = *Option::get();
    int k = opt.val<int>("recycle", 0);
    const unsigned char verbosity = opt.val<unsigned char>("verbosity");
    if(k <= 0) {
        if(verbosity)
            std::cout << "WARNING -- please choose a positive number of Ritz vectors to compute, now switching to BGMRES" << std::endl;
        return BGMRES(A, b, x, mu, comm);
    }
    const int n = excluded ? 0 : A.getDof();
    const unsigned short it = opt["max_it"];
    underlying_type<K> tol = opt["tol"];
    std::cout << std::scientific;
    epsilon(tol, verbosity);
    const unsigned short m = std::min(static_cast<unsigned short>(std::numeric_limits<short>::max()), std::min(static_cast<unsigned short>(opt["gmres_restart"]), it));
    const char variant = (opt["variant"] == 0 ? 'L' : opt["variant"] == 1 ? 'R' : 'F');

    int ldh = mu * (m + 1);
    K** const H = new K*[m * (3 + (variant == 'F')) + 1];
    K** const save = H + m;
    *save = new K[ldh * mu * m]();
    K** const v = save + m;
    int info;
    int N = 2 * mu;
    char id = opt.val<char>("orthogonalization", 0);
    int lwork = mu * std::max((1 + (variant == 'R')) * n, id != 1 ? ldh : mu);
    id += 4 * opt.val<char>("qr", 0);
    *H = new K[lwork + mu * ((m + 1) * ldh + n * (m * (1 + (variant == 'F')) + 1) + 2 * m) + (Wrapper<K>::is_complex ? (mu + 1) / 2 : mu)];
    *v = *H + m * mu * ldh;
    int ldv = mu * n;
    K* const s = *v + ldv * (m * (1 + (variant == 'F')) + 1);
    K* const tau = s + mu * ldh;
    K* const Ax = tau + m * N;
    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(Ax + lwork);
    underlying_type<K>* const beta = norm - mu;
    bool alloc = A.setBuffer(mu);

    A.template start<excluded>(b, x, mu);
    if(variant == 'L') {
        A.template apply<excluded>(b, *v, mu, Ax);
        for(unsigned short nu = 0; nu < mu; ++nu)
            norm[nu] = std::real(Blas<K>::dot(&n, *v + nu * n, &i__1, *v + nu * n, &i__1));
    }
    else
        localSquaredNorm(b, n, norm, mu);
    MPI_Allreduce(MPI_IN_PLACE, norm, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
    for(unsigned short nu = 0; nu < mu; ++nu) {
        norm[nu] = std::sqrt(norm[nu]);
        if(norm[nu] < HPDDM_EPS)
            norm[nu] = 1.0;
    }

    unsigned short j = 1;
    bool recycling;
    K* U, *C = nullptr;
    Recycling<K>& recycled = *Recycling<K>::get(mu);
    if(recycled.recycling()) {
        recycling = true;
        k = recycled.k();
        U = recycled.storage();
        C = U + k * ldv;
    }
    else
        recycling = false;
    short dim = mu * m;
    int* const piv = new int[mu];
    underlying_type<K>* workpiv = norm - 2 * mu;
    int deflated = -1;
    while(j <= it) {
        if(!excluded)
            A.GMV(x, variant == 'L' ? Ax : *v, mu);
        Blas<K>::axpby(mu * n, 1.0, b, 1, -1.0, variant == 'L' ? Ax : *v, 1);
        if(variant == 'L')
            A.template apply<excluded>(Ax, *v, mu);
        if(j == 1 && recycling) {
            K* pt;
            switch(variant) {
                case 'L': pt = U; break;
                case 'R': pt = *v + k * ldv;
                          for(unsigned short nu = 0; nu < k; ++nu)
                              A.template apply<excluded>(U + nu * ldv, pt + nu * ldv, mu, Ax);
                          break;
                default: std::copy_n(U, k * ldv, pt = *v + (m + 1) * ldv);
            }
            int bK = mu * k;
            if(!opt.val<unsigned short>("recycle_same_system")) {
                for(unsigned short nu = 0; nu < k; ++nu) {
                    if(variant == 'L') {
                        A.GMV(pt + nu * ldv, Ax, mu);
                        A.template apply<excluded>(Ax, C + nu * ldv, mu);
                    }
                    else
                        A.GMV(pt + nu * ldv, C + nu * ldv, mu);
                }
                K* work = new K[bK * bK];
                QR<excluded>(id / 4, n, bK, 1, C, work, bK, comm, Ax);
                delete [] work;
            }
            blockOrthogonalization<excluded>(id % 4, n, k, mu, C, *v, *H, ldh, Ax, comm);
            Blas<K>::gemm("N", "N", &n, &mu, &bK, &(Wrapper<K>::d__1), pt, &n, *H, &ldh, &(Wrapper<K>::d__1), x, &n);
        }
        VR<excluded>(n, mu, 1, v[0], s, mu, comm);
        if(!opt.set("initial_deflation_tol")) {
            Lapack<K>::potrf("U", &mu, s, &mu, &info);
            if(verbosity > 3) {
                std::cout << "BGCRODR diag(R), QR = block residual: ";
                std::cout << s[0];
                if(mu > 1) {
                    if(mu > 2)
                        std::cout << "\t...";
                    std::cout << "\t" << s[(mu - 1) * (mu + 1)];
                }
                std::cout << std::endl;
            }
            N = (info > 0 ? info - 1 : mu);
        }
        else {
            Lapack<K>::pstrf("U", &mu, s, &mu, piv, &N, &(Wrapper<underlying_type<K>>::d__0), workpiv, &info);
            if(verbosity > 3) {
                std::cout << "BGCRODR diag(R), QR = block residual, with pivoting: ";
                std::cout << s[0] << " (" << piv[0] << ")";
                if(mu > 1) {
                    if(mu > 2)
                        std::cout << "\t...";
                    std::cout << "\t" << s[(mu - 1) * (mu + 1)] << " (" << piv[mu - 1] << ")";
                }
                std::cout << std::endl;
            }
            if(info == 0) {
                N = mu;
                while(N > 1 && std::abs(s[(N - 1) * (mu + 1)] / s[0]) <= opt.val("initial_deflation_tol"))
                    --N;
            }
            Lapack<K>::lapmt(&i__1, &n, &mu, v[0], &n, piv);
            Lapack<underlying_type<K>>::lapmt(&i__1, &i__1, &mu, norm, &i__1, piv);
        }
        if(N != mu) {
            int nrhs = mu - N;
            Lapack<K>::trtrs("U", "N", "N", &N, &nrhs, s, &mu, s + N * mu, &mu, &info);
        }
        if(N != deflated) {
            deflated = N;
            dim = deflated * (j - 1 + m > it ? it - j + 1 : m);
            ldh = deflated * (m + 1);
            ldv = deflated * n;
            for(unsigned short i = 1; i < m; ++i) {
                H[i] = *H + i * deflated * ldh;
                save[i] = *save + i * deflated * ldh;
            }
            for(unsigned short i = 1; i < m * (1 + (variant == 'F')) + 1; ++i)
                v[i] = *v + i * ldv;
        }
        N *= 2;
        std::fill_n(tau, m * N, K());
        Wrapper<K>::template imatcopy<'N'>(mu, mu, s, mu, ldh);
        std::fill(*H, *v, K());
        if(recycling) {
            for(unsigned short i = 0; i < mu; ++i)
                std::copy_n(s + i * ldh, deflated, s + i * ldh + deflated * k);
            std::copy_n(*v, ldv, v[k]);
        }
        unsigned short i = (recycling ? k : 0);
        Blas<K>::trsm("R", "U", "N", "N", &n, &deflated, &(Wrapper<K>::d__1), s, &ldh, v[i], &n);
        for(unsigned short nu = 0; nu < deflated; ++nu)
            std::fill(s + i * deflated + nu * (ldh + 1) + 1, s + (nu + 1) * ldh, K());
        if(j == 1 && recycling) {
            if(variant == 'F')
                std::copy_n(*v + (m + 1) * mu * n, k * ldv, v[m + 1]);
            std::copy_n(C, k * ldv, *v);
        }
        while(i < m && j <= it) {
            if(variant == 'L') {
                if(!excluded)
                    A.GMV(v[i], Ax, deflated);
                A.template apply<excluded>(Ax, v[i + 1], deflated);
            }
            else {
                A.template apply<excluded>(v[i], variant == 'F' ? v[i + m + 1] : Ax, deflated, v[i + 1]);
                if(!excluded)
                    A.GMV(variant == 'F' ? v[i + m + 1] : Ax, v[i + 1], deflated);
            }
            if(recycling)
                blockOrthogonalization<excluded>(id % 4, n, k, deflated, C, v[i + 1], H[i], ldh, Ax, comm);
            if(BlockArnoldi<excluded>(id, m, H, v, tau, s, lwork, n, i++, deflated, Ax, comm, save, recycling ? k : 0)) {
                dim = deflated * (i - 1);
                i = j = 0;
                break;
            }
            unsigned short converged = 0;
            for(unsigned short nu = 0; nu < deflated; ++nu) {
                beta[nu] = Blas<K>::nrm2(&deflated, s + deflated * i + nu * ldh, &i__1);
                if(((tol > 0 && beta[nu] / norm[nu] <= tol) || (tol < 0 && beta[nu] <= -tol)))
                    ++converged;
            }
            if(verbosity > 0) {
                underlying_type<K>* max = std::max_element(beta, beta + deflated);
                if(tol > 0)
                    std::cout << "BGCRODR: " << std::setw(3) << j << " " << *max << " " <<  norm[std::distance(beta, max)] << " " <<  *max / norm[std::distance(beta, max)] << " < " << tol;
                else
                    std::cout << "BGCRODR: " << std::setw(3) << j << " " << *max << " < " << -tol;
                std::cout << " (rhs #" << std::distance(beta, max) + 1;
                if(converged > 0)
                    std::cout << ", " << converged << " converged rhs";
                if(deflated != mu)
                    std::cout << ", " << mu - deflated << " deflated rhs";
                std::cout << ")" << std::endl;
            }
            if(converged == deflated) {
                dim = deflated * i;
                i = 0;
                break;
            }
            else
                ++j;
        }
        bool converged;
        if(opt.set("initial_deflation_tol"))
            Lapack<K>::lapmt(&i__1, &n, &mu, x, &n, piv);
        if(j != it + 1 && i == m) {
            converged = false;
            if(opt.set("initial_deflation_tol"))
                Lapack<underlying_type<K>>::lapmt(&i__0, &i__1, &mu, norm, &i__1, piv);
            if(verbosity > 0)
                std::cout << "BGCRODR restart(" << m << ", " << k << ")" << std::endl;
        }
        else {
            if(i == 0 && j == 0)
                break;
            converged = true;
            if(!excluded && j != 0 && j == it + 1) {
                const int rem = (recycling ? (it - m) % (m - k) : it % m);
                if(rem != 0)
                    dim = deflated * (rem + recycling * k);
            }
        }
        if(!excluded)
            updateSolRecycling(A, variant, n, x, H, s, v, s, C, U, &dim, k, mu, Ax, comm, deflated);
        if(opt.set("initial_deflation_tol"))
            Lapack<K>::lapmt(&i__0, &n, &mu, x, &n, piv);
        if(i == m && id / 4 == 0) {
            if(recycling)
                i -= k;
            Blas<K>::trsm("R", "U", "N", "N", &n, &deflated, &(Wrapper<K>::d__1), save[i - 1] + i * deflated, &ldh, v[m], &n);
        }
        if(!recycling) {
            recycling = true;
            int dim = std::min(j, m);
            if(dim < k)
                k = dim;
            recycled.allocate(deflated * n, k);
            U = recycled.storage();
            C = U + k * ldv;
            std::fill_n(s, deflated * ldh, K());
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &deflated, &deflated, &deflated, &(Wrapper<K>::d__1), save[dim - 1] + dim * deflated, &ldh, save[m - 1] + dim * deflated, &ldh, &(Wrapper<K>::d__0), s + (dim - 1) * deflated, &ldh);
            dim *= deflated;
            Lapack<K>::trtrs("U", &(Wrapper<K>::transc), "N", &dim, &deflated, *H, &ldh, s, &ldh, &info);
            for(i = dim / deflated; i-- > 0; )
                Lapack<K>::mqr("L", "N", &N, &deflated, &N, H[i] + i * deflated, &ldh, tau + i * N, s + i * deflated, &ldh, Ax, &lwork, &info);
            for(i = 0; i < dim / deflated; ++i)
                for(unsigned short nu = 0; nu < deflated; ++nu)
                    std::fill(save[i] + nu * ldh + (i + 1) * deflated + nu + 1, save[i] + (nu + 1) * ldh, K());
            std::copy_n(*save, deflated * ldh * m, *H);
            for(i = 0; i < deflated; ++i)
                Blas<K>::axpy(&dim, &(Wrapper<K>::d__1), s + i * ldh, &i__1, H[dim / deflated - 1] + i * ldh, &i__1);
            int lwork = -1;
            int row = dim + deflated;
            int bK = deflated * k;
            K* w = new K[Wrapper<K>::is_complex ? dim : (2 * dim)];
            K* vr = new K[dim * dim];
            underlying_type<K>* rwork = Wrapper<K>::is_complex ? new underlying_type<K>[2 * n] : nullptr;
            {
                Lapack<K>::geev("N", "V", &dim, nullptr, &ldh, nullptr, nullptr, nullptr, &i__1, nullptr, &dim, vr, &lwork, nullptr, &info);
                vr[1] = std::max(static_cast<int>(std::real(*vr)), Wrapper<K>::is_complex ? dim * dim : (dim * (dim + 2)));
                Lapack<K>::geqrf(&row, &bK, nullptr, &ldh, nullptr, vr, &lwork, &info);
                vr[1] = std::max(std::real(*vr), std::real(vr[1]));
                Lapack<K>::mqr("R", "N", &n, &row, &bK, nullptr, &ldh, nullptr, nullptr, &n, vr, &lwork, &info);
                lwork = std::max(std::real(*vr), std::real(vr[1]));
            }
            K* work = new K[lwork];
            Lapack<K>::geev("N", "V", &dim, *H, &ldh, w, w + dim, nullptr, &i__1, vr, &dim, work, &lwork, rwork, &info);
            std::vector<std::pair<unsigned short, underlying_type<K>>> q;
            q.reserve(dim);
            for(i = 0; i < dim; ++i) {
                underlying_type<K> magnitude = Wrapper<K>::is_complex ? std::norm(w[i]) : std::real(w[i] * w[i] + w[dim + i] * w[dim + i]);
                q.emplace_back(i, magnitude);
            }
            std::sort(q.begin(), q.end(), [](const std::pair<unsigned short, underlying_type<K>>& lhs, const std::pair<unsigned short, underlying_type<K>>& rhs) { return lhs.second < rhs.second; });
            info = std::accumulate(q.cbegin(), q.cbegin() + bK, 0, [](int a, const std::pair<unsigned short, underlying_type<K>>& b) { return a + b.first; });
            for(i = bK; info != (bK * (bK - 1)) / 2 && i < dim; ++i)
                info += q[i].first;
            int* perm = new int[i];
            for(unsigned short j = 0; j < i; ++j)
                perm[j] = q[j].first + 1;
            Lapack<K>::lapmt(&i__1, &dim, &(info = i), vr, &dim, perm);
            delete [] perm;
            delete [] rwork;
            delete [] w;
            Blas<K>::gemm("N", "N", &n, &bK, &dim, &(Wrapper<K>::d__1), v[(m + 1) * (variant == 'F')], &n, vr, &dim, &(Wrapper<K>::d__0), U, &n);
            Blas<K>::gemm("N", "N", &row, &bK, &dim, &(Wrapper<K>::d__1), *save, &ldh, vr, &dim, &(Wrapper<K>::d__0), *H, &ldh);
            delete [] vr;
            K* tau = new K[bK];
            Lapack<K>::geqrf(&row, &bK, *H, &ldh, tau, work, &lwork, &info);
            Lapack<K>::mqr("R", "N", &n, &row, &bK, *H, &ldh, tau, *v, &n, work, &lwork, &info);
            std::copy_n(*v, k * ldv, C);
            Blas<K>::trsm("R", "U", "N", "N", &n, &bK, &(Wrapper<K>::d__1), *H, &ldh, U, &n);
            delete [] tau;
            delete [] work;
        }
        else if(!opt.val<unsigned short>("recycle_same_system")) {
            std::copy_n(C, k * ldv, *v);
            int bK = deflated * k;
            K* prod = new K[bK * (dim + deflated + 1)];
            if(variant == 'F')
                std::copy_n(v[m + 1], k * ldv, U);
            info =  dim + deflated;
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &info, &bK, &n, &(Wrapper<K>::d__1), *v, &n, U, &n, &(Wrapper<K>::d__0), prod, &info);
            for(unsigned short nu = 0; nu < bK; ++nu)
                prod[bK * (dim + deflated) + nu] = Blas<K>::dot(&n, U + nu * n, &i__1, U + nu * n, &i__1);
            MPI_Allreduce(MPI_IN_PLACE, prod, bK * (dim + deflated + 1), Wrapper<K>::mpi_type(), MPI_SUM, comm);
            for(unsigned short nu = 0; nu < bK; ++nu) {
                prod[bK * (dim + deflated) + nu] = 1.0 / std::sqrt(std::real(prod[bK * (dim + deflated) + nu]));
                Blas<K>::scal(&n, prod + bK * (dim + deflated) + nu, U + nu * n, &i__1);
            }
            for(i = 0; i < m - k; ++i)
                for(unsigned short nu = 0; nu < deflated; ++nu)
                    std::fill(save[i] + nu + 2 + nu * ldh + (i + 1) * deflated, save[i] + (nu + 1) * ldh, K());
            K* A = new K[dim * dim];
            for(i = 0; i < bK; ++i)
                for(unsigned short j = 0; j < bK; ++j)
                    A[j + i * dim] = (i == j ? prod[bK * (dim + deflated) + i] * prod[bK * (dim + deflated) + i] : Wrapper<K>::d__0);
            int diff = dim - bK;
            Wrapper<K>::template omatcopy<'N'>(diff, bK, H[k], ldh, A + bK * dim, dim);
            info = dim;
            for(unsigned short nu = 0; nu < bK; ++nu)
                Blas<K>::scal(&diff, prod + bK * (dim + deflated) + nu, A + bK * dim + nu, &info);
            Wrapper<K>::template omatcopy<'C'>(diff, bK, A + bK * dim, info, A + bK, info);
            int row = diff + deflated;
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &diff, &bK, &(Wrapper<K>::d__1), H[k], &ldh, H[k], &ldh, &(Wrapper<K>::d__0), A + bK * dim + bK, &info);
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &diff, &row, &(Wrapper<K>::d__1), *save, &ldh, *save, &ldh, &(Wrapper<K>::d__1), A + bK * dim + bK, &info);
            K* B = new K[deflated * m * (dim + deflated)]();
            row = dim + deflated;
            for(i = 0; i < bK; ++i)
                std::transform(prod + i * (dim + deflated), prod + (i + 1) * (dim + deflated), B + i * (dim + deflated), [&](const K& u) { return prod[bK * (dim + deflated) + i] * u; });
            Wrapper<K>::template omatcopy<'C'>(diff, diff, *save, ldh, B + bK + bK * row, row);
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &bK, &row, &(Wrapper<K>::d__1), *save, &ldh, B + bK, &row, &(Wrapper<K>::d__0), *H + deflated * (k + 1), &ldh);
            Wrapper<K>::template omatcopy<'N'>(bK, diff, *H + deflated * (k + 1), ldh, B + bK, row);
            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &diff, &bK, &bK, &(Wrapper<K>::d__1), H[k], &ldh, B, &row, &(Wrapper<K>::d__1), B + bK, &row);
            for(i = 0; i < bK; ++i)
                Blas<K>::scal(&bK, prod + bK * (dim + deflated) + i, B + i, &row);
            int bDim = dim;
            K* alpha = new K[(2 + !Wrapper<K>::is_complex) * bDim];
            int lwork = -1;
            K* vr = new K[bDim * bDim];
            Lapack<K>::ggev("N", "V", &bDim, A, &bDim, B, &row, alpha, alpha + 2 * bDim, alpha + bDim, nullptr, &i__1, nullptr, &bDim, alpha, &lwork, nullptr, &info);
            lwork = std::real(*alpha);
            K* work = new K[Wrapper<K>::is_complex ? (lwork + 4 * bDim) : lwork];
            underlying_type<K>* rwork = reinterpret_cast<underlying_type<K>*>(work + lwork);
            Lapack<K>::ggev("N", "V", &bDim, A, &bDim, B, &row, alpha, alpha + 2 * bDim, alpha + bDim, nullptr, &i__1, vr, &bDim, work, &lwork, rwork, &info);
            std::vector<std::pair<unsigned short, underlying_type<K>>> q;
            q.reserve(bDim);
            for(i = 0; i < bDim; ++i) {
                underlying_type<K> magnitude = Wrapper<K>::is_complex ? std::norm(alpha[i] / alpha[bDim + i]) : std::real((alpha[i] * alpha[i] + alpha[2 * bDim + i] * alpha[2 * bDim + i]) / (alpha[bDim + i] * alpha[bDim + i]));
                q.emplace_back(i, magnitude);
            }
            std::sort(q.begin(), q.end(), [](const std::pair<unsigned short, underlying_type<K>>& lhs, const std::pair<unsigned short, underlying_type<K>>& rhs) { return lhs.second < rhs.second; });
            info = std::accumulate(q.cbegin(), q.cbegin() + bK, 0, [](int a, const std::pair<unsigned short, underlying_type<K>>& b) { return a + b.first; });
            for(i = bK; info != (bK * (bK - 1)) / 2 && i < bDim; ++i)
                info += q[i].first;
            int* perm = new int[i];
            std::transform(q.cbegin(), q.cbegin() + i, perm, [](const std::pair<unsigned short, underlying_type<K>>& u) { return u.first + 1; });
            decltype(q)().swap(q);
            Lapack<K>::lapmt(&i__1, &bDim, &(info = i), vr, &bDim, perm);
            row = diff + deflated;
            Blas<K>::gemm("N", "N", &row, &bK, &diff, &(Wrapper<K>::d__1), *save, &ldh, vr + bK, &bDim, &(Wrapper<K>::d__0), *H + bK, &ldh);
            Wrapper<K>::template omatcopy<'N'>(bK, bK, vr, bDim, *H, ldh);
            for(i = 0; i < bK; ++i)
                Blas<K>::scal(&bK, prod + bK * (dim + deflated) + i, *H + i, &ldh);
            Blas<K>::gemm("N", "N", &bK, &bK, &diff, &(Wrapper<K>::d__1), H[k], &ldh, vr + bK, &bDim, &(Wrapper<K>::d__1), *H, &ldh);
            row = dim + deflated;
            K* tau = new K[bK];
            *perm = -1;
            Lapack<K>::geqrf(&row, &bK, nullptr, &ldh, nullptr, work, perm, &info);
            Lapack<K>::mqr("R", "N", &n, &row, &bK, nullptr, &ldh, nullptr, nullptr, &n, work + 1, perm, &info);
            delete [] perm;
            if(std::real(work[0]) > (Wrapper<K>::is_complex ? (lwork + 4 * bDim) : lwork) || std::real(work[1]) > (Wrapper<K>::is_complex ? (lwork + 4 * bDim) : lwork)) {
                lwork = std::max(std::real(work[0]), std::real(work[1]));
                delete [] work;
                work = new K[lwork];
            }
            Lapack<K>::geqrf(&row, &bK, *H, &ldh, tau, work, &lwork, &info);
            Wrapper<K>::template omatcopy<'N'>(bK, n, U, n, v[(m + 1) * (variant == 'F')], n);
            Blas<K>::gemm("N", "N", &n, &bK, &bDim, &(Wrapper<K>::d__1), v[(m + 1) * (variant == 'F')], &n, vr, &bDim, &(Wrapper<K>::d__0), U, &n);
            Blas<K>::trsm("R", "U", "N", "N", &n, &bK, &(Wrapper<K>::d__1), *H, &ldh, U, &n);
            Wrapper<K>::template omatcopy<'N'>(bK, n, C, n, *v, n);
            Lapack<K>::mqr("R", "N", &n, &row, &bK, *H, &ldh, tau, *v, &n, work, &lwork, &info);
            Wrapper<K>::template omatcopy<'N'>(bK, n, *v, n, C, n);
            delete [] tau;
            delete [] work;
            delete [] vr;
            delete [] alpha;
            delete [] B;
            delete [] A;
            delete [] prod;
        }
        if(converged)
            break;
    }
    delete [] piv;
    A.clearBuffer(alloc);
    delete [] *H;
    delete [] *save;
    delete [] H;
    std::cout.unsetf(std::ios_base::scientific);
    if(j != 0) {
        if(verbosity > 0) {
            if(j != it + 1)
                std::cout << "BGCRODR converges after " << j << " iteration" << (j > 1 ? "s" : "") << std::endl;
            else
                std::cout << "BGCRODR does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
        }
        return std::min(j, it);
    }
    else
        return GCRODR(A, b, x, mu, comm);
}
} // HPDDM
#endif // _HPDDM_GCRODR_
