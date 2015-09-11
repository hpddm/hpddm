/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@inf.ethz.ch>
        Date: 2014-11-05

   Copyright (C) 2011-2014 Université de Grenoble
                 2015      Eidgenössische Technische Hochschule Zürich

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

#ifndef _ITERATIVE_
#define _ITERATIVE_

#include "ScaLAPACK.hpp"

namespace HPDDM {
/* Class: Iterative method
 *  A class that implements various iterative methods. */
class IterativeMethod {
    private:
        template<class K, typename std::enable_if<std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static K conj(K& x) { return x; }
        template<class K, typename std::enable_if<!std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static K conj(K& x) { return std::conj(x); }
        /* Function: allocate
         *  Allocates workspace arrays for <Iterative method::CG>. */
        template<class K, typename std::enable_if<std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static void allocate(K*& dir, K*& p, const int& n, const unsigned short extra = 0, const unsigned short it = 1) {
            if(extra == 0) {
                dir = new K[2 + std::max(1, 4 * n)];
                p = dir + 2;
            }
            else {
                dir = new K[1 + 2 * it + std::max(1, (4 + extra * it) * n)];
                p = dir + 1 + 2 * it;
            }
        }
        template<class K, typename std::enable_if<!std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static void allocate(typename Wrapper<K>::ul_type*& dir, K*& p, const int& n, const unsigned short extra = 0, const unsigned short it = 1) {
            static_assert(std::is_same<K, std::complex<typename Wrapper<K>::ul_type>>::value, "Wrong types");
            if(extra == 0) {
                dir = new typename Wrapper<K>::ul_type[2];
                p = new K[std::max(1, 4 * n)];
            }
            else {
                dir = new typename Wrapper<K>::ul_type[1 + 2 * it];
                p = new K[std::max(1, (4 + extra * it) * n)];
            }
        }
        /* Function: depenalize
         *  Divides a scalar by <HPDDM_PEN>. */
        template<class K, typename std::enable_if<std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static void depenalize(const K& b, K& x) {
            x = b / HPDDM_PEN;
        }
        template<class K, typename std::enable_if<!std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static void depenalize(const K& b, K& x) {
            static_assert(std::is_same<K, std::complex<typename Wrapper<K>::ul_type>>::value, "Wrong types");
            x = b / std::complex<typename Wrapper<K>::ul_type>(HPDDM_PEN, HPDDM_PEN);
        }
        /* Function: update
         *
         *  Updates a solution vector after convergence of <Iterative method::GMRES>.
         *
         * Template Parameter:
         *    K              - Scalar type.
         *
         * Parameters:
         *    variant        - Type of preconditioning.
         *    n              - Size of the vector.
         *    x              - Solution vector.
         *    k              - Dimension of the Hessenberg matrix.
         *    h              - Hessenberg matrix.
         *    s              - Coefficients in the Krylov subspace.
         *    v              - Basis of the Krylov subspace. */
        template<class Operator, class K>
        static void update(const Operator& A, char variant, const int& n, K* const x, const K* const* const h, K* const s, const K* const* const v, const short* const hasConverged, const int& mu = 1, bool block = false) {
            int tmp = std::distance(h[0], h[1]);
            if(mu == 1 || block) {
                int dim = std::abs(*hasConverged);
                int info;
                if(block)
                    tmp /= mu;
                Lapack<K>::trtrs("U", "N", "N", &dim, &mu, *h, &tmp, s, &tmp, &info);
            }
            else
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    for(int i = std::abs(hasConverged[nu]); i-- > 0; ) {
                        K alpha = -(s[i * mu + nu] /= h[i][i * mu + nu]);
                        Wrapper<K>::axpy(&i, &alpha, h[i] + nu, &mu, s + nu, &mu);
                    }
                }
            if(!block) {
                tmp = mu * n;
                if(variant == 'L') {
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        if(hasConverged[nu] != -1) {
                            int dim = std::abs(hasConverged[nu]);
                            Wrapper<K>::gemv(&transa, &n, &dim, &(Wrapper<K>::d__1), *v + nu * n, &tmp, s + nu, &mu, &(Wrapper<K>::d__1), x + nu * n, &i__1);
                        }
                }
                else {
                    K* work = new K[(1 + (variant == 'R')) * mu * n];
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        int dim = std::abs(hasConverged[nu]);
                        Wrapper<K>::gemv(&transa, &n, &dim, &(Wrapper<K>::d__1), *v + nu * n, &tmp, s + nu, &mu, &(Wrapper<K>::d__0), work + nu * n, &i__1);
                    }
                    if(variant == 'R')
                        A.apply(work, work + mu * n, mu);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        if(hasConverged[nu] != -1)
                            Wrapper<K>::axpy(&n, &(Wrapper<K>::d__1), work + ((variant == 'R') * mu + nu) * n, &i__1, x + nu * n, &i__1);
                    delete [] work;
                }
            }
            else {
                int dim = *hasConverged;
                if(variant == 'L')
                    Wrapper<K>::gemm(&transa, &transa, &n, &mu, &dim, &(Wrapper<K>::d__1), *v, &n, s, &tmp, &(Wrapper<K>::d__1), x, &n);
                else {
                    K* work = new K[(1 + (variant == 'R')) * mu * n];
                    Wrapper<K>::gemm(&transa, &transa, &n, &mu, &dim, &(Wrapper<K>::d__1), *v, &n, s, &tmp, &(Wrapper<K>::d__0), work, &n);
                    if(variant == 'R')
                        A.apply(work, work + mu * n, mu);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Wrapper<K>::axpy(&n, &(Wrapper<K>::d__1), work + ((variant == 'R') * mu + nu) * n, &i__1, x + nu * n, &i__1);
                    delete [] work;
                }
            }
        }
        template<class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static void clean(T* const& pt) {
            delete [] *pt;
            delete []  pt;
        }
        template<class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static void clean(T* const& pt) {
            delete [] pt;
        }
        template<class K, class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static void axpy(const int* const n, const K* const a, const T* const x, const int* const incx, T* const y, const int* const incy) {
            static_assert(std::is_same<typename std::remove_pointer<T>::type, K>::value, "Wrong types");
            Wrapper<typename std::remove_pointer<T>::type>::axpy(n, a, *x, incx, *y, incy);
        }
        template<class K, class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static void axpy(const int* const, const K* const, const T* const, const int* const, T const, const int* const) { }
        template<class K, class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static void axpy(const int* const n, const K* const a, const T* const x, const int* const incx, T* const y, const int* const incy) {
            static_assert(std::is_same<T, K>::value, "Wrong types");
            Wrapper<T>::axpy(n, a, x, incx, y, incy);
        }
        template<class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static typename std::remove_pointer<T>::type dot(const int* const n, const T* const x, const int* const incx, const T* const y, const int* const incy) {
            return Wrapper<typename std::remove_pointer<T>::type>::dot(n, *x, incx, *y, incy) / 2.0;
        }
        template<class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static T dot(const int* const n, const T* const x, const int* const incx, const T* const y, const int* const incy) {
            return Wrapper<T>::dot(n, x, incx, y, incy);
        }
        template<class T, class U, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static void diag(const int&, const U* const* const, T* const, T* const = nullptr) { }
        template<class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static void diag(const int& n, const typename Wrapper<T>::ul_type* const d, T* const in, T* const out = nullptr) {
            if(out)
                Wrapper<T>::diag(n, d, in, out);
            else
                Wrapper<T>::diag(n, d, in);
        }
    public:
        /* Function: GMRES
         *
         *  Implements the GMRES.
         *
         * Template Parameters:
         *    Type           - See <Gmres>.
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *    K              - Scalar type.
         *
         * Parameters:
         *    A              - Global operator.
         *    x              - Solution vector(s).
         *    b              - Right-hand side(s).
         *    mu             - Number of right-hand sides.
         *    comm           - Global MPI communicator. */
        template<bool excluded = false, class Operator = void, class K = double>
        static int GMRES(const Operator& A, K* const x, const K* const b, const int& mu, const MPI_Comm& comm) {
            const Option& opt = *Option::get();
            const unsigned short it = opt["max_it"];
            typename Wrapper<K>::ul_type tol = opt["tol"];
            const int n = excluded ? 0 : A.getDof();
            const unsigned short m = std::min(static_cast<unsigned short>(opt["gmres_restart"]), it);
            K* const s = new K[mu * ((m + 1) * (m + 1) + m + 2 * n) + (std::is_same<K, typename Wrapper<K>::ul_type>::value ? (mu * m + 2 * mu) : ((mu * m) / 2 + mu + 1))];
            K* cs = s + mu * (m + 1);
            typename Wrapper<K>::ul_type* sn = reinterpret_cast<typename Wrapper<K>::ul_type*>(cs + mu * m);
            typename Wrapper<K>::ul_type* norm = sn + mu * m;
            typename Wrapper<K>::ul_type* beta = norm + mu;
            K* r = cs + mu * m + (std::is_same<K, typename Wrapper<K>::ul_type>::value ? (mu * m + 2 * mu) : ((mu * m) / 2 + mu + 1));
            K* Ax = r + mu * n;
            A.template apply<excluded>(b, r, mu, Ax);
            for(unsigned short nu = 0; nu < mu; ++nu)
                norm[nu] = std::real(Wrapper<K>::dot(&n, r + nu * n, &i__1, r + nu * n, &i__1));

            const char variant = (opt["variant"] == 0 ? 'L' : opt["variant"] == 1 ? 'R' : 'F');
            if(!excluded) {
                for(unsigned short nu = 0; nu < mu; ++nu)
                    for(unsigned int i = 0; i < n; ++i)
                        if(std::abs(b[nu * n + i]) > HPDDM_PEN * HPDDM_EPS)
                            depenalize(b[nu * n + i], x[nu * n + i]);
                A.GMV(x, variant == 'L' ? Ax : r, mu);
            }
            Wrapper<K>::axpby(mu * n, 1.0, b, 1, -1.0, variant == 'L' ? Ax : r, 1);
            if(variant == 'L')
                A.template apply<excluded>(Ax, r, mu);
            for(unsigned short nu = 0; nu < mu; ++nu)
                beta[nu] = std::real(Wrapper<K>::dot(&n, r + nu * n, &i__1, r + nu * n, &i__1));
            MPI_Allreduce(MPI_IN_PLACE, norm, 2 * mu, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);

            if(std::abs(tol) < std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon()) {
                if(opt.set("verbosity"))
                    std::cout << "WARNING -- the tolerance of the iterative method was set to " << tol << " which is lower than the machine epsilon for type " << demangle(typeid(typename Wrapper<K>::ul_type).name()) << ", forcing the tolerance to " << 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon() << std::endl;
                tol = 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon();
            }

            short* const hasConverged = new short[mu];
            std::fill_n(hasConverged, mu, -m);
            std::for_each(norm, norm + 2 * mu, [](typename Wrapper<K>::ul_type& b) { b = std::sqrt(b); });
            for(unsigned short nu = 0; nu < mu; ++nu) {
                if(norm[nu] < HPDDM_EPS)
                    norm[nu] = 1.0;
                if(tol > 0.0 && beta[nu] / norm[nu] < tol) {
                    if(norm[nu] > 1.0 / HPDDM_EPS)
                        norm[nu] = 1.0;
                    else
                        hasConverged[nu] = 0;
                }
                else if(beta[nu] < -tol)
                    hasConverged[nu] = 0;
            }
            if(std::find(hasConverged, hasConverged + mu, -m) == hasConverged + mu) {
                delete [] hasConverged;
                delete [] s;
                return 0;
            }

            K** const v = new K*[m * (1 + (variant == 'F')) + std::max(static_cast<unsigned short>(2), m)];
            K** const H = v + m * (1 + (variant == 'F'));
            if(!excluded) {
                *v = new K[m * (1 + (variant == 'F')) * mu * n]();
                for(unsigned short i = 1; i < m * (1 + (variant == 'F')); ++i)
                    v[i] = *v + i * mu * n;
            }

            *H = Ax + mu * n;
            if(m < 2)
                H[1] = *H + (m + 1) * mu;
            else
                for(unsigned short i = 1; i < m; ++i)
                    H[i] = *H + i * (m + 1) * mu;

            unsigned short j = 1;
            while(j <= it) {
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    Wrapper<K>::axpby(n, 1.0 / beta[nu], r + nu * n, 1, K(), *v + nu * n, 1);
                    s[nu] = beta[nu];
                }
                int i = 0;
                while(i < m && j <= it) {
                    if(variant == 'L') {
                        if(!excluded)
                            A.GMV(v[i], Ax, mu);
                        A.template apply<excluded>(Ax, r, mu);
                    }
                    else {
                        A.template apply<excluded>(v[i], variant == 'F' ? v[i + m] : Ax, mu, r);
                        if(!excluded)
                            A.GMV(variant == 'F' ? v[i + m] : Ax, r, mu);
                    }
                    if(excluded) {
                        std::fill_n(H[i], mu * (i + 1), K());
                        if(opt["gs"] == 1)
                            for(int k = 0; k < i + 1; ++k)
                                MPI_Allreduce(MPI_IN_PLACE, H[i] + mu * k, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        else
                            MPI_Allreduce(MPI_IN_PLACE, H[i], mu * (i + 1), Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        std::fill_n(beta, mu, typename Wrapper<K>::ul_type());
                        MPI_Allreduce(MPI_IN_PLACE, beta, mu, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                        std::transform(beta, beta + mu, H[i] + (i + 1) * mu, [](const typename Wrapper<K>::ul_type& b) { return std::sqrt(b); });
                    }
                    else {
                        if(opt["gs"] == 1)
                            for(unsigned short k = 0; k < i + 1; ++k) {
                                for(unsigned short nu = 0; nu < mu; ++nu)
                                    H[i][k * mu + nu] = Wrapper<K>::dot(&n, v[k] + nu * n, &i__1, r + nu * n, &i__1);
                                MPI_Allreduce(MPI_IN_PLACE, H[i] + k * mu, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                                std::transform(H[i] + k * mu, H[i] + (k + 1) * mu, H[i] + (i + 1) * mu, [](const K& h) { return -h; });
                                for(unsigned short nu = 0; nu < mu; ++nu)
                                    Wrapper<K>::axpy(&n, H[i] + (i + 1) * mu + nu, v[k] + nu * n, &i__1, r + nu * n, &i__1);
                            }
                        else {
                            int tmp[2] { i + 1, mu * n };
                            for(unsigned short nu = 0; nu < mu; ++nu)
                                Wrapper<K>::gemv(&(Wrapper<K>::transc), &n, tmp, &(Wrapper<K>::d__1), *v + nu * n, tmp + 1, r + nu * n, &i__1, &(Wrapper<K>::d__0), H[i] + nu, &mu);
                            MPI_Allreduce(MPI_IN_PLACE, H[i], (i + 1) * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                            if(opt["gs"] == 0)
                                for(unsigned short nu = 0; nu < mu; ++nu)
                                    Wrapper<K>::gemv(&transa, &n, tmp, &(Wrapper<K>::d__2), *v + nu * n, tmp + 1, H[i] + nu, &mu, &(Wrapper<K>::d__1), r + nu * n, &i__1);
                            else
                                for(unsigned short nu = 0; nu < mu; ++nu)
                                    Wrapper<K>::axpby(n, -H[i][i * mu + nu], v[i] + nu * n, 1, 1.0, r + nu * n, 1);
                        }
                        for(unsigned short nu = 0; nu < mu; ++nu)
                            beta[nu] = std::real(Wrapper<K>::dot(&n, r + nu * n, &i__1, r + nu * n, &i__1));
                        MPI_Allreduce(MPI_IN_PLACE, beta, mu, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            H[i][(i + 1) * mu + nu] = std::sqrt(beta[nu]);
                            if(i < m - 1)
                                Wrapper<K>::axpby(n, K(1.0) / H[i][(i + 1) * mu + nu], r + nu * n, 1, 0.0, v[i + 1] + nu * n, 1);
                        }
                    }
                    for(unsigned short k = 0; k < i; ++k) {
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            K gamma = conj(cs[k * mu + nu]) * H[i][k * mu + nu] + sn[k * mu + nu] * H[i][(k + 1) * mu + nu];
                            H[i][(k + 1) * mu + nu] = -sn[k * mu + nu] * H[i][k * mu + nu] + cs[k * mu + nu] * H[i][(k + 1) * mu + nu];
                            H[i][k * mu + nu] = gamma;
                        }
                    }
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        const int tmp = 2;
                        typename Wrapper<K>::ul_type delta = Wrapper<K>::nrm2(&tmp, H[i] + i * mu + nu, &mu); // std::sqrt(H[i][i] * H[i][i] + H[i][i + 1] * H[i][i + 1]);
                        cs[i * mu + nu] = H[i][i * mu + nu] / delta;
                        sn[i * mu + nu] = std::real(H[i][(i + 1) * mu + nu]) / delta;
                        H[i][i * mu + nu] = conj(cs[i * mu + nu]) * H[i][i * mu + nu] + sn[i * mu + nu] * H[i][(i + 1) * mu + nu];
                        s[(i + 1) * mu + nu] = -sn[i * mu + nu] * s[i * mu + nu];
                        s[i * mu + nu] *= conj(cs[i * mu + nu]);
                        if(hasConverged[nu] == -m && ((tol > 0 && std::abs(s[(i + 1) * mu + nu]) / norm[nu] <= tol) || (tol < 0 && std::abs(s[(i + 1) * mu + nu]) <= -tol)))
                            hasConverged[nu] = i + 1;
                    }
                    if(opt.set("verbosity")) {
                        int tmp[2] { 0, 0 };
                        *beta = std::abs(s[(i + 1) * mu]);
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            if(hasConverged[nu] != -m)
                                ++tmp[0];
                            else if(std::abs(s[(i + 1) * mu + nu]) > *beta) {
                                *beta = std::abs(s[(i + 1) * mu + nu]);
                                tmp[1] = nu;
                            }
                        }
                        if(tol > 0)
                            std::cout << "GMRES: " << std::setw(3) << j << " " << std::scientific << *beta << " " <<  norm[tmp[1]] << " " <<  *beta / norm[tmp[1]] << " < " << tol;
                        else
                            std::cout << "GMRES: " << std::setw(3) << j << " " << std::scientific << *beta << " < " << -tol;
                        if(mu > 1) {
                            std::cout << " (rhs #" << tmp[1] + 1;
                            if(tmp[0] > 0)
                                std::cout << ", " << tmp[0] << " converged rhs";
                            std::cout << ")";
                        }
                        std::cout << std::endl;
                    }
                    if(std::find(hasConverged, hasConverged + mu, -m) == hasConverged + mu)
                        break;
                    else
                        ++i, ++j;
                }
                if(j != it + 1 && i == m) {
                    if(!excluded) {
                        update(A, variant, n, x, H, s, v + m * (variant == 'F'), hasConverged, mu);
                        A.GMV(x, variant == 'L' ? Ax : r, mu);
                    }
                    Wrapper<K>::axpby(mu * n, 1.0, b, 1, -1.0, variant == 'L' ? Ax : r, 1);
                    if(variant == 'L')
                        A.template apply<excluded>(Ax, r, mu);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        beta[nu] = std::real(Wrapper<K>::dot(&n, r + nu * n, &i__1, r + nu * n, &i__1));
                        if(hasConverged[nu] > 0)
                            hasConverged[nu] = -1;
                    }
                    MPI_Allreduce(MPI_IN_PLACE, beta, mu, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                    std::for_each(beta, beta + mu, [](typename Wrapper<K>::ul_type& b) { b = std::sqrt(b); });
                    if(opt.set("verbosity"))
                        std::cout << "GMRES restart(" << m << ")" << std::endl;
                }
                else
                    break;
            }
            if(!excluded)
                update(A, variant, n, x, H, s, v + m * (variant == 'F'), hasConverged, mu);
            if(opt.set("verbosity")) {
                if(j != it + 1)
                    std::cout << "GMRES converges after " << j << " iteration" << (j > 1 ? "s" : "") << std::endl;
                else
                    std::cout << "GMRES does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
            }
            if(!excluded)
                delete [] *v;
            delete [] v;
            delete [] hasConverged;
            delete [] s;
            return std::min(j, it);
        }
        /* Function: CG
         *
         *  Implements the CG method.
         *
         * Template Parameters:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *    K              - Scalar type.
         *
         * Parameters:
         *    A              - Global operator.
         *    x              - Solution vector.
         *    b              - Right-hand side.
         *    comm           - Global MPI communicator. */
        template<bool excluded = false, class Operator, class K>
        static int CG(Operator& A, K* const x, const K* const b, const MPI_Comm& comm) {
            const Option& opt = *Option::get();
            if(opt.any_of("schwarz_method", { 0, 1, 4 }) || opt.any_of("schwarz_coarse_correction", { 0 }))
                return GMRES(A, x, b, 1, comm);
            const unsigned short it = opt["max_it"];
            typename Wrapper<K>::ul_type tol = opt["tol"];
            const int n = A.getDof();
            typename Wrapper<K>::ul_type* dir;
            K* trash;
            allocate(dir, trash, n, opt["variant"] == 2 ? 2 : (opt["gs"] != 2 ? 1 : 0), it);
            K* z = trash + n;
            K* r = z + n;
            K* p = r + n;
            const typename Wrapper<K>::ul_type* const d = A.getScaling();

            for(unsigned int i = 0; i < n; ++i)
                if(std::abs(b[i]) > HPDDM_PEN * HPDDM_EPS)
                    depenalize(b[i], x[i]);
            A.GMV(x, z);
            std::copy_n(b, n, r);
            Wrapper<K>::axpy(&n, &(Wrapper<K>::d__2), z, &i__1, r, &i__1);

            A.apply(r, p, 1, z);

            Wrapper<K>::diag(n, d, p, trash);
            dir[0] = std::real(Wrapper<K>::dot(&n, trash, &i__1, p, &i__1));
            MPI_Allreduce(MPI_IN_PLACE, dir, 1, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
            typename Wrapper<K>::ul_type resInit = std::sqrt(dir[0]);

            if(std::abs(tol) < std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon()) {
                if(opt.set("verbosity"))
                    std::cout << "WARNING -- the tolerance of the iterative method was set to " << tol << " which is lower than the machine epsilon for type " << demangle(typeid(typename Wrapper<K>::ul_type).name()) << ", forcing the tolerance to " << 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon() << std::endl;
                tol = 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon();
            }
            if(resInit <= tol) {
                delete [] dir;
                if(!std::is_same<K, typename Wrapper<K>::ul_type>::value)
                    delete [] trash;
                return 0;
            }

            unsigned short i = 1;
            while(i <= it) {
                dir[0] = std::real(Wrapper<K>::dot(&n, r, &i__1, trash, &i__1));
                if(opt["variant"] == 2 && i > 1) {
                    for(unsigned short k = 0; k < i - 1; ++k)
                        dir[1 + k] = -std::real(Wrapper<K>::dot(&n, trash, &i__1, p + (1 + it + k) * n, &i__1)) / dir[1 + it + k];
                    MPI_Allreduce(MPI_IN_PLACE, dir + 1, i - 1, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                    std::copy_n(z, n, p);
                    for(unsigned short k = 0; k < i - 1; ++k) {
                        trash[0] = dir[1 + k];
                        Wrapper<K>::axpy(&n, trash, p + (1 + k) * n, &i__1, p, &i__1);
                    }
                }
                A.GMV(p, z);
                if(opt["gs"] != 2 && i > 1) {
                    Wrapper<K>::diag(n, d, z, trash);
                    for(unsigned short k = 0; k < i - 1; ++k)
                        dir[1 + k] = -std::real(Wrapper<K>::dot(&n, trash, &i__1, p + (1 + k) * n, &i__1)) / dir[1 + it + k];
                    MPI_Allreduce(MPI_IN_PLACE, dir + 1, i - 1, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                    for(unsigned short k = 0; k < i - 1; ++k) {
                        trash[0] = dir[1 + k];
                        Wrapper<K>::axpy(&n, trash, p + (1 + k) * n, &i__1, p, &i__1);
                    }
                    A.GMV(p, z);
                }
                Wrapper<K>::diag(n, d, p, trash);
                dir[1] = std::real(Wrapper<K>::dot(&n, z, &i__1, trash, &i__1));
                MPI_Allreduce(MPI_IN_PLACE, dir, 2, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                if(opt["gs"] != 2 || opt["variant"] == 2) {
                    dir[it + i] = dir[1];
                    std::copy_n(p, n, p + i * n);
                    if(opt["variant"] == 2)
                        std::copy_n(z, n, p + (it + i) * n);
                }
                trash[0] = dir[0] / dir[1];
                Wrapper<K>::axpy(&n, trash, p, &i__1, x, &i__1);
                trash[0] = -trash[0];
                Wrapper<K>::axpy(&n, trash, z, &i__1, r, &i__1);

                A.apply(r, z, 1, trash);
                Wrapper<K>::diag(n, d, z, trash);
                dir[1] = std::real(Wrapper<K>::dot(&n, r, &i__1, trash, &i__1)) / dir[0];
                dir[0] = std::real(Wrapper<K>::dot(&n, z, &i__1, trash, &i__1));
                MPI_Allreduce(MPI_IN_PLACE, dir, 2, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                if(opt["variant"] != 2)
                    Wrapper<K>::axpby(n, 1.0, z, 1, dir[1], p, 1);
                dir[0] = std::sqrt(dir[0]);
                if(opt.set("verbosity")) {
                    if(tol > 0)
                        std::cout << "CG: " << std::setw(3) << i << " " << std::scientific << dir[0] << " " << resInit << " " << dir[0] / resInit << " < " << tol << std::endl;
                    else
                        std::cout << "CG: " << std::setw(3) << i << " " << std::scientific << dir[0] << " < " << -tol << std::endl;
                }
                if((tol > 0.0 && dir[0] / resInit <= tol) || (tol < 0.0 && dir[0] <= -tol))
                    break;
                else
                    ++i;
            }
            if(opt.set("verbosity")) {
                if(i != it + 1)
                    std::cout << "CG converges after " << i << " iteration" << (i > 1 ? "s" : "") << std::endl;
                else
                    std::cout << "CG does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
            }
            delete [] dir;
            if(!std::is_same<K, typename Wrapper<K>::ul_type>::value)
                delete [] trash;
            return std::min(i, it);
        }
        /* Function: PCG
         *
         *  Implements the projected CG method.
         *
         * Template Parameters:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *    K              - Scalar type.
         *
         * Parameters:
         *    A              - Global operator.
         *    x              - Solution vector.
         *    f              - Right-hand side.
         *    comm           - Global MPI communicator. */
        template<bool excluded = false, class Operator, class K>
        static int PCG(Operator& A, K* const x, const K* const f, const MPI_Comm& comm) {
            typedef typename std::conditional<std::is_pointer<typename std::remove_reference<decltype(*A.getScaling())>::type>::value, K**, K*>::type ptr_type;
            const Option& opt = *Option::get();
            const unsigned short it = opt["max_it"];
            typename Wrapper<K>::ul_type tol = opt["tol"];
            const int n = std::is_same<ptr_type, K*>::value ? A.getDof() : A.getMult();
            const int offset = std::is_same<ptr_type, K*>::value ? A.getEliminated() : 0;
            ptr_type storage[std::is_same<ptr_type, K*>::value ? 1 : 2];
            // storage[0] = r
            // storage[1] = lambda
            A.allocateArray(storage);
            auto m = A.getScaling();
            if(std::is_same<ptr_type, K*>::value)
                A.template start<excluded>(x + offset, f, nullptr, storage[0]);
            else
                A.template start<excluded>(x, f, storage[1], storage[0]);

            if(std::abs(tol) < std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon()) {
                if(opt.set("verbosity"))
                    std::cout << "WARNING -- the tolerance of the iterative method was set to " << tol << " which is lower than the machine epsilon for type " << demangle(typeid(typename Wrapper<K>::ul_type).name()) << ", forcing the tolerance to " << 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon() << std::endl;
                tol = 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon();
            }
            std::vector<ptr_type> z;
            z.reserve(it);
            ptr_type zCurr;
            A.allocateSingle(zCurr);
            z.emplace_back(zCurr);
            if(!excluded)
                A.precond(storage[0], zCurr);                                                              //     z_0 = M r_0

            typename Wrapper<K>::ul_type resInit;
            A.template computeDot<excluded>(&resInit, zCurr, zCurr, comm);
            resInit = std::sqrt(resInit);

            std::vector<ptr_type> p;
            p.reserve(it);
            ptr_type pCurr;
            A.allocateSingle(pCurr);
            p.emplace_back(pCurr);

            K* alpha = new K[excluded ? std::max(static_cast<unsigned short>(2), it) : 2 * it];
            typename Wrapper<K>::ul_type resRel = std::numeric_limits<typename Wrapper<K>::ul_type>::max();
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
                if(opt.set("verbosity"))
                    std::cout << "CG: " << std::setw(3) << i << " " << std::scientific << resRel << " " << resInit << " " << resRel / resInit << " < " << tol << std::endl;
                if(resRel / resInit <= tol)
                    break;
                else
                    ++i;
                if(!excluded) {
                    A.allocateSingle(pCurr);
                    p.emplace_back(pCurr);
                    diag(n, m, z[i - 1]);
                }
            }
            if(opt.set("verbosity")) {
                if(i != it + 1)
                    std::cout << "CG converges after " << i << " iteration" << (i > 1 ? "s" : "") << std::endl;
                else
                    std::cout << "CG does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
            }
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
            return std::min(i, it);
        }
};
} // HPDDM
#endif // _ITERATIVE_
