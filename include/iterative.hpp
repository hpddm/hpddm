 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2014-11-05

   Copyright (C) 2011-2014 Université de Grenoble
                 2015      Eidgenössische Technische Hochschule Zürich
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

#ifndef _HPDDM_ITERATIVE_
#define _HPDDM_ITERATIVE_

namespace HPDDM {
template<class K>
struct EmptyOperator {
    bool setBuffer(const int&, K* = nullptr, const int& = 0) const { return false; }
    template<bool = true> void start(const K* const, K* const, const unsigned short& = 1) const { }
    void clearBuffer(const bool) const { }
    const underlying_type<K>* getScaling() const { return nullptr; }

    const HPDDM::MatrixCSR<K>& _A;
    EmptyOperator(HPDDM::MatrixCSR<K>* A) : _A(*A) { }
    int getDof() const { return _A._n; }
    void GMV(const K* const in, K* const out, const int& mu = 1) const {
        HPDDM::Wrapper<K>::csrmm(_A._sym, &(_A._n), &mu, _A._a, _A._ia, _A._ja, in, out);
    }
};
/* Class: Iterative method
 *  A class that implements various iterative methods. */
class IterativeMethod {
    private:
        /* Function: allocate
         *  Allocates workspace arrays for <Iterative method::CG>. */
        template<class K, typename std::enable_if<!Wrapper<K>::is_complex>::type* = nullptr>
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
        template<class K, typename std::enable_if<Wrapper<K>::is_complex>::type* = nullptr>
        static void allocate(underlying_type<K>*& dir, K*& p, const int& n, const unsigned short extra = 0, const unsigned short it = 1) {
            if(extra == 0) {
                dir = new underlying_type<K>[2];
                p = new K[std::max(1, 4 * n)];
            }
            else {
                dir = new underlying_type<K>[1 + 2 * it];
                p = new K[std::max(1, (4 + extra * it) * n)];
            }
        }
        /* Function: updateSol
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
        template<class Operator, class K, class T>
        static void updateSol(const Operator& A, char variant, const int& n, K* const x, const K* const* const h, K* const s, T* const* const v, const short* const hasConverged, const int& mu, K* const work, const int& deflated = -1) {
            static_assert(std::is_same<K, typename std::remove_const<T>::type>::value, "Wrong types");
            computeMin(h, s, hasConverged, mu, deflated);
            addSol(A, variant, n, x, std::distance(h[0], h[1]) / std::abs(deflated), s, v, hasConverged, mu, work, deflated);
        }
        template<class K>
        static void computeMin(const K* const* const h, K* const s, const short* const hasConverged, const int& mu, const int& deflated = -1, const int& shift = 0) {
            int ldh = std::distance(h[0], h[1]) / std::abs(deflated);
            if(deflated != -1) {
                int dim = std::abs(*hasConverged) - shift;
                int info;
                Lapack<K>::trtrs("U", "N", "N", &dim, &deflated, *h + deflated * shift * (1 + ldh), &ldh, s, &ldh, &info);
            }
            else
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    int dim = std::abs(hasConverged[nu]);
                    if(dim != 0)
                        Blas<K>::trsv("U", "N", "N", &(dim -= shift), *h + shift * (1 + ldh) + (ldh / mu) * nu, &ldh, s + nu, &mu);
                }
        }
        template<class Operator, class K, class T>
        static void addSol(const Operator& A, char variant, const int& n, K* const x, const int& ldh, const K* const s, T* const* const v, const short* const hasConverged, const int& mu, K* const work, const int& deflated = -1) {
            static_assert(std::is_same<K, typename std::remove_const<T>::type>::value, "Wrong types");
            K* const correction = (variant == 'R' ? (std::is_const<T>::value ? (work + mu * n) : const_cast<K* const>(v[ldh / (deflated == -1 ? mu : deflated) - 1])) : work);
            if(deflated == -1) {
                int ldv = mu * n;
                if(variant == 'L') {
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        if(hasConverged[nu] != 0) {
                            int dim = std::abs(hasConverged[nu]);
                            Blas<K>::gemv("N", &n, &dim, &(Wrapper<K>::d__1), *v + nu * n, &ldv, s + nu, &mu, &(Wrapper<K>::d__1), x + nu * n, &i__1);
                        }
                }
                else {
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        int dim = std::abs(hasConverged[nu]);
                        Blas<K>::gemv("N", &n, &dim, &(Wrapper<K>::d__1), *v + nu * n, &ldv, s + nu, &mu, &(Wrapper<K>::d__0), work + nu * n, &i__1);
                    }
                    if(variant == 'R')
                        A.apply(work, correction, mu);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        if(hasConverged[nu] != 0)
                            Blas<K>::axpy(&n, &(Wrapper<K>::d__1), correction + nu * n, &i__1, x + nu * n, &i__1);
                }
            }
            else {
                int dim = *hasConverged;
                if(deflated == mu) {
                    if(variant == 'L')
                        Blas<K>::gemm("N", "N", &n, &mu, &dim, &(Wrapper<K>::d__1), *v, &n, s, &ldh, &(Wrapper<K>::d__1), x, &n);
                    else {
                        Blas<K>::gemm("N", "N", &n, &mu, &dim, &(Wrapper<K>::d__1), *v, &n, s, &ldh, &(Wrapper<K>::d__0), work, &n);
                        if(variant == 'R')
                            A.apply(work, correction, mu);
                        Blas<K>::axpy(&(dim = mu * n), &(Wrapper<K>::d__1), correction, &i__1, x, &i__1);
                    }
                }
                else {
                    Blas<K>::gemm("N", "N", &n, &deflated, &dim, &(Wrapper<K>::d__1), *v, &n, s, &ldh, &(Wrapper<K>::d__0), work, &n);
                    if(variant == 'R')
                        A.apply(work, correction, deflated);
                    Blas<K>::gemm("N", "N", &n, &(dim = mu - deflated), &deflated, &(Wrapper<K>::d__1), correction, &n, s + deflated * ldh, &ldh, &(Wrapper<K>::d__1), x + deflated * n, &n);
                    Blas<K>::axpy(&(dim = deflated * n), &(Wrapper<K>::d__1), correction, &i__1, x, &i__1);
                }
            }
        }
        template<class Operator, class K, class T>
        static void updateSolRecycling(const Operator& A, char variant, const int& n, K* const x, const K* const* const h, K* const s, K* const* const v, T* const norm, K* const C, K* const U, const short* const hasConverged, const int shift, const int mu, K* const work, const MPI_Comm& comm, const int& deflated = -1) {
            const int ldh = std::distance(h[0], h[1]) / std::abs(deflated);
            const int dim = ldh / (deflated == -1 ? mu : deflated);
            if(C != nullptr && U != nullptr) {
                computeMin(h, s + shift * (deflated == -1 ? mu : deflated), hasConverged, mu, deflated, shift);
                const int ldv = (deflated == -1 ? mu : deflated) * n;
                if(deflated == -1) {
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        if(std::abs(hasConverged[nu]) != 0) {
                            K alpha = norm[nu];
                            Blas<K>::gemv(&(Wrapper<K>::transc), &n, &shift, &alpha, C + nu * n, &ldv, v[shift] + nu * n , &i__1, &(Wrapper<K>::d__0), s + nu, &mu);
                        }
                    }
                    MPI_Allreduce(MPI_IN_PLACE, s, shift * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        if(std::abs(hasConverged[nu]) != 0) {
                            int diff = std::abs(hasConverged[nu]) - shift;
                            Blas<K>::gemv("N", &shift, &diff, &(Wrapper<K>::d__2), h[shift] + nu * dim, &ldh, s + shift * mu + nu, &mu, &(Wrapper<K>::d__1), s + nu, &mu);
                        }
                }
                else {
                    std::copy_n(v[shift], deflated * n, work);
                    Blas<K>::trmm("R", "U", "N", "N", &n, &deflated, &(Wrapper<K>::d__1), reinterpret_cast<K* const>(norm), &ldh, work, &n);
                    int bK = deflated * shift;
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &bK, &deflated, &n, &(Wrapper<K>::d__1), C, &n, work, &n, &(Wrapper<K>::d__0), s, &ldh);
                    for(unsigned short i = 0; i < deflated; ++i)
                        std::copy_n(s + i * ldh, bK, work + i * bK);
                    MPI_Allreduce(MPI_IN_PLACE, work, bK * deflated, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    for(unsigned short i = 0; i < deflated; ++i)
                        std::copy_n(work + i * bK, bK, s + i * ldh);
                    int diff = *hasConverged - deflated * shift;
                    Blas<K>::gemm("N", "N", &bK, &deflated, &diff, &(Wrapper<K>::d__2), h[shift], &ldh, s + shift * deflated, &ldh, &(Wrapper<K>::d__1), s, &ldh);
                }
                std::copy_n(U, shift * ldv, v[dim * (variant == 'F')]);
                addSol(A, variant, n, x, ldh, s, static_cast<const K* const* const>(v + dim * (variant == 'F')), hasConverged, mu, work, deflated);
            }
            else
                updateSol(A, variant, n, x, h, s, static_cast<const K* const* const>(v + dim * (variant == 'F')), hasConverged, mu, work, deflated);
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
            Blas<typename std::remove_pointer<T>::type>::axpy(n, a, *x, incx, *y, incy);
        }
        template<class K, class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static void axpy(const int* const, const K* const, const T* const, const int* const, T const, const int* const) { }
        template<class K, class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static void axpy(const int* const n, const K* const a, const T* const x, const int* const incx, T* const y, const int* const incy) {
            static_assert(std::is_same<T, K>::value, "Wrong types");
            Blas<T>::axpy(n, a, x, incx, y, incy);
        }
        template<class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static typename std::remove_pointer<T>::type dot(const int* const n, const T* const x, const int* const incx, const T* const y, const int* const incy) {
            return Blas<typename std::remove_pointer<T>::type>::dot(n, *x, incx, *y, incy) / 2.0;
        }
        template<class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static T dot(const int* const n, const T* const x, const int* const incx, const T* const y, const int* const incy) {
            return Blas<T>::dot(n, x, incx, y, incy);
        }
        template<class T, class U, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static void diag(const int&, const U* const* const, T* const, T* const = nullptr) { }
        template<class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static void diag(const int& n, const underlying_type<T>* const d, T* const in, T* const out = nullptr) {
            if(out)
                Wrapper<T>::diag(n, d, in, out);
            else
                Wrapper<T>::diag(n, d, in);
        }
        template<class K>
        static void localSquaredNorm(const K* const b, const unsigned int n, underlying_type<K>* const norm, const unsigned short mu = 1) {
            for(unsigned short nu = 0; nu < mu; ++nu) {
                norm[nu] = 0.0;
                for(unsigned int i = 0; i < n; ++i) {
                    if(std::abs(b[nu * n + i]) > HPDDM_PEN * HPDDM_EPS)
                        norm[nu] += std::norm(b[nu * n + i] / underlying_type<K>(HPDDM_PEN));
                    else
                        norm[nu] += std::norm(b[nu * n + i]);
                }
            }
        }
        /* Function: orthogonalization
         *
         *  Orthogonalizes a block of vectors against a contiguous set of block of vectors.
         *
         * Template Parameters:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *    K              - Scalar type.
         *
         * Parameters:
         *    id             - Type of orthogonalization procedure.
         *    n              - Size of the vectors to orthogonalize.
         *    k              - Size of the basis to orthogonalize against.
         *    mu             - Number of vectors in each block.
         *    B              - Pointer to the basis.
         *    v              - Input block of vectors.
         *    H              - Dot products.
         *    comm           - Global MPI communicator. */
        template<bool excluded, class K>
        static void orthogonalization(const char id, const int n, const int k, const int mu, const K* const B, K* const v, K* const H, const MPI_Comm& comm) {
            if(excluded) {
                std::fill_n(H, mu * k, K());
                if(id == 1)
                    for(unsigned short i = 0; i < k; ++i)
                        MPI_Allreduce(MPI_IN_PLACE, H + mu * i, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                else
                    MPI_Allreduce(MPI_IN_PLACE, H, mu * k, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            }
            else {
                if(id == 1)
                    for(unsigned short i = 0; i < k; ++i) {
                        for(unsigned short nu = 0; nu < mu; ++nu)
                            H[i * mu + nu] = Blas<K>::dot(&n, B + (i * mu + nu) * n, &i__1, v + nu * n, &i__1);
                        MPI_Allreduce(MPI_IN_PLACE, H + i * mu, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            K alpha = -H[i * mu + nu];
                            Blas<K>::axpy(&n, &alpha, B + (i * mu + nu) * n, &i__1, v + nu * n, &i__1);
                        }
                    }
                else {
                    int ldb = mu * n;
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::gemv(&(Wrapper<K>::transc), &n, &k, &(Wrapper<K>::d__1), B + nu * n, &ldb, v + nu * n, &i__1, &(Wrapper<K>::d__0), H + nu, &mu);
                    MPI_Allreduce(MPI_IN_PLACE, H, k * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::gemv("N", &n, &k, &(Wrapper<K>::d__2), B + nu * n, &ldb, H + nu, &mu, &(Wrapper<K>::d__1), v + nu * n, &i__1);
                }
            }
        }
        template<bool excluded, class K>
        static void blockOrthogonalization(const char id, const int n, const int k, const int mu, const K* const B, K* const v, K* const H, const int ldh, K* const work, const MPI_Comm& comm) {
            if(excluded) {
                std::fill_n(work, mu * mu * k, K());
                if(id == 1)
                    for(unsigned short i = 0; i < k; ++i)
                        MPI_Allreduce(MPI_IN_PLACE, work, mu * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                else
                    MPI_Allreduce(MPI_IN_PLACE, work, mu * mu * k, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            }
            else {
                if(id == 1) {
                    for(unsigned short i = 0; i < k; ++i) {
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &mu, &mu, &n, &(Wrapper<K>::d__1), B + i * mu * n, &n, v, &n, &(Wrapper<K>::d__0), work, &mu);
                        MPI_Allreduce(MPI_IN_PLACE, work, mu * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        Blas<K>::gemm("N", "N", &n, &mu, &mu, &(Wrapper<K>::d__2), B + i * mu * n, &n, work, &mu, &(Wrapper<K>::d__1), v, &n);
                        Wrapper<K>::template omatcopy<'N'>(mu, mu, work, mu, H + mu * i, ldh);
                    }
                }
                else {
                    int tmp = mu * k;
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &tmp, &mu, &n, &(Wrapper<K>::d__1), B, &n, v, &n, &(Wrapper<K>::d__0), work, &tmp);
                    MPI_Allreduce(MPI_IN_PLACE, work, mu * tmp, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    Blas<K>::gemm("N", "N", &n, &mu, &tmp, &(Wrapper<K>::d__2), B, &n, work, &tmp, &(Wrapper<K>::d__1), v, &n);
                    Wrapper<K>::template omatcopy<'N'>(mu, tmp, work, tmp, H, ldh);
                }
            }
        }
        /* Function: VR
         *  Computes the inverse of the upper triangular matrix of a QR decomposition using the Cholesky QR method. */
        template<bool excluded, class K>
        static void VR(const int n, const int k, const int mu, const K* const V, K* const R, const int ldr, const MPI_Comm& comm, K* work = nullptr) {
            const int ldv = mu * n;
            if(work == nullptr)
                work = R;
            for(unsigned short nu = 0; nu < mu; ++nu) {
                Blas<K>::herk("U", "C", &k, &n, &(Wrapper<underlying_type<K>>::d__1), V + nu * n, &ldv, &(Wrapper<underlying_type<K>>::d__0), work + nu * (k * (k + 1)) / 2, &k);
                for(unsigned short xi = 1; xi < k; ++xi)
                    std::copy_n(work + nu * (k * (k + 1)) / 2 + xi * k, xi + 1, work + nu * (k * (k + 1)) / 2 + (xi * (xi + 1)) / 2);
            }
            MPI_Allreduce(MPI_IN_PLACE, work, mu * (k * (k + 1)) / 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            for(unsigned short nu = mu; nu-- > 0; )
                for(unsigned short xi = k; xi > 0; --xi)
                    std::copy_backward(work + nu * (k * (k + 1)) / 2 + (xi * (xi - 1)) / 2, work + nu * (k * (k + 1)) / 2 + (xi * (xi + 1)) / 2, R + nu * k * k + xi * ldr - (ldr - xi));
        }
        /* Function: QR
         *  Computes a QR decomposition of a distributed matrix. */
        template<bool excluded, class K>
        static int QR(const char id, const int n, const int k, const int mu, K* const Q, K* const R, const int ldr, const MPI_Comm& comm, K* work = nullptr, bool update = true) {
            const int ldv = mu * n;
            if(id == 0) {
                VR<excluded>(n, k, mu, Q, R, ldr, comm, work);
                int info;
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    Lapack<K>::potrf("U", &k, R + nu * k * k, &ldr, &info);
                    if(info > 0)
                        return info;
                }
                if(update)
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::trsm("R", "U", "N", "N", &n, &k, &(Wrapper<K>::d__1), R + k * k * nu, &ldr, Q + nu * n, &ldv);

            }
            else {
                if(work == nullptr)
                    work = R;
                for(unsigned short xi = 0; xi < k; ++xi) {
                    if(xi > 0)
                        orthogonalization<excluded>(id - 1, n, xi, mu, Q, Q + xi * ldv, work + xi * k * mu, comm);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        work[xi * (k + 1) * mu + nu] = Blas<K>::dot(&n, Q + xi * ldv + nu * n, &i__1, Q + xi * ldv + nu * n, &i__1);
                    MPI_Allreduce(MPI_IN_PLACE, work + xi * (k + 1) * mu, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        work[xi * (k + 1) * mu + nu] = std::sqrt(work[xi * (k + 1) * mu + nu]);
                        if(std::real(work[xi * (k + 1) * mu + nu]) < HPDDM_EPS)
                            return 1;
                        K alpha = K(1.0) / work[xi * (k + 1) * mu + nu];
                        Blas<K>::scal(&n, &alpha, Q + xi * ldv + nu * n, &i__1);
                    }
                }
                if(work != R)
                    Wrapper<K>::template omatcopy<'N'>(k, k * mu, work, k * mu, R, ldr);
            }
            return 0;
        }
        /* Function: Arnoldi
         *  Computes one iteration of the Arnoldi method for generating one basis vector of a Krylov space. */
        template<bool excluded, class K>
        static void Arnoldi(const char id, const unsigned short m, K* const* const H, K* const* const v, K* const s, underlying_type<K>* const sn, const int n, const int i, const int mu, const MPI_Comm& comm, K* const* const save = nullptr, const unsigned short shift = 0) {
            orthogonalization<excluded>(id % 4, n, i + 1 - shift, mu, v[shift], v[i + 1], H[i] + shift * mu, comm);
            if(excluded) {
                std::fill_n(sn + i * mu, mu, underlying_type<K>());
                MPI_Allreduce(MPI_IN_PLACE, sn + i * mu, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
                std::fill_n(H[i] + (i + 1) * mu, mu, K());
            }
            else {
                for(unsigned short nu = 0; nu < mu; ++nu)
                    sn[i * mu + nu] = std::real(Blas<K>::dot(&n, v[i + 1] + nu * n, &i__1, v[i + 1] + nu * n, &i__1));
                MPI_Allreduce(MPI_IN_PLACE, sn + i * mu, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    H[i][(i + 1) * mu + nu] = std::sqrt(sn[i * mu + nu]);
                    if(i < m - 1)
                        std::for_each(v[i + 1] + nu * n, v[i + 1] + (nu + 1) * n, [&](K& y) { y /= H[i][(i + 1) * mu + nu]; });
                }
            }
            if(save)
                Wrapper<K>::template omatcopy<'T'>(i - shift + 2, mu, H[i] + shift * mu, mu, save[i - shift], m + 1);
            for(unsigned short k = shift; k < i; ++k) {
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    K gamma = Wrapper<K>::conj(H[k][(m + 1) * nu + k + 1]) * H[i][k * mu + nu] + sn[k * mu + nu] * H[i][(k + 1) * mu + nu];
                    H[i][(k + 1) * mu + nu] = -sn[k * mu + nu] * H[i][k * mu + nu] + H[k][(m + 1) * nu + k + 1] * H[i][(k + 1) * mu + nu];
                    H[i][k * mu + nu] = gamma;
                }
            }
            for(unsigned short nu = 0; nu < mu; ++nu) {
                const int tmp = 2;
                underlying_type<K> delta = Blas<K>::nrm2(&tmp, H[i] + i * mu + nu, &mu);
                sn[i * mu + nu] = std::real(H[i][(i + 1) * mu + nu]) / delta;
                H[i][(i + 1) * mu + nu] = H[i][i * mu + nu] / delta;
                H[i][i * mu + nu] = delta;
                s[(i + 1) * mu + nu] = -sn[i * mu + nu] * s[i * mu + nu];
                s[i * mu + nu] *= Wrapper<K>::conj(H[i][(i + 1) * mu + nu]);
            }
            if(mu > 1)
                Wrapper<K>::template imatcopy<'T'>(i + 2, mu, H[i], mu, m + 1);
        }
        /* Function: BlockArnoldi
         *  Computes one iteration of the Block Arnoldi method for generating one basis vector of a block Krylov space. */
        template<bool excluded, class K>
        static bool BlockArnoldi(const char id, const unsigned short m, K* const* const H, K* const* const v, K* const tau, K* const s, const int lwork, const int n, const int i, const int mu, K* const work, const MPI_Comm& comm, K* const* const save = nullptr, const unsigned short shift = 0) {
            int ldh = (m + 1) * mu;
            blockOrthogonalization<excluded>(id % 4, n, i + 1 - shift, mu, v[shift], v[i + 1], H[i] + shift * mu, ldh, work, comm);
            int info = QR<excluded>(id / 4, n, mu, 1, v[i + 1], H[i] + (i + 1) * mu, ldh, comm, work, i < m - 1);
            if(info > 0)
                return true;
            for(unsigned short nu = 0; nu < mu; ++nu)
                std::fill(H[i] + (i + 1) * mu + nu * ldh + nu + 1, H[i] + (nu + 1) * ldh, K());
            if(save)
                for(unsigned short nu = 0; nu < mu; ++nu)
                    std::copy_n(H[i] + shift * mu + nu * ldh, (i + 1 - shift) * mu + nu + 1, save[i - shift] + nu * ldh);
            int N = 2 * mu;
            for(unsigned short leading = shift; leading < i; ++leading)
                Lapack<K>::mqr("L", &(Wrapper<K>::transc), &N, &mu, &N, H[leading] + leading * mu, &ldh, tau + leading * N, H[i] + leading * mu, &ldh, work, &lwork, &info);
            Lapack<K>::geqrf(&N, &mu, H[i] + i * mu, &ldh, tau + i * N, work, &lwork, &info);
            Lapack<K>::mqr("L", &(Wrapper<K>::transc), &N, &mu, &N, H[i] + i * mu, &ldh, tau + i * N, s + i * mu, &ldh, work, &lwork, &info);
            return false;
        }
    public:
        /* Function: GMRES
         *
         *  Implements the GMRES.
         *
         * Template Parameters:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
         *    K              - Scalar type.
         *
         * Parameters:
         *    A              - Global operator.
         *    b              - Right-hand side(s).
         *    x              - Solution vector(s).
         *    mu             - Number of right-hand sides.
         *    comm           - Global MPI communicator. */
        template<bool excluded = false, class Operator = void, class K = double>
        static int GMRES(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm);
        template<bool excluded = false, class Operator = void, class K = double>
        static int BGMRES(const Operator&, const K* const, K* const, const int&, const MPI_Comm&);
        template<bool excluded = false, class Operator = void, class K = double>
        static int GCRODR(const Operator&, const K* const, K* const, const int&, const MPI_Comm&);
        template<bool excluded = false, class Operator = void, class K = double>
        static int BGCRODR(const Operator&, const K* const, K* const, const int&, const MPI_Comm&);
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
         *    b              - Right-hand side.
         *    x              - Solution vector.
         *    comm           - Global MPI communicator. */
        template<bool excluded = false, class Operator, class K>
        static int CG(const Operator& A, const K* const b, K* const x, const MPI_Comm& comm);
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
         *    b              - Right-hand side.
         *    x              - Solution vector.
         *    comm           - Global MPI communicator. */
        template<bool excluded = false, class Operator, class K>
        static int PCG(const Operator& A, const K* const b, K* const x, const MPI_Comm& comm);
        template<bool excluded = false, class Operator = void, class K = double, typename std::enable_if<!is_substructuring_method<Operator>::value>::type* = nullptr>
        static int solve(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
            const Option& opt = *Option::get();
            switch(opt.val<unsigned short>("krylov_method")) {
                case 4:  return HPDDM::IterativeMethod::template BGCRODR<excluded>(A, b, x, mu, comm); break;
                case 3:  return HPDDM::IterativeMethod::template GCRODR<excluded>(A, b, x, mu, comm); break;
                case 2:  return HPDDM::IterativeMethod::template CG<excluded>(A, b, x, comm); break;
                case 1:  return HPDDM::IterativeMethod::template BGMRES<excluded>(A, b, x, mu, comm); break;
                default: return HPDDM::IterativeMethod::template GMRES<excluded>(A, b, x, mu, comm);
            }
        }
        template<bool excluded = false, class Operator = void, class K = double, typename std::enable_if<is_substructuring_method<Operator>::value>::type* = nullptr>
        static int solve(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
            return HPDDM::IterativeMethod::template PCG<excluded>(A, b, x, comm);
        }
};
} // HPDDM
#endif // _HPDDM_ITERATIVE_
