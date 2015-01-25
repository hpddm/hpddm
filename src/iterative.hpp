/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2014-11-05

   Copyright (C) 2011-2014 Université de Grenoble

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

namespace HPDDM {
/* Class: Iterative method
 *  A class that implements various iterative methods. */
class IterativeMethod {
    private:
        /* Function: allocate
         *  Allocates workspace arrays for <Iterative method::CG>. */
        template<class K, typename std::enable_if<std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static inline void allocate(K*& dir, K*& p, const int& n) {
            dir = new K[3 + 4 * n];
            p = dir + 3;
        }
        template<class K, typename std::enable_if<!std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static inline void allocate(typename Wrapper<K>::ul_type*& dir, K*& p, const int& n) {
            static_assert(std::is_same<K, std::complex<typename Wrapper<K>::ul_type>>::value, "Wrong types");
            dir = new typename Wrapper<K>::ul_type[3];
            p = new K[4 * n];
        }
        /* Function: depenalize
         *  Divides a scalar by <HPDDM_PEN>. */
        template<class K, typename std::enable_if<std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static inline void depenalize(const K& b, K& x) {
            x = b / HPDDM_PEN;
        }
        template<class K, typename std::enable_if<!std::is_same<K, typename Wrapper<K>::ul_type>::value>::type* = nullptr>
        static inline void depenalize(const K& b, K& x) {
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
         *    n              - Size of the vector.
         *    x              - Solution vector.
         *    k              - Dimension of the Hessenberg matrix.
         *    h              - Hessenberg matrix.
         *    s              - Coefficients in the Krylov subspace.
         *    v              - Basis of the Krylov subspace. */
        template<class K>
        static inline void update(const int& n, K* const x, const int& k, const K* const* const h, K* const s, const K* const* const v) {
            for(int i = k; i-- > 0; ) {
                K alpha = -(s[i] /= h[i][i]);
                Wrapper<K>::axpy(&i, &alpha, h[i], &i__1, s, &i__1);
            }
            Wrapper<K>::gemm(&transa, &transa, &n, &i__1, &k, &(Wrapper<K>::d__1), *v, &n, s, &k, &(Wrapper<K>::d__1), x, &n);
        }
        template<class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static inline void clean(T* const& pt) {
            delete [] *pt;
            delete []  pt;
        }
        template<class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static inline void clean(T* const& pt) {
            delete [] pt;
        }
        template<class K, class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static inline void axpy(const int* const n, const K* const a, const T* const x, const int* const incx, T* const y, const int* const incy) {
            static_assert(std::is_same<typename std::remove_pointer<T>::type, K>::value, "Wrong types");
            Wrapper<typename std::remove_pointer<T>::type>::axpy(n, a, *x, incx, *y, incy);
        }
        template<class K, class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static inline void axpy(const int* const, const K* const, const T* const, const int* const, T const, const int* const) { }
        template<class K, class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static inline void axpy(const int* const n, const K* const a, const T* const x, const int* const incx, T* const y, const int* const incy) {
            static_assert(std::is_same<T, K>::value, "Wrong types");
            Wrapper<T>::axpy(n, a, x, incx, y, incy);
        }
        template<class T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static inline typename std::remove_pointer<T>::type dot(const int* const n, const T* const x, const int* const incx, const T* const y, const int* const incy) {
            return Wrapper<typename std::remove_pointer<T>::type>::dot(n, *x, incx, *y, incy) / 2.0;
        }
        template<class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static inline T dot(const int* const n, const T* const x, const int* const incx, const T* const y, const int* const incy) {
            return Wrapper<T>::dot(n, x, incx, y, incy);
        }
        template<class T, class U, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
        static inline void diagv(const int&, const U* const* const, T* const, T* const = nullptr) { }
        template<class T, typename std::enable_if<!std::is_pointer<T>::value>::type* = nullptr>
        static inline void diagv(const int& n, const typename Wrapper<T>::ul_type* const d, T* const in, T* const out = nullptr) {
            if(out)
                Wrapper<T>::diagv(n, d, in, out);
            else
                Wrapper<T>::diagv(n, d, in);
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
         *    x              - Solution vector.
         *    b              - Right-hand side.
         *    m              - Maximum size of the Krylov subspace.
         *    it             - Maximum number of iterations.
         *    tol            - Tolerance for relative residual decrease.
         *    comm           - Global MPI communicator.
         *    verbosity      - Level of verbosity. */
        template<Gmres Type = CLASSICAL, bool excluded = false, class Operator, class K>
        static inline int GMRES(const Operator& A, K* const x, const K* const b,
                                unsigned short& m, unsigned short& it, typename Wrapper<K>::ul_type tol,
                                const MPI_Comm& comm, unsigned short verbosity) {
            const int n = excluded ? 0 : A.getDof();
            m = std::min(m, it);
            K* const storage = new K[3 * (m + 1) + 2 * n];
            K* s = storage;
            K* cs = storage + m + 1;
            K* sn = storage + 2 * m + 2;
            K* r = storage + 3 * m + 3;
            K* Ax = r + n;
            double timing[2];
            timing[0] = timing[1] = MPI_Wtime();
            std::copy(b, b + n, Ax);
            A.template apply<excluded>(Ax, r);
            timing[1] -= MPI_Wtime();
            storage[0] = Wrapper<K>::dot(&n, r, &i__1, r, &i__1);

            if(!excluded) {
                for(unsigned int i = 0; i < n; ++i)
                    if(std::abs(b[i]) > HPDDM_PEN * HPDDM_EPS)
                        depenalize(b[i], x[i]);
                A.GMV(x, Ax);
            }
            Wrapper<K>::axpby(n, 1.0, b, 1, -1.0, Ax, 1);
            timing[1] += MPI_Wtime();
            A.template apply<excluded>(Ax, r);
            timing[1] -= MPI_Wtime();
            storage[1] = Wrapper<K>::dot(&n, r, &i__1, r, &i__1);
            MPI_Allreduce(MPI_IN_PLACE, storage, 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);

            typename Wrapper<K>::ul_type norm = std::sqrt(std::real(storage[0]));
            typename Wrapper<K>::ul_type beta = std::sqrt(std::real(storage[1]));

            if(norm < HPDDM_EPS)
                norm = 1.0;
            if(std::abs(tol) < std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon()) {
                if(verbosity)
                    std::cout << "WARNING -- the tolerance of the iterative method was set to " << tol << " which is lower than the machine epsilon for type " << demangle(typeid(typename Wrapper<K>::ul_type).name()) << ", forcing the tolerance to " << 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon() << std::endl;
                tol = 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon();
            }
            if(beta / norm < tol) {
                if(norm > 1.0 / HPDDM_EPS)
                    norm = 1.0;
                else {
                    it = 0;
                    delete [] storage;
                    return 0;
                }
            }

            K** const v = new K*[(m + 1 + (Type == FUSED || Type == CLASSICAL)) * (1 + (Type != CLASSICAL))];
            if(!excluded) {
                *v = new K[(m + 1 + (Type == FUSED || Type == CLASSICAL)) * (1 + (Type != CLASSICAL)) * n + (m + 2) * (m + 2) * (Type == FUSED)]();
                for(unsigned short i = 1; i < (m + 1 + (Type == FUSED || Type == CLASSICAL)); ++i)
                    v[i] = *v + i * n;
                if(Type != CLASSICAL)
                    for(unsigned short i = 0; i < (m + 1 + (Type == FUSED)); ++i)
                        v[m + 1 + (Type == FUSED) + i] = v[m + (Type == FUSED)] + i * (n + (m + 2) * (Type == FUSED));
            }

            K** const H = new K*[m];
            if(Type != FUSED || excluded) {
                *H = new K[(m + 1) * m];
                for(unsigned short i = 1; i < m; ++i)
                    H[i] = *H + i * (m + 1);
            }
            else
                for(unsigned short i = 0; i < m; ++i)
                    H[i] = v[m + i + 3] + n;

            unsigned short j = 1;
            int i;
            if(Type != CLASSICAL)
                it = std::max(it, static_cast<unsigned short>((it / m) * m + 3));
            while(j < it) {
                Wrapper<K>::axpby(n, 1.0 / beta, r, 1, 0.0, v[0], 1);
                s[0] = beta;
                MPI_Request rq;
                if(Type != CLASSICAL)
                    std::copy(*v, *v + n, v[m + 1]);
                for(i = 0; i < m && j <= it; ++i, ++j) {
#if (OMPI_MAJOR_VERSION > 1 || (OMPI_MAJOR_VERSION == 1 && OMPI_MINOR_VERSION >= 7)) || MPICH_NUMVERSION >= 30000000
                    if(Type == PIPELINED && i > 0)
                        MPI_Iallreduce(MPI_IN_PLACE, H[i - 1], i + 1 - (i == 1), Wrapper<K>::mpi_type(), MPI_SUM, comm, &rq);
#endif
                    if(!excluded)
                        A.GMV(v[i + (Type != CLASSICAL) * (m + 1)], Ax);
                    timing[1] += MPI_Wtime();
                    A.template apply<excluded>(Ax, v[m + (Type == CLASSICAL ? 1 : i + 2)], Type == FUSED ? i + (i > 1) : 0);
                    timing[1] -= MPI_Wtime();
                    if(Type == CLASSICAL || i > 1) {
                        if(Type == CLASSICAL) {
                            if(excluded) {
                                std::fill(H[i], H[i] + i + 1, 0.0);
                                MPI_Allreduce(MPI_IN_PLACE, H[i], i + 1, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                                beta = 0.0;
                                MPI_Allreduce(MPI_IN_PLACE, &beta, 1, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                                H[i][i + 1] = std::sqrt(beta);
                            }
                            else {
                                int i_ = i + 1;
                                Wrapper<K>::gemv(&(Wrapper<K>::transc), &n, &i_, &(Wrapper<K>::d__1), *v, &n, v[m + 1], &i__1, &(Wrapper<K>::d__0), H[i], &i__1);
                                MPI_Allreduce(MPI_IN_PLACE, H[i], i + 1, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                                Wrapper<K>::gemv(&transa, &n, &i_, &(Wrapper<K>::d__2), *v, &n, H[i], &i__1, &(Wrapper<K>::d__1), v[m + 1], &i__1);
                                beta = Wrapper<K>::dot(&n, v[m + 1], &i__1, v[m + 1], &i__1);
                                MPI_Allreduce(MPI_IN_PLACE, &beta, 1, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                                H[i][i + 1] = std::sqrt(beta);
                                Wrapper<K>::axpby(n, K(1.0) / H[i][i + 1], v[m + 1], 1, 0.0, v[i + 1], 1);
                            }
                        }
                        else {
                            if(Type == PIPELINED)
                                MPI_Wait(&rq, MPI_STATUS_IGNORE);
                            H[i - 2][i - 1] = std::sqrt(H[i - 1][i]);
                            i -= 2;
                        }
                        for(unsigned short k = 0; k < i; ++k) {
                            K gamma = cs[k] * H[i][k] + sn[k] * H[i][k + 1];
                            H[i][k + 1] = -sn[k] * H[i][k] + cs[k] * H[i][k + 1];
                            H[i][k] = gamma;
                        }
                        int two = 2;
                        K delta = Wrapper<K>::nrm2(&two, H[i] + i, &i__1); // std::sqrt(H[i][i] * H[i][i] + H[i][i + 1] * H[i][i + 1]);
                        cs[i] = H[i][i] / delta;
                        sn[i] = H[i][i + 1] / delta;
                        H[i][i] = cs[i] * H[i][i] + sn[i] * H[i][i + 1];
                        s[i + 1] = -sn[i] * s[i];
                        s[i] *= cs[i];
                        if(verbosity)
                            std::cout << "GMRES: " << std::setw(3) << j << " " << std::scientific << std::abs(s[i + 1]) << " " <<  norm << " " <<  std::abs(s[i + 1]) / norm << " < " << tol << std::endl;
                        if(std::abs(s[i + 1]) / norm <= tol) {
                            timing[0] -= MPI_Wtime();
                            break;
                        }
                        if(Type != CLASSICAL) {
                            ++i;
                            delta = K(1.0) / H[i - 1][i];
                            Wrapper<K>::scal(&n, &delta, v[i], &i__1);
                            if(Type == FUSED) {
                                Wrapper<K>::scal(&n, &delta, v[m + i + 2], &i__1);
                                Wrapper<K>::scal(&n, &delta, v[m + i + 3], &i__1);
                            }
                            else {
                                two = 2 * n;
                                Wrapper<K>::scal(&two, &delta, v[m + i + 2], &i__1);
                            }
                            two = i + 1;
                            Wrapper<K>::scal(&two, &delta, H[i], &i__1);
                            H[i][i] /= H[i - 1][i];
                            ++i;
                        }
                    }
                    if(Type != CLASSICAL) {
                        if(Type == PIPELINED && i == 1)
                            MPI_Wait(&rq, MPI_STATUS_IGNORE);
                        int lda = n + (m + 2) * (Type == FUSED);
                        Wrapper<K>::gemv(&transa, &n, &i, &(Wrapper<K>::d__2), v[m + 2], &lda, H[i - 1], &i__1, &(Wrapper<K>::d__1), v[m + i + 2], &i__1);
                        if(i > 0) {
                            std::copy(v[m + i + 1], v[m + i + 1] + n, v[i]);
                            Wrapper<K>::gemv(&transa, &n, &i, &(Wrapper<K>::d__2), *v, &n, H[i - 1], &i__1, &(Wrapper<K>::d__1), v[i], &i__1);
                            H[i][i + 1] = Wrapper<K>::dot(&n, v[i], &i__1, v[i], &i__1);
                        }
                        lda = i + 1;
                        Wrapper<K>::gemv(&transb, &n, &lda, &(Wrapper<K>::d__1), *v, &n, v[m + i + 2], &i__1, &(Wrapper<K>::d__0), H[i], &i__1);
                    }
                }
                if(j != it + 1 && i != m)
                    break;
                else if(i == m) {
                    if(j == it + 1) {
                        --i;
                        break;
                    }
                    else {
                        if(Type != CLASSICAL)
                            i -= 2;
                        if(!excluded) {
                            update(n, x, i, H, s, v);
                            A.GMV(x, Ax);
                        }
                        Wrapper<K>::axpby(n, 1.0, b, 1, -1.0, Ax, 1);
                        timing[1] += MPI_Wtime();
                        A.template apply<excluded>(Ax, r);
                        timing[1] -= MPI_Wtime();
                        beta = Wrapper<K>::dot(&n, r, &i__1, r, &i__1);
                        MPI_Allreduce(MPI_IN_PLACE, &beta, 1, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                        beta = std::sqrt(beta);
                        if(verbosity)
                            std::cout << "GMRES restart(" << m << "): " << j - 1 << " " << beta << " " <<  norm << " " <<  beta / norm << " < " << tol << std::endl;
                    }
                }
            }
            if(i == m && j != it + 1) {
                --i;
                if(Type != CLASSICAL)
                    i -= 2;
            }
            if(!excluded)
                update(n, x, i + 1, H, s, v);
            it = j;
            if(verbosity) {
                if(std::abs(s[i + 1]) / norm <= tol)
                    std::cout << "GMRES converges after " << j << " iteration" << (j > 1 ? "s" : "") << " in " << -timing[0] << ". Time spent preconditioning: " << -timing[1] << std::endl;
                else
                    std::cout << "GMRES does not converges after " << j - 1 << " iteration" << (j > 2 ? "s" : "") << std::endl;
            }
            if(!excluded)
                delete [] *v;
            delete [] v;
            if(Type != FUSED || excluded)
                delete [] *H;
            delete [] H;
            delete [] storage;
            return 0;
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
         *    it             - Maximum number of iterations.
         *    tol            - Tolerance for relative residual decrease.
         *    comm           - Global MPI communicator.
         *    verbosity      - Level of verbosity. */
        template<bool excluded = false, class Operator, class K>
        static inline int CG(Operator& A, K* const x, const K* const b,
                             unsigned short& it, typename Wrapper<K>::ul_type tol,
                             const MPI_Comm& comm, unsigned short verbosity) {
            const int n = A.getDof();
            typename Wrapper<K>::ul_type* dir;
            K* p;
            allocate(dir, p, n);
            K* z = p + n;
            K* r = p + 2 * n;
            K* trash = p + 3 * n;
            const typename Wrapper<K>::ul_type* const d = A.getScaling();

            for(unsigned int i = 0; i < n; ++i)
                if(std::abs(b[i]) > HPDDM_PEN * HPDDM_EPS)
                    depenalize(b[i], x[i]);
            A.GMV(x, z);
            std::copy(b, b + n, r);
            Wrapper<K>::axpy(&n, &(Wrapper<K>::d__2), z, &i__1, r, &i__1);

            A.apply(r, z);

            Wrapper<K>::diagv(n, d, z, p);
            dir[0] = Wrapper<K>::dot(&n, z, &i__1, p, &i__1);
            MPI_Allreduce(MPI_IN_PLACE, dir, 1, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
            typename Wrapper<K>::ul_type resInit = std::sqrt(dir[0]);

            if(std::abs(tol) < std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon()) {
                if(verbosity)
                    std::cout << "WARNING -- the tolerance of the iterative method was set to " << tol << " which is lower than the machine epsilon for type " << demangle(typeid(typename Wrapper<K>::ul_type).name()) << ", forcing the tolerance to " << 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon() << std::endl;
                tol = 2 * std::numeric_limits<typename Wrapper<K>::ul_type>::epsilon();
            }
            if(resInit <= tol)
                it = 0;

            std::copy(z, z + n, p);
            unsigned short i = 0;
            while(i++ < it) {
                Wrapper<K>::diagv(n, d, r, trash);
                dir[0] = Wrapper<K>::dot(&n, z, &i__1, trash, &i__1);
                A.GMV(p, z);
                Wrapper<K>::diagv(n, d, p, trash);
                dir[1] = Wrapper<K>::dot(&n, z, &i__1, trash, &i__1);
                MPI_Allreduce(MPI_IN_PLACE, dir, 2, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                K alpha = dir[0] / dir[1];
                Wrapper<K>::axpy(&n, &alpha, p, &i__1, x, &i__1);
                alpha = -alpha;
                Wrapper<K>::axpy(&n, &alpha, z, &i__1, r, &i__1);

                A.apply(r, z);
                Wrapper<K>::diagv(n, d, z, trash);
                dir[1] = Wrapper<K>::dot(&n, r, &i__1, trash, &i__1);
                dir[2] = Wrapper<K>::dot(&n, z, &i__1, trash, &i__1);
                MPI_Allreduce(MPI_IN_PLACE, dir + 1, 2, Wrapper<typename Wrapper<K>::ul_type>::mpi_type(), MPI_SUM, comm);
                Wrapper<K>::axpby(n, 1.0, z, 1, dir[1] / dir[0], p, 1);

                dir[0] = std::sqrt(dir[2]);
                if(verbosity)
                    std::cout << "CG: " << std::setw(3) << i << " " << std::scientific << dir[0] << " " << resInit << " " << dir[0] / resInit << " < " << tol << std::endl;
                if(dir[0] / resInit <= tol) {
                    it = i;
                    break;
                }
            }
            if(verbosity) {
                if(dir[0] / resInit <= tol)
                    std::cout << "CG converges after " << i << " iteration" << (i > 1 ? "s" : "") << std::endl;
                else
                    std::cout << "CG does not converges after " << i - 1 << " iteration" << (i > 2 ? "s" : "") << std::endl;
            }
            delete [] dir;
            if(!std::is_same<K, typename Wrapper<K>::ul_type>::value)
                delete [] p;
            return 0;
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
         *    it             - Maximum number of iterations.
         *    tol            - Tolerance for relative residual decrease.
         *    comm           - Global MPI communicator.
         *    verbosity      - Level of verbosity. */
        template<bool excluded = false, class Operator, class K>
        static inline int PCG(Operator& A, K* const x, const K* const f,
                              unsigned short& it, typename Wrapper<K>::ul_type tol,
                              const MPI_Comm& comm, unsigned short verbosity) {
            typedef typename std::conditional<std::is_pointer<typename std::remove_reference<decltype(*A.getScaling())>::type>::value, K**, K*>::type ptr_type;
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
                if(verbosity)
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

            K* alpha = new K[2 * it];
            unsigned short i = 0;
            typename Wrapper<K>::ul_type resRel = std::numeric_limits<typename Wrapper<K>::ul_type>::max();
            while(i++ < it) {
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
                        diagv(n, m, pCurr, zCurr);
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
                    std::fill(alpha + it, alpha + it + i, 0.0);
                    MPI_Allreduce(MPI_IN_PLACE, alpha + it, i - 1, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    std::fill(alpha + i - 1, alpha + it + i + 2, 0.0);
                    MPI_Allreduce(MPI_IN_PLACE, alpha + i - 1, 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    A.template project<excluded, 'T'>(storage[0]);
                }
                A.template computeDot<excluded>(&resRel, zCurr, zCurr, comm);
                resRel = std::sqrt(resRel);
                if(verbosity)
                    std::cout << "CG: " << std::setw(3) << i << " " << std::scientific << resRel << " " << resInit << " " << resRel / resInit << " < " << tol << std::endl;
                if(resRel / resInit <= tol) {
                    it = i;
                    break;
                }
                if(!excluded) {
                    A.allocateSingle(pCurr);
                    p.emplace_back(pCurr);
                    diagv(n, m, z[i - 1]);
                }
            }
            if(verbosity) {
                if(resRel / resInit <= tol)
                    std::cout << "CG converges after " << i << " iteration" << (i > 1 ? "s" : "") << std::endl;
                else
                    std::cout << "CG does not converges after " << i - 1 << " iteration" << (i > 2 ? "s" : "") << std::endl;
            }
            if(std::is_same<ptr_type, K*>::value)
                A.template computeSolution<excluded>(x, f);
            else
                A.template computeSolution<excluded>(x, storage[1]);
            delete [] alpha;
            for(auto zCurr : z)
                clean(zCurr);
            for(auto pCurr : p)
                clean(pCurr);
            clean(storage[0]);
            return 0;
        }
};
} // HPDDM
#endif // _ITERATIVE_
