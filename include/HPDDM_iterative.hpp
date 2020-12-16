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

#include "HPDDM_LAPACK.hpp"

#if !defined(_KSPIMPL_H)
#define HPDDM_CHKERRQ(ierr)    if((ierr) < 0) return (ierr)
#define HPDDM_TOL(tol, A)      tol
#define HPDDM_MAX_IT(max, A)   max
#define HPDDM_IT(i, A)         i
#define HPDDM_RET(i)           i
#else
#define HPDDM_CHKERRQ(ierr)    CHKERRQ(ierr)
#define HPDDM_TOL(tol, A)      ((A._ksp)->rtol)
#define HPDDM_MAX_IT(max, A)   ((A._ksp)->max_it)
#define HPDDM_IT(i, A)         ((A._ksp)->its)
#define HPDDM_RET(i)           0
#endif

namespace HPDDM {
template<class K, class T = int>
struct EmptyOperator : OptionsPrefix<K> {
    typedef T integer_type;
    const T _n;
    explicit EmptyOperator(T n) : OptionsPrefix<K>(), _n(n) { }
    constexpr T getDof() const { return _n; }
    static constexpr underlying_type<K>* getScaling() { return nullptr; }
    template<bool> static constexpr bool start(const K* const, K* const, const unsigned short& = 1) { return false; }
    static constexpr std::unordered_map<unsigned int, K> boundaryConditions() { return std::unordered_map<unsigned int, K>(); }
    static constexpr bool end(const bool) { return false; }
};
template<class Operator, class K, class T = int>
struct CustomOperator : EmptyOperator<K> {
    const Operator* const _A;
    CustomOperator(const Operator* const A, T n) : EmptyOperator<K, T>(n), _A(A) { }
    int GMV(const K* const in, K* const out, const int& mu = 1) const;
};
template<class K>
class CustomOperator<MatrixCSR<K>, K> : public EmptyOperator<K> {
    private:
        const MatrixCSR<K>* _A;
    public:
        explicit CustomOperator(const MatrixCSR<K>* const A) : EmptyOperator<K>(A ? A->_n : 0), _A(A) { }
        const MatrixCSR<K>* getMatrix() const { return _A; }
        void setMatrix(MatrixCSR<K>* const A) {
            if(A && A->_n == EmptyOperator<K>::_n) {
                if(_A)
                    delete _A;
                _A = A;
            }
        }
        int GMV(const K* const in, K* const out, const int& mu = 1) const {
            Wrapper<K>::csrmm(_A->_sym, &(EmptyOperator<K>::_n), &mu, _A->_a, _A->_ia, _A->_ja, in, out);
            return 0;
        }
};

/* Class: Iterative method
 *  A class that implements various iterative methods. */
class IterativeMethod {
    private:
        /* Function: outputResidual
         *  Prints information about the residual at a given iteration. */
        template<char T, class K>
        static void checkConvergence(const char verbosity, const unsigned short j, const unsigned short i, const underlying_type<K>& tol, const int& mu, const underlying_type<K>* const norm, const K* const res, short* const conv, const short sentinel) {
            for(unsigned short nu = 0; nu < mu; ++nu)
                if(conv[nu] == -sentinel && ((tol > 0.0 && std::abs(res[nu]) / norm[nu] <= tol) || (tol < 0.0 && std::abs(res[nu]) <= -tol)))
                    conv[nu] = i;
            if(verbosity > 2) {
#if !defined(HPDDM_PETSC)
                constexpr auto method = (T == 2 ? "CG" : (T == 4 ? "GCRODR" : "GMRES"));
                unsigned short tmp[2] { 0, 0 };
                underlying_type<K> beta = std::abs(res[0]);
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    if(conv[nu] != -sentinel)
                        ++tmp[0];
                    else if(std::abs(res[nu]) > beta) {
                        beta = std::abs(res[nu]);
                        tmp[1] = nu;
                    }
                }
                if(tol > 0.0)
                    std::cout << method << ": " << std::setw(3) << j << " " << beta << " " << norm[tmp[1]] << " " << beta / norm[tmp[1]] << " < " << tol;
                else
                    std::cout << method << ": " << std::setw(3) << j << " " << beta << " < " << -tol;
                if(mu > 1) {
                    std::cout << " (rhs #" << tmp[1] + 1;
                    if(tmp[0])
                        std::cout << ", " << tmp[0] << " converged rhs";
                    std::cout << ")";
                }
                std::cout << std::endl;
#else
                ignore(j);
#endif
            }
        }
        template<char T, class K>
        static unsigned short checkBlockConvergence(const char verbosity, const int i, const underlying_type<K>& tol, const int mu, const int d, const underlying_type<K>* const norm, const K* const res, const int ldh, K* const work, const unsigned short t) {
            underlying_type<K>* pt = reinterpret_cast<underlying_type<K>*>(work);
            unsigned short conv = 0;
            if(T == 3 || T == 6) {
                for(unsigned short nu = 0; nu < mu / t; ++nu) {
                    pt[nu] = std::sqrt(std::real(res[nu]));
                    if(((tol > 0.0 && pt[nu] / norm[nu] <= tol) || (tol < 0.0 && pt[nu] <= -tol)))
                        ++conv;
                }
            }
            else if(t <= 1) {
                conv = mu - d;
                for(unsigned short nu = 0; nu < d; ++nu) {
                    int dim = nu + 1;
                    pt[nu] = Blas<K>::nrm2(&dim, res + nu * ldh, &i__1);
                    if(((tol > 0.0 && pt[nu] / norm[nu] <= tol) || (tol < 0.0 && pt[nu] <= -tol)))
                        ++conv;
                }
            }
            else {
                std::fill_n(work, d, K());
                for(unsigned short nu = 0; nu < t; ++nu) {
                    int dim = nu + 1;
                    Blas<K>::axpy(&dim, &(Wrapper<K>::d__1), res + nu * ldh, &i__1, work, &i__1);
                }
                *pt = Blas<K>::nrm2(&d, work, &i__1);
                if(((tol > 0.0 && *pt / *norm <= tol) || (tol < 0.0 && *pt <= -tol)))
                    ++conv;
            }
            if(verbosity > 2) {
#if !defined(HPDDM_PETSC)
                constexpr auto method = (T == 3 ? "BCG" : (T == 5 ? "BGCRODR" : (T == 6 ? "BFBCG" : "BGMRES")));
                underlying_type<K>* max;
                if(tol > 0.0) {
                    unsigned short j = 0;
                    for(unsigned short k = 1; k < d / t; ++k) {
                        if(pt[j] / norm[j] < pt[k] / norm[k])
                            j = k;
                    }
                    max = pt + j;
                    std::cout << method << ": " << std::setw(3) << i << " " << *max << " " <<  norm[j] << " " <<  *max / norm[j] << " < " << tol;
                }
                else {
                    max = std::max_element(pt, pt + d / t);
                    std::cout << method << ": " << std::setw(3) << i << " " << *max << " < " << -tol;
                }
                if(d != t || t != mu) {
                    std::cout << " (rhs #" << std::distance(pt, max) + 1;
                    if(conv > d)
                        std::cout << ", " << t * conv - d << " converged rhs";
                    if(d != mu)
                        std::cout << ", " << mu - d << " deflated rhs";
                    std::cout << ")";
                }
                std::cout <<  std::endl;
#else
                ignore(i);
#endif
            }
            return t * conv;
        }
        template<char T>
        static void convergence(const char verbosity, const unsigned short i, const unsigned short m) {
            if(verbosity) {
                constexpr auto method = (T == 1 ? "BGMRES" : (T == 2 ? "CG" : (T == 3 ? "BCG" : (T == 4 ? "GCRODR" : (T == 5 ? "BGCRODR" : (T == 6 ? "BFBCG" : (T == 7 ? "PCG" : "GMRES")))))));
                if(i != m + 1)
                    std::cout << method << " converges after " << i << " iteration" << (i > 1 ? "s" : "") << std::endl;
                else
                    std::cout << method << " does not converge after " << m << " iteration" << (m > 1 ? "s" : "") << std::endl;
            }
        }
        template<char T, class K, class Operator>
        static void options(const Operator& A, K* const d, int* const i, unsigned short* const m, char* const id) {
            const std::string prefix = A.prefix();
#if !HPDDM_PETSC
            const Option& opt = *Option::get();
            m[T == 1 || T == 5 ? 2 : (T == 0 || T == 3 || T == 4 || T == 6 ? 1 : 0)] = std::min(opt.val<short>(prefix + "max_it", 100), std::numeric_limits<short>::max());
            if(T == 7) {
                d[0] = opt.val(prefix + "richardson_damping_factor", 1.0);
                return;
            }
            d[T == 1 || T == 5 || T == 6] = opt.val(prefix + "tol", 1.0e-6);
            id[0] = opt.val<char>(prefix + "verbosity", 0);
            if(T == 1 || T == 5 || T == 6) {
                d[0] = opt.val(prefix + "deflation_tol", -1.0);
                m[T != 6] = opt.val<unsigned short>(prefix + "enlarge_krylov_subspace", 1);
            }
            if(T == 0 || T == 1 || T == 4 || T == 5) {
                id[2] = opt.val<char>(prefix + "orthogonalization", HPDDM_ORTHOGONALIZATION_CGS) + (opt.val<char>(prefix + "qr", HPDDM_QR_CHOLQR) << 2);
                m[0] = std::min(static_cast<unsigned short>(std::numeric_limits<short>::max()), std::min(opt.val<unsigned short>(prefix + "gmres_restart", 40), m[T == 1 || T == 5 ? 2 : 1]));
            }
            if(T == 0 || T == 1 || T == 2 || T == 4 || T == 5)
                id[1] = opt.val<char>(prefix + "variant", HPDDM_VARIANT_RIGHT);
            if(T == 3 || T == 6)
                id[1] = opt.val<char>(prefix + "qr", HPDDM_QR_CHOLQR);
            if(T == 4 || T == 5) {
                *i = std::min(m[0] - 1, opt.val<int>(prefix + "recycle", 0));
                id[3] = opt.val<char>(prefix + "recycle_target", HPDDM_RECYCLE_TARGET_SM);
                id[4] = opt.val<char>(prefix + "recycle_strategy", HPDDM_RECYCLE_STRATEGY_A) + 4 * (std::min(opt.val<unsigned short>(prefix + "recycle_same_system"), static_cast<unsigned short>(2)));
            }
            if(std::abs(d[T == 1 || T == 5 || T == 6]) < std::numeric_limits<underlying_type<K>>::epsilon()) {
                if(id[0])
                    std::cout << "WARNING -- the tolerance of the iterative method was set to " << d[T == 1 || T == 5 || T == 6]
#if __cpp_rtti || defined(__GXX_RTTI) || defined(__INTEL_RTTI__) || defined(_CPPRTTI)
                     << " which is lower than the machine epsilon for type " << demangle(typeid(underlying_type<K>).name())
#endif
                     << ", forcing the tolerance to " << 4 * std::numeric_limits<underlying_type<K>>::epsilon() << std::endl;
                d[T == 1 || T == 5 || T == 6] = 4 * std::numeric_limits<underlying_type<K>>::epsilon();
            }
#endif
        }
        /* Function: allocate
         *  Allocates workspace arrays for <Iterative method::CG>. */
        template<class K, typename std::enable_if<!Wrapper<K>::is_complex>::type* = nullptr>
        static void allocate(K*& dir, K*& p, const int& n, const unsigned short extra = 0, const unsigned short it = 1, const unsigned short mu = 1) {
            if(extra == 0) {
                dir = new K[(3 + std::max(1, 4 * n)) * mu];
                p = dir + 3 * mu;
            }
            else {
                dir = new K[(3 + 2 * it + std::max(1, (4 + 2 * it) * n)) * mu];
                p = dir + (3 + 2 * it) * mu;
            }
        }
        template<class K, typename std::enable_if<Wrapper<K>::is_complex>::type* = nullptr>
        static void allocate(underlying_type<K>*& dir, K*& p, const int& n, const unsigned short extra = 0, const unsigned short it = 1, const unsigned short mu = 1) {
            if(extra == 0) {
                dir = new underlying_type<K>[3 * mu];
                p = new K[std::max(1, 4 * n) * mu];
            }
            else {
                dir = new underlying_type<K>[(3 + 2 * it) * mu];
                p = new K[std::max(1, (4 + 2 * it) * n) * mu];
            }
        }
        /* Function: updateSol
         *
         *  Updates a solution vector after convergence of <Iterative method::GMRES>.
         *
         * Template Parameters:
         *    excluded       - True if the master processes are excluded from the domain decomposition, false otherwise.
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
        template<bool excluded, class Operator, class K, class T>
        static int updateSol(const Operator& A, const char variant, const int& n, K* const x, const K* const* const h, K* const s, T* const* const v, const short* const hasConverged, const int& mu, K* const work, const int& deflated = -1) {
            static_assert(std::is_same<K, typename std::remove_const<T>::type>::value, "Wrong types");
            if(!excluded)
                computeMin(h, s, hasConverged, mu, deflated);
            return addSol<excluded>(A, variant, n, x, std::distance(h[0], h[1]) / std::abs(deflated), s, v, hasConverged, mu, work, deflated);
        }
        template<class K>
        static void computeMin(const K* const* const h, K* const s, const short* const hasConverged, const int& mu, const int& deflated = -1, const int& shift = 0) {
            int ldh = std::distance(h[0], h[1]) / std::abs(deflated);
            if(deflated != -1) {
                int dim = std::abs(*hasConverged) - deflated * shift;
                int info;
                Lapack<K>::trtrs("U", "N", "N", &dim, &deflated, *h + deflated * shift * (1 + ldh), &ldh, s, &ldh, &info);
            }
            else
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    int dim = std::abs(hasConverged[nu]);
                    if(dim)
                        Blas<K>::trsv("U", "N", "N", &(dim -= shift), *h + shift * (1 + ldh) + (ldh / mu) * nu, &ldh, s + nu, &mu);
                }
        }
        template<bool excluded, class Operator, class K, class T>
        static int addSol(const Operator& A, const char variant, const int& n, K* const x, const int& ldh, const K* const s, T* const* const v, const short* const hasConverged, const int& mu, K* const work, const int& deflated = -1) {
            static_assert(std::is_same<K, typename std::remove_const<T>::type>::value, "Wrong types");
            int ierr;
            K* const correction = (variant == HPDDM_VARIANT_RIGHT ? (std::is_const<T>::value ? (work + mu * n) : const_cast<K*>(v[ldh / (deflated == -1 ? mu : deflated) - 1])) : work);
            if(excluded || !n) {
                if(variant == HPDDM_VARIANT_RIGHT) {
                    ierr = A.template apply<excluded>(work, correction, deflated == -1 ? mu : deflated);HPDDM_CHKERRQ(ierr);
                }
            }
            else {
                if(deflated == -1) {
                    int ldv = mu * n;
                    if(variant == HPDDM_VARIANT_LEFT) {
                        for(unsigned short nu = 0; nu < mu; ++nu)
                            if(hasConverged[nu]) {
                                int dim = std::abs(hasConverged[nu]);
                                Blas<K>::gemv("N", &n, &dim, &(Wrapper<K>::d__1), *v + nu * n, &ldv, s + nu, &mu, &(Wrapper<K>::d__1), x + nu * n, &i__1);
                            }
                    }
                    else {
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            int dim = std::abs(hasConverged[nu]);
                            Blas<K>::gemv("N", &n, &dim, &(Wrapper<K>::d__1), *v + nu * n, &ldv, s + nu, &mu, &(Wrapper<K>::d__0), work + nu * n, &i__1);
                        }
                        if(variant == HPDDM_VARIANT_RIGHT) {
                            ierr = A.template apply<excluded>(work, correction, mu);HPDDM_CHKERRQ(ierr);
                        }
                        for(unsigned short nu = 0; nu < mu; ++nu)
                            if(hasConverged[nu])
                                Blas<K>::axpy(&n, &(Wrapper<K>::d__1), correction + nu * n, &i__1, x + nu * n, &i__1);
                    }
                }
                else {
                    int dim = *hasConverged;
                    if(deflated == mu) {
                        if(variant == HPDDM_VARIANT_LEFT)
                            Blas<K>::gemm("N", "N", &n, &mu, &dim, &(Wrapper<K>::d__1), *v, &n, s, &ldh, &(Wrapper<K>::d__1), x, &n);
                        else {
                            Blas<K>::gemm("N", "N", &n, &mu, &dim, &(Wrapper<K>::d__1), *v, &n, s, &ldh, &(Wrapper<K>::d__0), work, &n);
                            if(variant == HPDDM_VARIANT_RIGHT) {
                                ierr = A.template apply<excluded>(work, correction, mu);HPDDM_CHKERRQ(ierr);
                            }
                            Blas<K>::axpy(&(dim = mu * n), &(Wrapper<K>::d__1), correction, &i__1, x, &i__1);
                        }
                    }
                    else {
                        Blas<K>::gemm("N", "N", &n, &deflated, &dim, &(Wrapper<K>::d__1), *v, &n, s, &ldh, &(Wrapper<K>::d__0), work, &n);
                        if(variant == HPDDM_VARIANT_RIGHT) {
                            ierr = A.template apply<excluded>(work, correction, deflated);HPDDM_CHKERRQ(ierr);
                        }
                        Blas<K>::gemm("N", "N", &n, &(dim = mu - deflated), &deflated, &(Wrapper<K>::d__1), correction, &n, s + deflated * ldh, &ldh, &(Wrapper<K>::d__1), x + deflated * n, &n);
                        Blas<K>::axpy(&(dim = deflated * n), &(Wrapper<K>::d__1), correction, &i__1, x, &i__1);
                    }
                }
            }
            return 0;
        }
        template<bool excluded, class Operator, class K, class T>
        static int updateSolRecycling(const Operator& A, const char variant, const int& n, K* const x, const K* const* const h, K* const s, K* const* const v, T* const norm, const K* const C, const K* const U, const short* const hasConverged, const int shift, const int mu, K* const work, const MPI_Comm& comm, const int& deflated = -1) {
#if !HPDDM_PETSC
            const bool same = Option::get()->template val<unsigned short>(A.prefix("recycle_same_system"));
#else
            const bool same = false;
#endif
            const int ldh = std::distance(h[0], h[1]) / std::abs(deflated);
            const int dim = ldh / (deflated == -1 ? mu : deflated);
            if(C && U) {
                computeMin(h, s + shift * (deflated == -1 ? mu : deflated), hasConverged, mu, deflated, shift);
                const int ldv = (deflated == -1 ? mu : deflated) * n;
                if(deflated == -1) {
                    if(same)
                        std::fill_n(s, shift * mu, K());
                    else {
                        if(!excluded && n) {
                            K* const pt = A.getScaling() ? work : v[shift];
                            if(A.getScaling())
                                Wrapper<K>::diag(n, A.getScaling(), v[shift], pt, mu);
                            for(unsigned short nu = 0; nu < mu; ++nu) {
                                if(std::abs(hasConverged[nu])) {
                                    K alpha = norm[nu];
                                    Blas<K>::gemv(&(Wrapper<K>::transc), &n, &shift, &alpha, C + nu * n, &ldv, pt + nu * n, &i__1, &(Wrapper<K>::d__0), s + nu, &mu);
                                }
                            }
                        }
                        else
                            std::fill_n(s, shift * mu, K());
                        MPI_Allreduce(MPI_IN_PLACE, s, shift * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    }
                    if(!excluded && n)
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            if(std::abs(hasConverged[nu])) {
                                int diff = std::abs(hasConverged[nu]) - shift;
                                Blas<K>::gemv("N", &shift, &diff, &(Wrapper<K>::d__2), h[shift] + nu * dim, &ldh, s + shift * mu + nu, &mu, &(Wrapper<K>::d__1), s + nu, &mu);
                            }
                        }
                }
                else {
                    int bK = deflated * shift;
                    K beta = K();
                    if(!same) {
                        if(!excluded && n) {
                            std::copy_n(v[shift], deflated * n, work);
                            Blas<K>::trmm("R", "U", "N", "N", &n, &deflated, &(Wrapper<K>::d__1), reinterpret_cast<K*>(norm), &ldh, work, &n);
                            Wrapper<K>::diag(n, A.getScaling(), work, mu);
                            Blas<K>::gemm(&(Wrapper<K>::transc), "N", &bK, &deflated, &n, &(Wrapper<K>::d__1), C, &n, work, &n, &(Wrapper<K>::d__0), s, &ldh);
                            for(unsigned short i = 0; i < deflated; ++i)
                                std::copy_n(s + i * ldh, bK, work + i * bK);
                        }
                        else
                            std::fill_n(work, bK * deflated, K());
                        MPI_Allreduce(MPI_IN_PLACE, work, bK * deflated, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        for(unsigned short i = 0; i < deflated; ++i)
                            std::copy_n(work + i * bK, bK, s + i * ldh);
                        beta = Wrapper<K>::d__1;
                    }
                    int diff = *hasConverged - deflated * shift;
                    Blas<K>::gemm("N", "N", &bK, &deflated, &diff, &(Wrapper<K>::d__2), h[shift], &ldh, s + shift * deflated, &ldh, &beta, s, &ldh);
                }
                std::copy_n(U, shift * ldv, v[dim * (variant == HPDDM_VARIANT_FLEXIBLE)]);
                return addSol<excluded>(A, variant, n, x, ldh, s, static_cast<const K* const*>(v + dim * (variant == HPDDM_VARIANT_FLEXIBLE)), hasConverged, mu, work, deflated);
            }
            else
                return updateSol<excluded>(A, variant, n, x, h, s, static_cast<const K* const*>(v + dim * (variant == HPDDM_VARIANT_FLEXIBLE)), hasConverged, mu, work, deflated);
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
        template<bool excluded, class Operator, class K>
        static int initializeNorm(const Operator& A, const char variant, const K* const b, K* const x, K* const v, const int n, K* work, underlying_type<K>* const norm, const unsigned short mu, const unsigned short k, bool& allocate) {
            allocate = A.template start<excluded>(b, x, mu);
            const underlying_type<K>* const d = A.getScaling();
            if(variant == HPDDM_VARIANT_LEFT) {
                int ierr = A.template apply<excluded>(b, v, mu, work);HPDDM_CHKERRQ(ierr);
                if(d)
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        norm[nu / k] = 0.0;
                        for(int i = 0; i < n; ++i)
                            norm[nu / k] += d[i] * std::norm(v[i + nu * n]);
                    }
                else
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        norm[nu / k] = std::real(Blas<K>::dot(&n, v + nu * n, &i__1, v + nu * n, &i__1));
            }
            else {
                if(k <= 1)
                    work = const_cast<K*>(b);
                else {
                    std::fill_n(work, n, K());
                    for(unsigned short nu = 0; nu < k; ++nu)
                        Blas<K>::axpy(&n, &(Wrapper<K>::d__1), b + nu * n, &i__1, work, &i__1);
                }
                const std::unordered_map<unsigned int, K> map = A.boundaryConditions();
                for(unsigned short nu = 0; nu < mu / k; ++nu) {
                    norm[nu] = 0.0;
                    for(int i = 0; i < n; ++i) {
                        if(std::abs(work[nu * n + i]) > HPDDM_PEN * HPDDM_EPS && map.find(i) != map.cend())
                            norm[nu] += (d ? d[i] : 1.0) * std::norm(work[nu * n + i] / underlying_type<K>(HPDDM_PEN));
                        else
                            norm[nu] += (d ? d[i] : 1.0) * std::norm(work[nu * n + i]);
                    }
                }
            }
            return 0;
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
        static void orthogonalization(const char id, const int n, const int k, const int mu, const K* const B, K* const v, K* const H, const underlying_type<K>* const d, K* const work, const MPI_Comm& comm) {
            if(excluded || !n) {
                std::fill_n(H, k * mu, K());
                if(id == 1)
                    for(unsigned short i = 0; i < k; ++i)
                        MPI_Allreduce(MPI_IN_PLACE, H + i * mu, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                else
                    MPI_Allreduce(MPI_IN_PLACE, H, k * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            }
            else {
                if(id == 1) {
                    for(unsigned short i = 0; i < k; ++i) {
                        if(d)
                            for(unsigned short nu = 0; nu < mu; ++nu) {
                                H[i * mu + nu] = K();
                                for(int j = 0; j < n; ++j)
                                    H[i * mu + nu] += d[j] * Wrapper<K>::conj(B[(i * mu + nu) * n + j]) * v[nu * n + j];
                            }
                        else
                            for(unsigned short nu = 0; nu < mu; ++nu)
                                H[i * mu + nu] = Blas<K>::dot(&n, B + (i * mu + nu) * n, &i__1, v + nu * n, &i__1);
                        MPI_Allreduce(MPI_IN_PLACE, H + i * mu, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            K alpha = -H[i * mu + nu];
                            Blas<K>::axpy(&n, &alpha, B + (i * mu + nu) * n, &i__1, v + nu * n, &i__1);
                        }
                    }
                }
                else {
                    int ldb = mu * n;
                    K* const pt = d ? work : v;
                    if(d)
                        Wrapper<K>::diag(n, d, v, work, mu);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::gemv(&(Wrapper<K>::transc), &n, &k, &(Wrapper<K>::d__1), B + nu * n, &ldb, pt + nu * n, &i__1, &(Wrapper<K>::d__0), H + nu, &mu);
                    MPI_Allreduce(MPI_IN_PLACE, H, k * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::gemv("N", &n, &k, &(Wrapper<K>::d__2), B + nu * n, &ldb, H + nu, &mu, &(Wrapper<K>::d__1), v + nu * n, &i__1);
                }
            }
        }
        template<bool excluded, class K>
        static void blockOrthogonalization(const char id, const int n, const int k, const int mu, const K* const B, K* const v, K* const H, const int ldh, const underlying_type<K>* const d, K* const work, const MPI_Comm& comm) {
            if(excluded || !n) {
                std::fill_n(work, k * mu * mu, K());
                if(id == 1)
                    for(unsigned short i = 0; i < k; ++i) {
                        MPI_Allreduce(MPI_IN_PLACE, work, mu * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        Wrapper<K>::template omatcopy<'N'>(mu, mu, work, mu, H + mu * i, ldh);
                    }
                else {
                    MPI_Allreduce(MPI_IN_PLACE, work, k * mu * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    Wrapper<K>::template omatcopy<'N'>(mu, k * mu, work, k * mu, H, ldh);
                }
            }
            else {
                K* const pt = d ? work + k * mu * mu : v;
                if(id == 1) {
                    for(unsigned short i = 0; i < k; ++i) {
                        if(d)
                            Wrapper<K>::diag(n, d, v, pt, mu);
                        Blas<K>::gemm(&(Wrapper<K>::transc), "N", &mu, &mu, &n, &(Wrapper<K>::d__1), B + i * mu * n, &n, pt, &n, &(Wrapper<K>::d__0), work, &mu);
                        MPI_Allreduce(MPI_IN_PLACE, work, mu * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                        Blas<K>::gemm("N", "N", &n, &mu, &mu, &(Wrapper<K>::d__2), B + i * mu * n, &n, work, &mu, &(Wrapper<K>::d__1), v, &n);
                        Wrapper<K>::template omatcopy<'N'>(mu, mu, work, mu, H + mu * i, ldh);
                    }
                }
                else {
                    if(d)
                        Wrapper<K>::diag(n, d, v, pt, mu);
                    const int tmp = k * mu;
                    Blas<K>::gemm(&(Wrapper<K>::transc), "N", &tmp, &mu, &n, &(Wrapper<K>::d__1), B, &n, pt, &n, &(Wrapper<K>::d__0), work, &tmp);
                    MPI_Allreduce(MPI_IN_PLACE, work, mu * tmp, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    Blas<K>::gemm("N", "N", &n, &mu, &tmp, &(Wrapper<K>::d__2), B, &n, work, &tmp, &(Wrapper<K>::d__1), v, &n);
                    Wrapper<K>::template omatcopy<'N'>(mu, tmp, work, tmp, H, ldh);
                }
            }
        }
        /* Function: VR
         *  Computes the inverse of the upper triangular matrix of a QR decomposition using the Cholesky QR method. */
        template<bool excluded, class K>
        static void VR(const int n, const int k, const int mu, const K* const V, K* const R, const int ldr, const underlying_type<K>* const d, K* work, const MPI_Comm& comm) {
            K* const pt = work != R && ldr != k ? work + k * k * mu : work;
            const int ldv = mu * n;
            if(ldr == k)
                work = R;
            else if(mu > 1)
                std::cout << "WARNING -- not implemented" << std::endl;
            if(!excluded && n)
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    if(!d)
                        Blas<K>::herk("U", "C", &k, &n, &(Wrapper<underlying_type<K>>::d__1), V + nu * n, &ldv, &(Wrapper<underlying_type<K>>::d__0), work + nu * (k * (k + 1)) / 2, &k);
                    else {
                        if(mu == 1)
                            Wrapper<K>::diag(n, d, V, pt, k);
                        else {
                            for(unsigned short xi = 0; xi < k; ++xi)
                                Wrapper<K>::diag(n, d, V + nu * n + xi * ldv, pt + xi * n);
                        }
                        Blas<K>::gemmt("U", &(Wrapper<K>::transc), "N", &k, &n, &(Wrapper<K>::d__1), V + nu * n, &ldv, pt, &n, &(Wrapper<K>::d__0), work + nu * (k * (k + 1)) / 2, &k);
                    }
                    for(unsigned short xi = 1; xi < k; ++xi)
                        std::copy_n(work + nu * (k * (k + 1)) / 2 + xi * k, xi + 1, work + nu * (k * (k + 1)) / 2 + (xi * (xi + 1)) / 2);
                }
            else
                std::fill_n(work, mu * (k * (k + 1)) / 2, K());
            MPI_Allreduce(MPI_IN_PLACE, work, mu * (k * (k + 1)) / 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            for(unsigned short nu = mu; nu-- > 0; )
                for(unsigned short xi = k; xi > 0; --xi)
                    std::copy_backward(work + nu * (k * (k + 1)) / 2 + (xi * (xi - 1)) / 2, work + nu * (k * (k + 1)) / 2 + (xi * (xi + 1)) / 2, R + nu * k * k + xi * ldr - (ldr - xi));
        }
        template<bool excluded, class K>
        static void RRQR(const char id, const int n, const int k, K* const Q, K* const R, const underlying_type<K> tol, int& rank, int* const piv, const underlying_type<K>* const d, K* const work, const MPI_Comm& comm) {
            if(tol < -0.9)
                rank = QR<excluded>(id, n, k, Q, R, k, d, work, comm);
            else {
                VR<excluded>(n, k, 1, Q, R, k, d, work, comm);
                int info;
                Lapack<K>::pstrf("U", &k, R, &k, piv, &rank, &(Wrapper<underlying_type<K>>::d__0), reinterpret_cast<underlying_type<K>*>(work), &info);
                while(rank > 1 && std::abs(R[(rank - 1) * (k + 1)] / R[0]) <= tol)
                    --rank;
                Lapack<K>::lapmt(&i__1, &n, &k, Q, &n, piv);
                if(!excluded && n)
                    Blas<K>::trsm("R", "U", "N", "N", &n, &rank, &(Wrapper<K>::d__1), R, &k, Q, &n);
            }
        }
        template<char T, class K>
        static void diagonal(const char verbosity, const K* const R, const int k, const underlying_type<K> tol = -1.0, const int* const piv = nullptr) {
            if(verbosity > 3) {
                constexpr auto method = (T == 3 ? "BCG" : (T == 5 ? "BGCRODR" : (T == 6 ? "BFBCG" : "BGMRES")));
                if(tol < -0.9) {
                    std::cout << method << " diag(R), QR = block residual: ";
                    std::cout << *R;
                    if(k > 1) {
                        if(k > 2)
                            std::cout << "\t...";
                        std::cout << "\t" << R[(k - 1) * (k + 1)];
                    }
                    std::cout << std::endl;
                }
                else {
                    std::cout << method << " diag(R), QR = block residual, with pivoting: ";
                    std::cout << *R << " (" << *piv << ")";
                    if(k > 1) {
                        if(k > 2)
                            std::cout << "\t...";
                        std::cout << "\t" << R[(k - 1) * (k + 1)] << " (" << piv[k - 1] << ")";
                    }
                    std::cout << std::endl;
                }
            }
        }
        /* Function: QR
         *  Computes a QR decomposition of a distributed matrix. */
        template<bool excluded, class K>
        static int QR(const char id, const int n, const int k, K* const Q, K* const R, const int ldr, const underlying_type<K>* const d, K* work, const MPI_Comm& comm, bool update = true, const int mu = 1) {
            const int ldv = mu * n;
            int rank = k;
            if(id == HPDDM_QR_CHOLQR) {
                VR<excluded>(n, k, mu, Q, R, ldr, d, work, comm);
                int info;
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    Lapack<K>::potrf("U", &k, R + nu * k * ldr, &ldr, &info);
                    if(info > 0)
                        rank = info - 1;
                }
                if(!excluded && n && update)
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        Blas<K>::trsm("R", "U", "N", "N", &n, &rank, &(Wrapper<K>::d__1), R + nu * k * ldr, &ldr, Q + nu * n, &ldv);
            }
            else {
                if(!work)
                    work = R;
                K* const pt = (d || work == R) ? work + k * k * mu : work;
                for(unsigned short xi = 0; xi < rank; ++xi) {
                    if(xi > 0)
                        orthogonalization<excluded>(id - 1, n, xi, mu, Q, Q + xi * ldv, work + xi * k * mu, d, pt, comm);
                    if(d)
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            work[xi * (k + 1) * mu + nu] = K();
                            for(int j = 0; j < n; ++j)
                                work[xi * (k + 1) * mu + nu] += d[j] * std::norm(Q[xi * ldv + nu * n + j]);
                        }
                    else
                        for(unsigned short nu = 0; nu < mu; ++nu)
                            work[xi * (k + 1) * mu + nu] = Blas<K>::dot(&n, Q + xi * ldv + nu * n, &i__1, Q + xi * ldv + nu * n, &i__1);
                    MPI_Allreduce(MPI_IN_PLACE, work + xi * (k + 1) * mu, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        work[xi * (k + 1) * mu + nu] = std::sqrt(work[xi * (k + 1) * mu + nu]);
                        if(std::real(work[xi * (k + 1) * mu + nu]) < HPDDM_EPS)
                            rank = xi;
                    }
                    if(rank != xi)
                        for(unsigned short nu = 0; nu < mu; ++nu) {
                            K alpha = K(1.0) / work[xi * (k + 1) * mu + nu];
                            Blas<K>::scal(&n, &alpha, Q + xi * ldv + nu * n, &i__1);
                        }
                }
                if(work != R) {
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        for(unsigned short i = 0; i < rank; ++i)
                            for(unsigned short j = 0; j < rank; ++j)
                                R[(nu * k + i) * ldr + j] = work[(i * k + j) * mu + nu];
                }
            }
            return rank;
        }
        /* Function: Arnoldi
         *  Computes one iteration of the Arnoldi method for generating one basis vector of a Krylov space. */
        template<bool excluded, class K>
        static void Arnoldi(const char id, const unsigned short m, K* const* const H, K* const* const v, K* const s, underlying_type<K>* const sn, const int n, const int i, const int mu, const underlying_type<K>* const d, K* const work, const MPI_Comm& comm, K* const* const save = nullptr, const unsigned short shift = 0) {
#ifdef PETSCHPDDM_H
            PetscLogEventBegin(KSP_GMRESOrthogonalization, 0, 0, 0, 0);
#endif
            orthogonalization<excluded>(id & 3, n, i + 1 - shift, mu, v[shift], v[i + 1], H[i] + shift * mu, d, work, comm);
            if(excluded)
                std::fill_n(sn + i * mu, mu, 0.0);
            else if(d)
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    sn[i * mu + nu] = 0.0;
                    for(int j = 0; j < n; ++j)
                        sn[i * mu + nu] += d[j] * std::norm(v[i + 1][nu * n + j]);
                }
            else
                for(unsigned short nu = 0; nu < mu; ++nu)
                    sn[i * mu + nu] = std::real(Blas<K>::dot(&n, v[i + 1] + nu * n, &i__1, v[i + 1] + nu * n, &i__1));
            MPI_Allreduce(MPI_IN_PLACE, sn + i * mu, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                H[i][(i + 1) * mu + nu] = std::sqrt(sn[i * mu + nu]);
                if(!excluded && i < m - 1)
                    std::for_each(v[i + 1] + nu * n, v[i + 1] + (nu + 1) * n, [&](K& y) { y /= H[i][(i + 1) * mu + nu]; });
            }
            if(save)
                Wrapper<K>::template omatcopy<'T'>(i + 2 - shift, mu, H[i] + shift * mu, mu, save[i - shift], m + 1);
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
#ifdef PETSCHPDDM_H
            PetscLogEventEnd(KSP_GMRESOrthogonalization, 0, 0, 0, 0);
#endif
        }
        /* Function: BlockArnoldi
         *  Computes one iteration of the Block Arnoldi method for generating one basis vector of a block Krylov space. */
        template<bool excluded, class K>
        static bool BlockArnoldi(const char id, const unsigned short m, K* const* const H, K* const* const v, K* const tau, K* const s, const int lwork, const int n, const int i, const int mu, const underlying_type<K>* const d, K* const work, const MPI_Comm& comm, K* const* const save = nullptr, const unsigned short shift = 0) {
#ifdef PETSCHPDDM_H
            PetscLogEventBegin(KSP_GMRESOrthogonalization, 0, 0, 0, 0);
#endif
            int ldh = (m + 1) * mu;
            blockOrthogonalization<excluded>(id & 3, n, i + 1 - shift, mu, v[shift], v[i + 1], H[i] + shift * mu, ldh, d, work, comm);
            int info = QR<excluded>((id >> 2) & 7, n, mu, v[i + 1], H[i] + (i + 1) * mu, ldh, d, work, comm, i < m - 1);
            if(info != mu)
                return true;
            for(unsigned short nu = 0; nu < mu; ++nu)
                std::fill(H[i] + (i + 1) * mu + nu * ldh + nu + 1, H[i] + (nu + 1) * ldh, K());
            if(save)
                for(unsigned short nu = 0; nu < mu; ++nu)
                    std::copy_n(H[i] + shift * mu + nu * ldh, (i + 1 - shift) * mu + nu + 1, save[i - shift] + nu * ldh);
            const int N = 2 * mu;
            for(unsigned short k = shift; k < i; ++k)
                Lapack<K>::mqr("L", &(Wrapper<K>::transc), &N, &mu, &N, H[k] + k * mu, &ldh, tau + k * N, H[i] + k * mu, &ldh, work, &lwork, &info);
            Lapack<K>::geqrf(&N, &mu, H[i] + i * mu, &ldh, tau + i * N, work, &lwork, &info);
            Lapack<K>::mqr("L", &(Wrapper<K>::transc), &N, &mu, &N, H[i] + i * mu, &ldh, tau + i * N, s + i * mu, &ldh, work, &lwork, &info);
#ifdef PETSCHPDDM_H
            PetscLogEventEnd(KSP_GMRESOrthogonalization, 0, 0, 0, 0);
#endif
            return false;
        }
        template<bool excluded, class K>
        static void equilibrate(int n, K* sb, K* sx, std::function<K* (K*, unsigned int*, unsigned int*, int)>& lambda, unsigned int* local, unsigned short k, int rank, int div, const MPI_Comm& comm) {
            unsigned int* global = local + k;
            unsigned short j = 0;
            std::function<unsigned int* (unsigned int*, unsigned short)> find_zero = [](unsigned int* global, unsigned short k) { return std::find_if(global, global + k, [](const unsigned int& v) { return v == 0; }); };
            for(unsigned int* pt = find_zero(global, k); pt != global + k; pt = find_zero(global, k), ++j) {
                if((rank < k * div && (rank % div) == j) || (rank >= k * div && (j + (k - 1) * div == rank))) {
                    while(pt != global + k) {
                        unsigned int* swap = local;
                        for(unsigned short nu = 0; nu < k; ++nu, ++swap) {
                            if(*swap > 1 || (*swap == 1 && global[nu] > 1))
                                break;
                        }
                        if(swap != local + k) {
                            K* addr = lambda(sb, local, swap, n);
                            if(addr != sb + (std::distance(local, swap) + 1) * n) {
                                sb[std::distance(global, pt) * n + std::distance(sb + std::distance(local, swap) * n, addr)] = *addr;
                                sx[std::distance(global, pt) * n + std::distance(sb + std::distance(local, swap) * n, addr)] = sx[std::distance(sb, addr)];
                                *addr = sx[std::distance(sb, addr)] = K();
                                --*swap;
                                --global[std::distance(local, swap)];
                                ++local[std::distance(global, pt)];
                                ++global[std::distance(global, pt)];
                            }
                        }
                        unsigned int* next = find_zero(global, k);
                        if(next != pt)
                            pt = next;
                        else
                            pt = global + k;
                    }
                }
                MPI_Allreduce(local, global, k, MPI_UNSIGNED, MPI_SUM, comm);
            }
        }
        template<bool, class Operator, class K, typename std::enable_if<hpddm_method_id<Operator>::value>::type* = nullptr>
        static void preprocess(const Operator&, const K* const, K*&, K* const, K*&, const int&, unsigned short&, const MPI_Comm&);
        template<bool excluded, class Operator, class K, typename std::enable_if<!hpddm_method_id<Operator>::value>::type* = nullptr>
        static void preprocess(const Operator& A, const K* const b, K*& sb, K* const x, K*& sx, const int& mu, unsigned short& k, const MPI_Comm& comm) {
            static_assert(!excluded, "Not implemented");
            int size;
            MPI_Comm_size(comm, &size);
            if(k < 2 || size == 1 || mu > 1) {
                sx = x;
                sb = const_cast<K*>(b);
                k = 1;
            }
            else {
                int rank;
                MPI_Comm_rank(comm, &rank);
                k = std::min(k, static_cast<unsigned short>(size));
                unsigned int* local = new unsigned int[2 * k];
                unsigned int* global = local + k;
                const int n = A.getDof();
                unsigned short j = std::min(k - 1, rank / (size / k));
                std::function<void ()> check_size = [&] {
                    std::fill_n(local, k, 0U);
                    for(int i = 0; i < n; ++i) {
                        if(std::abs(b[i]) > HPDDM_EPS) {
                            if(++local[j] > k)
                                break;
                        }
                    }
                    MPI_Allreduce(local, global, k, MPI_UNSIGNED, MPI_SUM, comm);
                };
                check_size();
                {
                    unsigned int max = 0;
                    for(unsigned short nu = 0; nu < k && max < k; ++nu)
                        max += global[nu];
                    if(max < k) {
                        k = std::max(1U, max);
                        global = local + k;
                        j = std::min(k - 1, rank / (size / k));
                        check_size();
                    }
                }
                if(k > 1) {
                    sx = new K[k * n]();
                    sb = new K[k * n]();
                    std::copy_n(x, n, sx + j * n);
                    std::copy_n(b, n, sb + j * n);
                    std::function<K* (K*, unsigned int*, unsigned int*, int)> lambda = [](K* sb, unsigned int* local, unsigned int* swap, int n) { return static_cast<K*>(std::find_if(sb + std::distance(local, swap) * n, sb + (std::distance(local, swap) + 1) * n, [](const K& v) { return std::abs(v) > HPDDM_EPS; })); };
                    equilibrate<excluded>(n, sb, sx, lambda, local, k, rank, size / k, comm);
                }
                else {
                    sx = x;
                    sb = const_cast<K*>(b);
                }
                delete [] local;
            }
            checkEnlargedMethod(A, k);
        }
        template<class Operator>
        static void checkEnlargedMethod(const Operator& A, const unsigned short& k) {
            const std::string prefix = A.prefix();
#if !HPDDM_PETSC
            Option& opt = *Option::get();
            if(k <= 1)
                opt.remove(prefix + "enlarge_krylov_subspace");
            else {
                opt[prefix + "enlarge_krylov_subspace"] = k;
                if(!opt.any_of(prefix + "krylov_method", { HPDDM_KRYLOV_METHOD_BGMRES, HPDDM_KRYLOV_METHOD_BCG, HPDDM_KRYLOV_METHOD_BGCRODR, HPDDM_KRYLOV_METHOD_BFBCG })) {
                    opt[prefix + "krylov_method"] = HPDDM_KRYLOV_METHOD_BGMRES;
                    if(opt.val<char>(prefix + "verbosity", 0))
                        std::cout << "WARNING -- block iterative methods should be used when enlarging Krylov subspaces, now switching to BGMRES" << std::endl;
                }
            }
#endif
        }
        template<bool excluded, class Operator, class K>
        static void postprocess(const Operator& A, const K* const b, K*& sb, K* const x, K*& sx, unsigned short& k) {
            if(sb != b) {
                const int n = A.getDof();
                std::fill_n(x, n, K());
                for(unsigned short j = 0; j < k; ++j)
                    Blas<K>::axpy(&n, &(Wrapper<K>::d__1), sx + j * n, &i__1, x, &i__1);
                delete [] sb;
                delete [] sx;
            }
        }
        template<class Operator = void, class K = double>
        static void printResidual(const Operator& A, const K* const b, const K* const x, const unsigned short mu, const unsigned short norm, const MPI_Comm& comm) {
            HPDDM::underlying_type<K>* storage = new HPDDM::underlying_type<K>[2 * mu]();
            computeResidual(A, b, x, storage, mu, norm);
            if(!hpddm_method_id<Operator>::value) {
                if(norm == HPDDM_COMPUTE_RESIDUAL_L2 || norm == HPDDM_COMPUTE_RESIDUAL_L1) {
                    MPI_Allreduce(MPI_IN_PLACE, storage, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
                    if(norm == HPDDM_COMPUTE_RESIDUAL_L2)
                        std::for_each(storage, storage + 2 * mu, [](underlying_type<K>& b) { b = std::sqrt(b); });
                }
                else
                    MPI_Allreduce(MPI_IN_PLACE, storage, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_MAX, comm);
            }
            int rank;
            MPI_Comm_rank(comm, &rank);
            if(rank == 0) {
                std::string header = "Final residual";
                if(mu > 1)
                    header = header + "s";
                const std::string prefix = A.prefix();
                if(prefix.size() > 0)
                    header = header + " (" + prefix + ")";
                header = header + ": ";
                std::cout << header << storage[1] << " / " << storage[0];
                if(mu > 1)
                    std::cout << " (rhs #1)\n";
                else
                    std::cout << "\n";
                for(unsigned short nu = 1; nu < mu; ++nu)
                    std::cout << std::string(header.size(), ' ') << storage[2 * nu + 1] << " / " << storage[2 * nu] << " (rhs #" << (nu + 1) << ")\n";
            }
            delete [] storage;
        }
        template<class Operator, class K, typename std::enable_if<hpddm_method_id<Operator>::value != 0>::type* = nullptr>
        static void computeResidual(const Operator& A, const K* const b, const K* const x, underlying_type<K>* const storage, const unsigned short mu, const unsigned short norm) {
            A.computeResidual(x, b, storage, mu, norm);
        }
        template<class Operator, class K, typename std::enable_if<hpddm_method_id<Operator>::value == 0>::type* = nullptr>
        static void computeResidual(const Operator& A, const K* const b, const K* const x, underlying_type<K>* const storage, const unsigned short mu, const unsigned short norm) {
            int dim = mu * A.getDof();
            K* tmp = new K[dim];
            A.GMV(x, tmp, mu);
            Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), b, &i__1, tmp, &i__1);
            if(norm == HPDDM_COMPUTE_RESIDUAL_L1) {
                for(unsigned int i = 0, n = A.getDof(); i < n; ++i) {
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        storage[2 * nu + 1] += std::abs(tmp[nu * n + i]);
                        if(std::abs(b[nu * n + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] += std::abs(b[nu * n + i] / underlying_type<K>(HPDDM_PEN));
                        else
                            storage[2 * nu] += std::abs(b[nu * n + i]);
                    }
                }
            }
            else if(norm == HPDDM_COMPUTE_RESIDUAL_LINFTY) {
                for(unsigned int i = 0, n = A.getDof(); i < n; ++i) {
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        storage[2 * nu + 1] = std::max(std::abs(tmp[nu * n + i]), storage[2 * nu + 1]);
                        if(std::abs(b[nu * n + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] = std::max(std::abs(b[nu * n + i] / underlying_type<K>(HPDDM_PEN)), storage[2 * nu]);
                        else
                            storage[2 * nu] = std::max(std::abs(b[nu * n + i]), storage[2 * nu]);
                    }
                }
            }
            else {
                for(unsigned int i = 0, n = A.getDof(); i < n; ++i) {
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        storage[2 * nu + 1] += std::norm(tmp[nu * n + i]);
                        if(std::abs(b[nu * n + i]) > HPDDM_EPS * HPDDM_PEN)
                            storage[2 * nu] += std::norm(b[nu * n + i] / underlying_type<K>(HPDDM_PEN));
                        else
                            storage[2 * nu] += std::norm(b[nu * n + i]);
                    }
                }
            }
            delete [] tmp;
        }
    public:
        template<class K>
        static void orthogonalization(const char id, const int n, const int k, const K* const B, K* const v) {
            K* H = new K[k];
            orthogonalization<false>(id, n, k, 1, B, v, H, static_cast<underlying_type<K>*>(nullptr), static_cast<K*>(nullptr), MPI_COMM_SELF);
            delete [] H;
        }
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
        template<bool, class Operator, class K>
        static int GMRES(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm);
        template<bool, class Operator, class K>
        static int BGMRES(const Operator&, const K* const, K* const, const int&, const MPI_Comm&);
        template<bool, class Operator, class K>
        static int GCRODR(const Operator&, const K* const, K* const, const int&, const MPI_Comm&);
        template<bool, class Operator, class K>
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
        template<bool, class Operator, class K>
        static int CG(const Operator& A, const K* const b, K* const x, const int&, const MPI_Comm& comm);
        template<bool, class Operator, class K>
        static int BCG(const Operator&, const K* const, K* const, const int&, const MPI_Comm&);
        template<bool, class Operator, class K>
        static int BFBCG(const Operator&, const K* const, K* const, const int&, const MPI_Comm&);
#if !defined(_KSPIMPL_H)
        template<bool excluded, class Operator, class K>
        static int Richardson(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm&) {
            K factor;
            unsigned short it;
            {
                underlying_type<K> d;
                options<7>(A, &d, nullptr, &it, nullptr);
                factor = d;
            }
            int ierr;
            const int n = excluded ? 0 : mu * A.getDof();
            K* work = new K[2 * n];
            K* r = work + n;
            bool allocate = A.template start<excluded>(b, x, mu);
            unsigned short j = 1;
            while(j++ <= it) {
                if(!excluded) {
                    ierr = A.GMV(x, r, mu);HPDDM_CHKERRQ(ierr);
                }
                Blas<K>::axpby(n, 1.0, b, 1, -1.0, r, 1);
                ierr = A.template apply<excluded>(r, work, mu);HPDDM_CHKERRQ(ierr);
                Blas<K>::axpy(&n, &factor, work, &i__1, x, &i__1);
            }
            delete [] work;
            A.end(allocate);
            return it;
        }
#endif
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
#if !HPDDM_PETSC || defined(_KSPIMPL_H)
        template<bool excluded = false, class Operator = void, class K = double, typename std::enable_if<!is_substructuring_method<Operator>::value>::type* = nullptr>
        static int solve(const Operator& A, const K* const b, K* const x, const int& mu
#if HPDDM_MPI
                                                                        , const MPI_Comm& comm) {
#else
                                                                                              ) {
            int comm = 0;
#endif
            std::ios_base::fmtflags ff(std::cout.flags());
            std::cout << std::scientific;
            const std::string prefix = A.prefix();
#if !defined(_KSPIMPL_H)
            Option& opt = *Option::get();
#if HPDDM_MIXED_PRECISION
            opt[prefix + "variant"] = HPDDM_VARIANT_FLEXIBLE;
#endif
            unsigned short k = opt.val<unsigned short>(prefix + "enlarge_krylov_subspace", 0);
            const char method = opt.val<char>(prefix + "krylov_method");
#else
            unsigned short k = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->scntl[reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG];
            char method = reinterpret_cast<KSP_HPDDM*>(A._ksp->data)->cntl[0];
#endif
            K* sx = nullptr;
            K* sb = nullptr;
            if(k)
                preprocess<excluded>(A, b, sb, x, sx, mu, k, comm);
            else {
                sx = x;
                sb = const_cast<K*>(b);
                k = 1;
            }
            int it;
            switch(method) {
                case HPDDM_KRYLOV_METHOD_NONE:     { const bool allocate = A.template start<excluded>(sb, sx, k * mu);
                                                     K* work = (hpddm_method_id<Operator>::value ? new K[k * mu * A.getDof()] : nullptr);
                                                     it = A.template apply<excluded>(sb, sx, k * mu, work);
                                                     delete [] work;
                                                     A.end(allocate);
                                                     HPDDM_IT(it, A) = 1;
#if defined(_KSPIMPL_H)
                                                     if(it) {
                                                         A._ksp->its = 0;
                                                         A._ksp->reason = KSP_DIVERGED_PC_FAILED;
                                                     }
                                                     else A._ksp->reason = KSP_CONVERGED_ITS;
#endif
                                                     break; }
#if !defined(_KSPIMPL_H)
                case HPDDM_KRYLOV_METHOD_RICHARDSON: it = Richardson<excluded>(A, sb, sx, k * mu, comm); break;
#endif
                case HPDDM_KRYLOV_METHOD_BFBCG:      it = BFBCG<excluded>(A, sb, sx, k * mu, comm); break;
                case HPDDM_KRYLOV_METHOD_BGCRODR:    it = BGCRODR<excluded>(A, sb, sx, k * mu, comm); break;
                case HPDDM_KRYLOV_METHOD_GCRODR:     it = GCRODR<excluded>(A, sb, sx, k * mu, comm); break;
                case HPDDM_KRYLOV_METHOD_BCG:        it = BCG<excluded>(A, sb, sx, k * mu, comm); break;
                case HPDDM_KRYLOV_METHOD_CG:         it = CG<excluded>(A, sb, sx, k * mu, comm); break;
                case HPDDM_KRYLOV_METHOD_BGMRES:     it = BGMRES<excluded>(A, sb, sx, k * mu, comm); break;
                default:                             it = GMRES<excluded>(A, sb, sx, k * mu, comm);
            }
            HPDDM_CHKERRQ(it);
            if(HPDDM_IT(it, A) >= 0) {
                postprocess<excluded>(A, b, sb, x, sx, k);
#if !HPDDM_PETSC
                k = opt.val<unsigned short>(prefix + "compute_residual", 10);
                if(!excluded && k != 10)
                    printResidual(A, b, x, mu, k, comm);
#endif
            }
            std::cout.flags(ff);
            return HPDDM_RET(it);
        }
#endif
        template<bool excluded = false, class Operator = void, class K = double, typename std::enable_if<is_substructuring_method<Operator>::value>::type* = nullptr>
        static int solve(const Operator& A, const K* const b, K* const x, const int&, const MPI_Comm& comm) {
            std::ios_base::fmtflags ff(std::cout.flags());
            std::cout << std::scientific;
            int it = PCG<excluded>(A, b, x, comm);
#if !HPDDM_PETSC
            unsigned short k = Option::get()->val<unsigned short>(A.prefix() + "compute_residual", 10);
            if(!excluded && k == HPDDM_COMPUTE_RESIDUAL_L2)
                printResidual(A, b, x, 1, HPDDM_COMPUTE_RESIDUAL_L2, comm);
#endif
            std::cout.flags(ff);
            return it;
        }
};
} // HPDDM
#endif // _HPDDM_ITERATIVE_
