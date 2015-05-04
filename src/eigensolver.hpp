/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@inf.ethz.ch>
        Date: 2012-12-15

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

#ifndef _EIGENSOLVER_
#define _EIGENSOLVER_

#define HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(C, T, B, U)                                                     \
void HPDDM_F77(B ## gesdd)(const char*, const int*, const int*, U*, const int*, U*, U*, const int*, U*,      \
                           const int*, U*, const int*, int*, int*);                                          \
void HPDDM_F77(C ## gesdd)(const char*, const int*, const int*, T*, const int*, U*, T*, const int*, T*,      \
                           const int*, T*, const int*, U*, int*, int*);

#if !defined(INTEL_MKL_VERSION)
extern "C" {
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX(z, std::complex<double>, d, double)
}
#endif // INTEL_MKL_VERSION
#undef HPDDM_GENERATE_EXTERN_LAPACK_COMPLEX

namespace HPDDM {
/* Class: Eigensolver
 *
 *  A base class used to interface eigenvalue problem solvers such as <Arpack> or <Lapack>.
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Eigensolver {
    private:
        /* Function: gesdd
         *  Computes the singular value decomposition of a rectangular matrix, and optionally the left and/or right singular vectors, using a divide and conquer algorithm. */
        static inline void gesdd(const char*, const int*, const int*, K*, const int*, typename Wrapper<K>::ul_type*, K*, const int*, K*, const int*, K*, const int*, typename Wrapper<K>::ul_type*, int*, int*);
    protected:
        /* Variable: tol
         *  Relative tolerance of the eigenvalue problem solver. */
        typename Wrapper<K>::ul_type       _tol;
        /* Variable: threshold
         *  Threshold criterion. */
        typename Wrapper<K>::ul_type _threshold;
        /* Variable: n
         *  Number of rows of the eigenvalue problem. */
        int                                  _n;
        /* Variable: nu
         *  Number of desired eigenvalues. */
        int                                 _nu;
    public:
        Eigensolver(int n)                                                                                    : _tol(), _threshold(), _n(n), _nu() { }
        Eigensolver(int n, int& nu)                                                                           : _tol(1.0e-6), _threshold(), _n(n), _nu(std::max(1, std::min(nu, n))) { nu = _nu; }
        Eigensolver(typename Wrapper<K>::ul_type threshold, int n, int& nu)                                   : _tol(threshold > 0.0 ? HPDDM_EPS : 1.0e-6), _threshold(threshold), _n(n), _nu(std::max(1, std::min(nu, n))) { nu = _nu; }
        Eigensolver(typename Wrapper<K>::ul_type tol, typename Wrapper<K>::ul_type threshold, int n, int& nu) : _tol(threshold > 0.0 ? HPDDM_EPS : tol), _threshold(threshold), _n(n), _nu(std::max(1, std::min(nu, n))) { nu = _nu; }
        /* Function: selectNu
         *
         *  Computes a uniform threshold criterion.
         *
         * Parameters:
         *    eigenvalues   - Input array used to store eigenvalues in ascending order.
         *    communicator  - MPI communicator (usually <Subdomain::communicator>) on which the criterion <Eigensolver::nu> has to be uniformized. */
        template<class T>
        inline void selectNu(const T* const eigenvalues, const MPI_Comm& communicator) {
            static_assert(std::is_same<T, K>::value || std::is_same<T, typename Wrapper<K>::ul_type>::value, "Wrong types");
            unsigned short nev = _nu ? std::min(static_cast<int>(std::distance(eigenvalues, std::upper_bound(eigenvalues, eigenvalues + _nu, _threshold, [](const T& lhs, const T& rhs) { return std::real(lhs) < std::real(rhs); }))), _nu) : std::numeric_limits<unsigned short>::max();
            MPI_Allreduce(MPI_IN_PLACE, &nev, 1, MPI_UNSIGNED_SHORT, MPI_MIN, communicator);
            _nu = std::min(_nu, static_cast<int>(nev));
        }
        /* Function: getTol
         *  Returns the value of <Eigensolver::tol>. */
        inline typename Wrapper<K>::ul_type getTol() const { return _tol; }
        /* Function: getNu
         *  Returns the value of <Eigensolver::nu>. */
        inline int getNu() const { return _nu; }
        inline int workspace(const char* jobz, const int* const m) const {
            int info;
            int lwork = -1;
            K wkopt;
            gesdd(jobz, &(Eigensolver<K>::_n), m, nullptr, &(Eigensolver<K>::_n), nullptr, nullptr, &(Eigensolver<K>::_n), nullptr, m, &wkopt, &lwork, nullptr, nullptr, &info);
            return static_cast<int>(std::real(wkopt));
        }
        inline void svd(const char* jobz, const int* m, K* a, typename Wrapper<K>::ul_type* s, K* u, K* vt, K* work, const int* lwork, int* iwork, typename Wrapper<K>::ul_type* rwork = nullptr) const {
            int info;
            gesdd(jobz, &_n, m, a, &_n, s, u, &_n, vt, m, work, lwork, rwork, iwork, &info);
        }
        inline void purify(K* ev, const typename Wrapper<K>::ul_type* const d = nullptr) {
            int lwork = workspace("N", &_nu);
            K* a, *work;
            typename Wrapper<K>::ul_type* rwork;
            typename Wrapper<K>::ul_type* s;
            if(std::is_same<K, typename Wrapper<K>::ul_type>::value) {
                a = new K[_n * _nu + lwork + _nu];
                work = a + _n * _nu;
                s = reinterpret_cast<typename Wrapper<K>::ul_type*>(work) + lwork;
                rwork = nullptr;
            }
            else {
                a = new K[_n * _nu + lwork];
                work = a + _n * _nu;
                s = new typename Wrapper<K>::ul_type[_nu + std::max(1, _nu * std::max(5 * _nu + 7, 2 * _n + 2 * _nu + 1))];
                rwork = s + _nu;
            }
            if(d)
                Wrapper<K>::diagm(_n, _nu, d, ev, a);
            else
                std::copy_n(ev, _nu * _n, a);
            int* iwork = new int[8 * _n];
            int info;
            gesdd("N", &_n, &_nu, a, &_n, s, nullptr, &_n, nullptr, &_nu, work, &lwork, rwork, iwork, &info);
            delete [] iwork;
            if(!std::is_same<K, typename Wrapper<K>::ul_type>::value)
                delete [] s;
            delete [] a;
        }
};

#define HPDDM_GENERATE_LAPACK_COMPLEX(C, T, B, U)                                                            \
template<>                                                                                                   \
inline void Eigensolver<U>::gesdd(const char* jobz, const int* m, const int* n, U* a, const int* lda, U* s,  \
                                  U* u, const int* ldu, U* vt, const int* ldvt, U* work, const int* lwork,   \
                                  U*, int* iwork, int* info) {                                               \
    HPDDM_F77(B ## gesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);                \
}                                                                                                            \
template<>                                                                                                   \
inline void Eigensolver<T>::gesdd(const char* jobz, const int* m, const int* n, T* a, const int* lda, U* s,  \
                                  T* u, const int* ldu, T* vt, const int* ldvt, T* work, const int* lwork,   \
                                  U* rwork, int* iwork, int* info) {                                         \
    HPDDM_F77(C ## gesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);         \
}
HPDDM_GENERATE_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HPDDM_GENERATE_LAPACK_COMPLEX(z, std::complex<double>, d, double)
#undef HPDDM_GENERATE_LAPACK_COMPLEX
} // HPDDM
#endif // _EIGENSOLVER_
