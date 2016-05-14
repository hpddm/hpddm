/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-09-30

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

#ifndef _HPDDM_DISSECTION_
#define _HPDDM_DISSECTION_

#ifndef INTEL_MKL_VERSION
# define BLAS_GENERIC
#else
# define BLAS_MKL
#endif
#define DD_REAL
#include "Driver/DissectionSolver.hpp"

#ifdef DISSECTIONSUB
#undef HPDDM_CHECK_COARSEOPERATOR
#define HPDDM_CHECK_SUBDOMAIN
#include "preprocessor_check.hpp"
#define SUBDOMAIN HPDDM::DissectionSub
namespace HPDDM {
template<class K>
class DissectionSub {
    private:
        DissectionSolver<K, underlying_type<K>>* _dslv;
    public:
        DissectionSub() : _dslv() { }
        DissectionSub(const DissectionSub&) = delete;
        ~DissectionSub() { delete  _dslv; }
        static constexpr char _numbering = 'C';
        template<char N = HPDDM_NUMBERING>
        void numfact(MatrixCSR<K>* const& A, bool detection = false, K* const& schur = nullptr) {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            static_assert(std::is_same<double, underlying_type<K>>::value, "Dissection only supports double-precision floating-point numbers");
            std::vector<unsigned int> missingCoefficients;
            for(unsigned int i = 0; i < A->_n; ++i)
                if(A->_ia[i + 1] == A->_ia[i] || (!A->_sym && !std::binary_search(A->_ja + A->_ia[i] - (N == 'F'), A->_ja + A->_ia[i + 1] - (N == 'F'), i + (N == 'F'))) || (A->_sym && A->_ja[A->_ia[i + 1] - (N == 'F') - 1] - (N == 'F') != i)) {
                    missingCoefficients.emplace_back(i);
                }
            int* ja;
            K* a;
            if(!missingCoefficients.empty()) {
                ja = new int[A->_nnz + missingCoefficients.size()]();
                a = new K[A->_nnz + missingCoefficients.size()]();
                unsigned int prev = 0;
                for(unsigned int i = 0; i < missingCoefficients.size(); ++i) {
                    std::copy(A->_ja + A->_ia[prev] - (N == 'F'), A->_ja + A->_ia[missingCoefficients[i]] - (N == 'F'), ja + A->_ia[prev] - (N == 'F') + i);
                    std::copy(A->_a + A->_ia[prev] - (N == 'F'), A->_a + A->_ia[missingCoefficients[i]] - (N == 'F'), a + A->_ia[prev] - (N == 'F') + i);
                    int dist = !A->_sym ? std::distance(A->_ja + A->_ia[missingCoefficients[i]] - (N == 'F'), std::lower_bound(A->_ja + A->_ia[missingCoefficients[i]] - (N == 'F'), A->_ja + A->_ia[missingCoefficients[i] + 1] - (N == 'F'), missingCoefficients[i] + (N == 'F'))) : A->_ia[missingCoefficients[i] + 1] - A->_ia[missingCoefficients[i]];
                    std::copy_n(A->_ja + A->_ia[missingCoefficients[i]] - (N == 'F'), dist, ja + A->_ia[missingCoefficients[i]] - (N == 'F') + i);
                    std::copy_n(A->_a + A->_ia[missingCoefficients[i]] - (N == 'F'), dist, a + A->_ia[missingCoefficients[i]] - (N == 'F') + i);
                    ja[A->_ia[missingCoefficients[i]] - (N == 'F') + i + dist] = missingCoefficients[i] + (N == 'F');
                    a[A->_ia[missingCoefficients[i]] - (N == 'F') + i + dist] = K();
                    if(!A->_sym) {
                        std::copy_n(A->_ja + A->_ia[missingCoefficients[i]] - (N == 'F') + dist, A->_ia[missingCoefficients[i] + 1] - A->_ia[missingCoefficients[i]] - dist, ja + A->_ia[missingCoefficients[i]] - (N == 'F') + i + dist + 1);
                        std::copy_n(A->_a + A->_ia[missingCoefficients[i]] - (N == 'F') + dist, A->_ia[missingCoefficients[i] + 1] - A->_ia[missingCoefficients[i]] - dist, a + A->_ia[missingCoefficients[i]] - (N == 'F') + i + dist + 1);
                    }
                    std::for_each(A->_ia + prev, A->_ia + missingCoefficients[i] + 1, [&](int& j) { j += i; });
                    prev = missingCoefficients[i] + 1;
                }
                std::copy(A->_ja + A->_ia[prev] - (N == 'F'), A->_ja + A->_nnz, ja + A->_ia[prev] - (N == 'F') + missingCoefficients.size());
                std::copy(A->_a + A->_ia[prev] - (N == 'F'), A->_a + A->_nnz, a + A->_ia[prev] - (N == 'F') + missingCoefficients.size());
                std::for_each(A->_ia + prev, A->_ia + A->_n + 1, [&](int& j) { j += missingCoefficients.size(); });
                A->_nnz += missingCoefficients.size();
            }
            else {
                ja = A->_ja;
                a = A->_a;
            }
            if(!_dslv) {
#ifdef _OPENMP
                int num_threads = omp_get_max_threads();
#else
                int num_threads = 1;
#endif
                _dslv = new DissectionSolver<K, underlying_type<K>>(num_threads, false, 0, nullptr);
                if(N == 'F') {
                    std::for_each(A->_ia, A->_ia + A->_n + 1, [](int& i) { --i; });
                    std::for_each(ja, ja + A->_nnz, [](int& i) { --i; });
                }
                _dslv->SymbolicFact(A->_n, A->_ia, ja, A->_sym, false);
                if(N == 'F') {
                    if(missingCoefficients.empty())
                        std::for_each(ja, ja + A->_nnz, [](int& i) { ++i; });
                    std::for_each(A->_ia, A->_ia + A->_n + 1, [](int& i) { ++i; });
                }
            }
            _dslv->NumericFact(0, a, Option::get()->val<char>("dissection_kkt_scaling", 0) ? KKT_SCALING : DIAGONAL_SCALING, Option::get()->val("dissection_pivot_tol", 1.0 / HPDDM_PEN));
            if(!missingCoefficients.empty()) {
                delete [] ja;
                delete [] a;
                unsigned int prev = 0;
                for(unsigned int i = 0; i < missingCoefficients.size(); ++i) {
                    std::for_each(A->_ia + prev, A->_ia + missingCoefficients[i] + 1, [&](int& j) { j -= i; });
                    prev = missingCoefficients[i] + 1;
                }
                std::for_each(A->_ia + prev, A->_ia + A->_n + 1, [&](int& i) { i -= missingCoefficients.size(); });
                A->_nnz -= missingCoefficients.size();
            }
        }
        unsigned short deficiency() const { return _dslv->kern_dimension(); }
        void solve(K* const x, const unsigned short& n = 1) const {
            if(n == 1)
                _dslv->SolveSingle(x, false, false, true);
            else
                _dslv->SolveMulti(x, n, false, false, true);
        }
        void solve(const K* const b, K* const x, const unsigned short& n = 1) const {
            std::copy_n(b, n * _dslv->dimension(), x);
            solve(x, n);
        }
};
} // HPDDM
#endif // DISSECTIONSUB
#endif // _HPDDM_DISSECTION_
