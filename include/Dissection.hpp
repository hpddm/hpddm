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
        ~DissectionSub() { dtor(); }
        static constexpr char _numbering = 'C';
        void dtor() {
            delete _dslv;
            _dslv = nullptr;
        }
        template<char N = HPDDM_NUMBERING>
        void numfact(MatrixCSR<K>* const& A, bool = false, K* const& = nullptr) {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            static_assert(std::is_same<double, underlying_type<K>>::value, "Dissection only supports double-precision floating-point numbers");
            const MatrixCSR<K>* B = A->template symmetrizedStructure<N, 'C'>();
            if(!_dslv) {
#ifdef _OPENMP
                int num_threads = omp_get_max_threads();
#else
                int num_threads = 1;
#endif
                _dslv = new DissectionSolver<K, underlying_type<K>>(num_threads, false, 0, nullptr);
                _dslv->SymbolicFact(B->_n, B->_ia, B->_ja, B->_sym, false);
                if(N == 'F' && B == A) {
                    std::for_each(A->_ja, A->_ja + A->_nnz, [](int& i) { ++i; });
                    std::for_each(A->_ia, A->_ia + A->_n + 1, [](int& i) { ++i; });
                }
            }
            _dslv->NumericFact(0, B->_a, Option::get()->val<char>("dissection_kkt_scaling", 0) ? KKT_SCALING : DIAGONAL_SCALING, Option::get()->val("dissection_pivot_tol", 1.0 / HPDDM_PEN));
            if(B != A)
                delete B;
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
