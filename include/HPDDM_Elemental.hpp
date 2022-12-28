/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2018-04-08

   Copyright (C) 2018-     Centre National de la Recherche Scientifique

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

#ifndef HPDDM_ELEMENTAL_HPP_
#define HPDDM_ELEMENTAL_HPP_

#ifdef DELEMENTAL
#include <El.hpp>
#endif

namespace HPDDM {

#ifdef DELEMENTAL
#undef HPDDM_CHECK_SUBDOMAIN
#define HPDDM_CHECK_COARSEOPERATOR
#include "HPDDM_preprocessor_check.hpp"
#define COARSEOPERATOR HPDDM::Elemental
/* Class: Elemental
 *
 *  A class inheriting from <DMatrix> to use <Elemental>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Elemental : public DMatrix {
    private:
        template<class T>
        using ElT = typename std::conditional<Wrapper<T>::is_complex, typename El::Complex<underlying_type<T>>, T>::type;
        El::Grid*               _grid;
        El::DistMatrix<ElT<K>>*    _A;
        El::DistMatrix<ElT<K>>*    _B;
        El::DistMatrix<ElT<K>>*    _d;
        El::DistPermutation*       _P;
        El::DistPermutation*       _Q;
        const int*          _loc2glob;
        unsigned short            _bs;
        unsigned short          _type;
    protected:
        /* Variable: numbering
         *  0-based indexing. */
        static constexpr char _numbering = 'C';
    public:
        Elemental() : _grid(), _A(), _B(), _d(), _P(), _Q(), _loc2glob(), _bs(), _type() {
            El::Initialize();
        }
        ~Elemental() {
            if(_loc2glob) {
                delete [] _loc2glob;
                _loc2glob = nullptr;
                delete _Q;
                delete _P;
                delete _d;
                delete _B;
                delete _A;
                delete _grid;
            }
            El::Finalize();
        }
        /* Function: numfact
         *
         *  Initializes <Elemental::pt> and <Elemental::iparm>, and factorizes the supplied matrix.
         *
         * Template Parameter:
         *    S              - 'S'ymmetric or 'G'eneral factorization.
         *
         * Parameters:
         *    I              - Array of row pointers.
         *    loc2glob       - Lower and upper bounds of the local domain.
         *    J              - Array of column indices.
         *    C              - Array of data. */
        template<char S>
        void numfact(unsigned short bs, int* I, int* loc2glob, int* J, K*& C) {
            El::mpi::Comm comm = DMatrix::_communicator;
            _grid = new El::Grid(comm, El::mpi::Size(comm), El::ROW_MAJOR);
            _A = new El::DistMatrix<ElT<K>>(*_grid);
            El::Zeros(*_A, DMatrix::_n, DMatrix::_n);
            if(I && J) {
                _A->Reserve(I[loc2glob[1] - loc2glob[0] + 1] * bs * bs);
                for(int i = 0; i < loc2glob[1] - loc2glob[0] + 1; ++i)
                    for(int j = I[i]; j < I[i + 1]; ++j)
                        for(int n = 0; n < bs; ++n)
                            for(int m = 0; m < bs; ++m) {
                                if(S == 'S') {
                                    if((loc2glob[0] + i) * bs + n < J[j] * bs + m)
                                        _A->QueueUpdate((loc2glob[0] + i) * bs + n, J[j] * bs + m, C[j * bs * bs + m + n * bs]);
                                    else if((loc2glob[0] + i) * bs + n == J[j] * bs + m)
                                        _A->QueueUpdate((loc2glob[0] + i) * bs + n, J[j] * bs + m, C[j * bs * bs + m + n * bs] / 2.0);
                                }
                                else
                                    _A->QueueUpdate((loc2glob[0] + i) * bs + n, J[j] * bs + m, C[j * bs * bs + m + n * bs]);
                            }
            }
            else {
                _A->Reserve((loc2glob[1] - loc2glob[0] + 1) * bs * bs * DMatrix::_n);
                for(int i = 0; i < loc2glob[1] - loc2glob[0] + 1; ++i)
                    for(int j = 0; j < DMatrix::_n; ++j)
                        for(int n = 0; n < bs; ++n)
                            for(int m = 0; m < bs; ++m) {
                                if(S == 'S') {
                                    if((loc2glob[0] + i) * bs + n < j * bs + m)
                                        _A->QueueUpdate((loc2glob[0] + i) * bs + n, j * bs + m, C[j * bs * bs + m + n * bs + i * bs * bs * DMatrix::_n]);
                                    else if((loc2glob[0] + i) * bs + n == j * bs + m)
                                        _A->QueueUpdate((loc2glob[0] + i) * bs + n, j * bs + m, C[j * bs * bs + m + n * bs + i * bs * bs * DMatrix::_n] / 2.0);
                                }
                                else
                                    _A->QueueUpdate((loc2glob[0] + i) * bs + n, j * bs + m, C[j * bs * bs + m + n * bs + i * bs * bs * DMatrix::_n]);
                            }
            }
            _A->ProcessQueues();
            if(S == 'S') {
                El::DistMatrix<ElT<K>> C(*_grid);
                El::Transpose(*_A, C);
                El::Axpy(Wrapper<K>::d__1, C, *_A);
            }
            _B = new El::DistMatrix<ElT<K>>(*_grid);
            El::Zeros(*_B, DMatrix::_n, 1);
            _loc2glob = loc2glob;
            _bs = bs;
            if(S == 'S')
                _type = Option::get()->val<char>("operator_spd", 0) ? 1 : 2;
            if(_type == 1) {
                _P = new El::DistPermutation(*_grid);
                El::Cholesky(El::LOWER, *_A, *_P);
            }
            else if(_type == 2) {
                _d = new El::DistMatrix<ElT<K>>(*_grid);
                _P = new El::DistPermutation(*_grid);
                El::LDL(*_A, *_d, *_P, false);
            }
            else {
                _P = new El::DistPermutation(*_grid);
                _Q = new El::DistPermutation(*_grid);
                El::LU(*_A, *_P, *_Q);
            }
        }
        /* Function: solve
         *
         *  Solves the system in-place.
         *
         * Parameters:
         *    rhs            - Input right-hand sides, solution vectors are stored in-place.
         *    n              - Number of right-hand sides. */
#if !HPDDM_INEXACT_COARSE_OPERATOR
        void solve(K* rhs, const unsigned short& n) const {
#else
        void solve(const K* const rhs, K* const x, const unsigned short& n) const {
#endif
            _B->Resize(DMatrix::_n * _bs, n);
            El::Fill(*_B, ElT<K>());
            _B->Reserve((_loc2glob[1] - _loc2glob[0] + 1) * n * _bs);
            for(int j = 0; j < n; ++j) {
                for(int i = 0; i < (_loc2glob[1] - _loc2glob[0] + 1) * _bs; ++i)
                    _B->QueueUpdate(_loc2glob[0] * _bs + i, j, rhs[i + (_loc2glob[1] - _loc2glob[0] + 1) * _bs * j]);
            }
            _B->ProcessQueues();
            if(_type == 1)
                El::cholesky::SolveAfter(El::LOWER, El::NORMAL, *_A, *_P, *_B);
            else if(_type == 2)
                El::ldl::SolveAfter(*_A, *_d, *_P, *_B, false);
            else
                El::lu::SolveAfter(El::NORMAL, *_A, *_P, *_Q, *_B);
            _B->ReservePulls((_loc2glob[1] - _loc2glob[0] + 1) * n * _bs);
            for(int j = 0; j < n; ++j) {
                for(int i = 0; i < (_loc2glob[1] - _loc2glob[0] + 1) * _bs; ++i)
                    _B->QueuePull(_loc2glob[0] * _bs + i, j);
            }
#if !HPDDM_INEXACT_COARSE_OPERATOR
            _B->ProcessPullQueue(reinterpret_cast<ElT<K>*>(rhs));
#else
            _B->ProcessPullQueue(reinterpret_cast<ElT<K>*>(x));
#endif
        }
};
#endif // DELEMENTAL
} // HPDDM
#endif // HPDDM_ELEMENTAL_HPP_
