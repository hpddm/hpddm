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
        El::Grid*               grid_;
        El::DistMatrix<ElT<K>>*    A_;
        El::DistMatrix<ElT<K>>*    B_;
        El::DistMatrix<ElT<K>>*    d_;
        El::DistPermutation*       P_;
        El::DistPermutation*       Q_;
        const int*          loc2glob_;
        unsigned short            bs_;
        unsigned short          type_;
    protected:
        /* Variable: numbering
         *  0-based indexing. */
        static constexpr char numbering_ = 'C';
    public:
        Elemental() : grid_(), A_(), B_(), d_(), P_(), Q_(), loc2glob_(), bs_(), type_() {
            El::Initialize();
        }
        ~Elemental() {
            if(loc2glob_) {
                delete [] loc2glob_;
                loc2glob_ = nullptr;
                delete Q_;
                delete P_;
                delete d_;
                delete B_;
                delete A_;
                delete grid_;
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
            El::mpi::Comm comm = DMatrix::communicator_;
            grid_ = new El::Grid(comm, El::mpi::Size(comm), El::ROW_MAJOR);
            A_ = new El::DistMatrix<ElT<K>>(*grid_);
            El::Zeros(*A_, DMatrix::n_, DMatrix::n_);
            if(I && J) {
                A_->Reserve(I[loc2glob[1] - loc2glob[0] + 1] * bs * bs);
                for(int i = 0; i < loc2glob[1] - loc2glob[0] + 1; ++i)
                    for(int j = I[i]; j < I[i + 1]; ++j)
                        for(int n = 0; n < bs; ++n)
                            for(int m = 0; m < bs; ++m) {
                                if(S == 'S') {
                                    if((loc2glob[0] + i) * bs + n < J[j] * bs + m)
                                        A_->QueueUpdate((loc2glob[0] + i) * bs + n, J[j] * bs + m, C[j * bs * bs + m + n * bs]);
                                    else if((loc2glob[0] + i) * bs + n == J[j] * bs + m)
                                        A_->QueueUpdate((loc2glob[0] + i) * bs + n, J[j] * bs + m, C[j * bs * bs + m + n * bs] / 2.0);
                                }
                                else
                                    A_->QueueUpdate((loc2glob[0] + i) * bs + n, J[j] * bs + m, C[j * bs * bs + m + n * bs]);
                            }
            }
            else {
                A_->Reserve((loc2glob[1] - loc2glob[0] + 1) * bs * bs * DMatrix::n_);
                for(int i = 0; i < loc2glob[1] - loc2glob[0] + 1; ++i)
                    for(int j = 0; j < DMatrix::n_; ++j)
                        for(int n = 0; n < bs; ++n)
                            for(int m = 0; m < bs; ++m) {
                                if(S == 'S') {
                                    if((loc2glob[0] + i) * bs + n < j * bs + m)
                                        A_->QueueUpdate((loc2glob[0] + i) * bs + n, j * bs + m, C[j * bs * bs + m + n * bs + i * bs * bs * DMatrix::n_]);
                                    else if((loc2glob[0] + i) * bs + n == j * bs + m)
                                        A_->QueueUpdate((loc2glob[0] + i) * bs + n, j * bs + m, C[j * bs * bs + m + n * bs + i * bs * bs * DMatrix::n_] / 2.0);
                                }
                                else
                                    A_->QueueUpdate((loc2glob[0] + i) * bs + n, j * bs + m, C[j * bs * bs + m + n * bs + i * bs * bs * DMatrix::n_]);
                            }
            }
            A_->ProcessQueues();
            if(S == 'S') {
                El::DistMatrix<ElT<K>> C(*grid_);
                El::Transpose(*A_, C);
                El::Axpy(Wrapper<K>::d__1, C, *A_);
            }
            B_ = new El::DistMatrix<ElT<K>>(*grid_);
            El::Zeros(*B_, DMatrix::n_, 1);
            loc2glob_ = loc2glob;
            bs_ = bs;
            if(S == 'S')
                type_ = Option::get()->val<char>("operator_spd", 0) ? 1 : 2;
            if(type_ == 1) {
                P_ = new El::DistPermutation(*grid_);
                El::Cholesky(El::LOWER, *A_, *P_);
            }
            else if(type_ == 2) {
                d_ = new El::DistMatrix<ElT<K>>(*grid_);
                P_ = new El::DistPermutation(*grid_);
                El::LDL(*A_, *d_, *P_, false);
            }
            else {
                P_ = new El::DistPermutation(*grid_);
                Q_ = new El::DistPermutation(*grid_);
                El::LU(*A_, *P_, *Q_);
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
            B_->Resize(DMatrix::n_ * bs_, n);
            El::Fill(*B_, ElT<K>());
            B_->Reserve((loc2glob_[1] - loc2glob_[0] + 1) * n * bs_);
            for(int j = 0; j < n; ++j) {
                for(int i = 0; i < (loc2glob_[1] - loc2glob_[0] + 1) * bs_; ++i)
                    B_->QueueUpdate(loc2glob_[0] * bs_ + i, j, rhs[i + (loc2glob_[1] - loc2glob_[0] + 1) * bs_ * j]);
            }
            B_->ProcessQueues();
            if(type_ == 1)
                El::cholesky::SolveAfter(El::LOWER, El::NORMAL, *A_, *P_, *B_);
            else if(type_ == 2)
                El::ldl::SolveAfter(*A_, *d_, *P_, *B_, false);
            else
                El::lu::SolveAfter(El::NORMAL, *A_, *P_, *Q_, *B_);
            B_->ReservePulls((loc2glob_[1] - loc2glob_[0] + 1) * n * bs_);
            for(int j = 0; j < n; ++j) {
                for(int i = 0; i < (loc2glob_[1] - loc2glob_[0] + 1) * bs_; ++i)
                    B_->QueuePull(loc2glob_[0] * bs_ + i, j);
            }
#if !HPDDM_INEXACT_COARSE_OPERATOR
            B_->ProcessPullQueue(reinterpret_cast<ElT<K>*>(rhs));
#else
            B_->ProcessPullQueue(reinterpret_cast<ElT<K>*>(x));
#endif
        }
};
#endif // DELEMENTAL
} // HPDDM
#endif // HPDDM_ELEMENTAL_HPP_
