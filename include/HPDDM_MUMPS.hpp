/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2012-10-04

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

#ifndef HPDDM_MUMPS_HPP_
#define HPDDM_MUMPS_HPP_

#include <smumps_c.h>
#include <dmumps_c.h>
#include <cmumps_c.h>
#include <zmumps_c.h>

#define HPDDM_GENERATE_MUMPS(C, c, T, M)                                                                     \
template<>                                                                                                   \
struct MUMPS_STRUC_C<T> {                                                                                    \
    typedef C ## MUMPS_STRUC_C trait;                                                                        \
    typedef M mumps_type;                                                                                    \
    static void mumps_c(C ## MUMPS_STRUC_C* const id) { c ## mumps_c(id); }                                  \
};

namespace HPDDM {
template<class>
struct MUMPS_STRUC_C { };
HPDDM_GENERATE_MUMPS(S, s, float, float)
HPDDM_GENERATE_MUMPS(D, d, double, double)
HPDDM_GENERATE_MUMPS(C, c, std::complex<float>, mumps_complex)
HPDDM_GENERATE_MUMPS(Z, z, std::complex<double>, mumps_double_complex)

#ifdef DMUMPS
#undef HPDDM_CHECK_SUBDOMAIN
#define HPDDM_CHECK_COARSEOPERATOR
#include "HPDDM_preprocessor_check.hpp"
#define COARSEOPERATOR HPDDM::Mumps
/* Class: Mumps
 *
 *  A class inheriting from <DMatrix> to use <Mumps>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Mumps : public DMatrix {
    private:
        /* Variable: id
         *  Internal data pointer. */
        typename MUMPS_STRUC_C<K>::trait* id_;
    protected:
        /* Variable: numbering
         *  1-based indexing. */
        static constexpr char numbering_ = 'F';
#if HPDDM_INEXACT_COARSE_OPERATOR
        std::pair<unsigned int, unsigned int> range_;
#endif
    public:
        Mumps() : id_() { }
        ~Mumps() {
            if(id_) {
#if HPDDM_INEXACT_COARSE_OPERATOR
                if(id_->nrhs)
                    delete [] id_->rhs;
#endif
                id_->job = -2;
                MUMPS_STRUC_C<K>::mumps_c(id_);
                delete id_;
            }
        }
        /* Function: numfact
         *
         *  Initializes <Mumps::id> and factorizes the supplied matrix.
         *
         * Template Parameter:
         *    S              - 'S'ymmetric or 'G'eneral factorization.
         *
         * Parameters:
         *    nz             - Number of nonzero entries.
         *    I              - Array of row indices.
         *    J              - Array of column indices.
         *    C              - Array of data. */
        template<char S>
        void numfact(unsigned int nz, int* I, int* J, K* C) {
            id_ = new typename MUMPS_STRUC_C<K>::trait();
            id_->job = -1;
            id_->par = 1;
            id_->comm_fortran = MPI_Comm_c2f(DMatrix::communicator_);
            const Option& opt = *Option::get();
            if(S == 'S')
                id_->sym = opt.val<char>("operator_spd", 0) ? 1 : 2;
            else
                id_->sym = 0;
            MUMPS_STRUC_C<K>::mumps_c(id_);
            id_->n = id_->lrhs = DMatrix::n_;
            id_->nz_loc = nz;
            id_->irn_loc = I;
            id_->jcn_loc = J;
            id_->a_loc = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(C);
#if !HPDDM_INEXACT_COARSE_OPERATOR
            id_->nrhs = 1;
#else
            id_->nrhs = 0;
            id_->icntl[20] = 0;
#endif
            id_->icntl[4]  = 0;
            id_->icntl[13] = opt.val<int>("mumps_icntl_14", 80);
            id_->icntl[17] = 3;
            for(unsigned short i : { 5, 6, 7, 11, 12, 13, 22, 23, 26, 27, 28, 34, 35, 36 }) {
                int val = opt.val<int>("mumps_icntl_" + to_string(i + 1));
                if(val != std::numeric_limits<int>::lowest())
                    id_->icntl[i] = val;
            }
            for(unsigned short i : { 0, 1, 2, 3, 4, 6 }) {
                double val = opt.val("mumps_cntl_" + to_string(i + 1));
                if(val >= std::numeric_limits<double>::lowest() / 10.0)
                    id_->cntl[i] = val;
            }
            id_->job = 4;
            if(opt.val<char>("verbosity", 0) < 3)
                id_->icntl[2] = 0;
            MUMPS_STRUC_C<K>::mumps_c(id_);
            if(DMatrix::rank_ == 0 && id_->infog[0] != 0)
                std::cerr << "BUG MUMPS, INFOG(1) = " << id_->infog[0] << std::endl;
            id_->icntl[2] = 0;
            delete [] I;
        }
        /* Function: solve
         *
         *  Solves the system in-place.
         *
         * Template Parameter:
         *    D              - Distribution of right-hand sides and solution vectors.
         *
         * Parameters:
         *    rhs            - Input right-hand sides, solution vectors are stored in-place.
         *    n              - Number of right-hand sides. */
#if !HPDDM_INEXACT_COARSE_OPERATOR
        template<DMatrix::Distribution D>
        void solve(K* const rhs, const unsigned short& n) {
            id_->nrhs = n;
            if(D == DMatrix::DISTRIBUTED_SOL) {
                id_->icntl[20] = 1;
                int info = id_->info[22];
                int* isol_loc = new int[info];
                K* sol_loc = new K[n * info];
                id_->sol_loc = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(sol_loc);
                id_->lsol_loc = info;
                id_->isol_loc = isol_loc;
                id_->rhs = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(rhs);
                id_->job = 3;
                MUMPS_STRUC_C<K>::mumps_c(id_);
                if(!DMatrix::mapOwn_ && !DMatrix::mapRecv_) {
                    int nloc = DMatrix::ldistribution_[DMatrix::rank_];
                    DMatrix::initializeMap<0>(info, id_->isol_loc, sol_loc, rhs);
                    DMatrix::ldistribution_ = new int[1];
                    *DMatrix::ldistribution_ = nloc;
                }
                else
                    DMatrix::redistribute<0>(sol_loc, rhs);
                for(unsigned short nu = 1; nu < n; ++nu)
                    DMatrix::redistribute<0>(sol_loc + nu * info, rhs + nu * *DMatrix::ldistribution_);
                delete [] sol_loc;
                delete [] isol_loc;
            }
            else {
                id_->icntl[20] = 0;
                id_->rhs = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(rhs);
                id_->job = 3;
                MUMPS_STRUC_C<K>::mumps_c(id_);
            }
        }
#else
        void solve(const K* const rhs, K* const x, const unsigned short& n) const {
            if(n > id_->nrhs) {
                if(id_->nrhs)
                    delete [] reinterpret_cast<K*>(id_->rhs);
                id_->rhs = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(new K[n * id_->n]);
                id_->nrhs = n;
            }
            std::fill_n(reinterpret_cast<K*>(id_->rhs), n * id_->n, K());
            for(unsigned short i = 0; i < n; ++i)
                std::copy_n(rhs + i * (range_.second - range_.first), range_.second - range_.first, reinterpret_cast<K*>(id_->rhs) + i * id_->n + range_.first);
            MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<K*>(id_->rhs), n * id_->n, Wrapper<K>::mpi_type(), MPI_SUM, DMatrix::communicator_);
            id_->job = 3;
            MUMPS_STRUC_C<K>::mumps_c(id_);
            MPI_Bcast(reinterpret_cast<K*>(id_->rhs), n * id_->n, Wrapper<K>::mpi_type(), 0, DMatrix::communicator_);
            for(unsigned short i = 0; i < n; ++i)
                std::copy_n(reinterpret_cast<K*>(id_->rhs) + i * id_->n + range_.first, range_.second - range_.first, x + i * (range_.second - range_.first));
        }
#endif
};
#endif // DMUMPS

#ifdef MUMPSSUB
#undef HPDDM_CHECK_COARSEOPERATOR
#define HPDDM_CHECK_SUBDOMAIN
#include "HPDDM_preprocessor_check.hpp"
#define SUBDOMAIN HPDDM::MumpsSub
template<class K>
class MumpsSub {
    private:
        typename MUMPS_STRUC_C<K>::trait* id_;
        int*                               I_;
    public:
        MumpsSub() : id_(), I_() { }
        MumpsSub(const MumpsSub&) = delete;
        ~MumpsSub() { dtor(); }
        static constexpr char numbering_ = 'F';
        void dtor() {
            delete [] I_;
            if(id_) {
                id_->job = -2;
                MUMPS_STRUC_C<K>::mumps_c(id_);
                delete id_;
                id_ = nullptr;
                I_ = nullptr;
            }
        }
        template<char N = HPDDM_NUMBERING>
        void numfact(MatrixCSR<K>* const& A, bool detection = false, K* const& schur = nullptr) {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            const Option& opt = *Option::get();
            if(!id_) {
                id_ = new typename MUMPS_STRUC_C<K>::trait();
                id_->job = -1;
                id_->par = 1;
                id_->comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);
                id_->sym = A->sym_ ? 2 - (opt.val<char>("operator_spd", 0) && !detection) : 0;
                MUMPS_STRUC_C<K>::mumps_c(id_);
            }
            id_->icntl[23] = detection;
            id_->cntl[2] = -1.0e-6;
            if(N == 'C')
                std::for_each(A->ja_, A->ja_ + A->nnz_, [](int& i) { ++i; });
            id_->jcn = A->ja_;
            id_->a = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(A->a_);
            int* listvar = nullptr;
            if(opt.val<char>("verbosity", 0) >= 4) {
                id_->icntl[0] = 6;
                id_->icntl[2] = 6;
                id_->icntl[3] = 2;
            }
            else {
                id_->icntl[0] = 0;
                id_->icntl[2] = 0;
                id_->icntl[3] = 0;
            }
            id_->icntl[13] = opt.val<int>("mumps_icntl_14", 80);
            for(unsigned short i : { 5, 6, 7, 11, 12, 13, 22, 23, 26, 27, 28, 34, 35, 36 }) {
                int val = opt.val<int>("mumps_icntl_" + to_string(i + 1));
                if(val != std::numeric_limits<int>::lowest())
                    id_->icntl[i] = val;
            }
            for(unsigned short i : { 0, 1, 2, 3, 4, 6 }) {
                double val = opt.val("mumps_cntl_" + to_string(i + 1));
                if(val >= std::numeric_limits<double>::lowest() / 10.0)
                    id_->cntl[i] = val;
            }
            if(id_->job == -1) {
                id_->nrhs = 1;
                id_->n = A->n_;
                id_->lrhs = A->n_;
                I_ = new int[A->nnz_];
                id_->nz = A->nnz_;
                for(int i = 0; i < A->n_; ++i)
                    std::fill(I_ + A->ia_[i] - (N == 'F'), I_ + A->ia_[i + 1] - (N == 'F'), i + 1);
                id_->irn = I_;
                if(schur) {
                    listvar = new int[static_cast<int>(std::real(schur[0]))];
                    std::iota(listvar, listvar + static_cast<int>(std::real(schur[0])), static_cast<int>(std::real(schur[1])));
                    id_->size_schur = id_->schur_lld = static_cast<int>(std::real(schur[0]));
                    id_->icntl[18] = 2;
                    id_->icntl[25] = 0;
                    id_->listvar_schur = listvar;
                    id_->nprow = id_->npcol = 1;
                    id_->mblock = id_->nblock = 100;
                    id_->schur = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(schur);
                }
                id_->job = 4;
            }
            else
                id_->job = 2;
            MUMPS_STRUC_C<K>::mumps_c(id_);
            delete [] listvar;
            if(id_->infog[0] != 0)
                std::cerr << "BUG MUMPS, INFOG(1) = " << id_->infog[0] << std::endl;
            id_->icntl[2] = 0;
            if(N == 'C')
                std::for_each(A->ja_, A->ja_ + A->nnz_, [](int& i) { --i; });
        }
        template<char N = HPDDM_NUMBERING>
        int inertia(MatrixCSR<K>* const& A) {
            Option& opt = *Option::get();
            double& v = opt["mumps_icntl_13"];
            bool remove = (v == 0.0);
            v = 1;
            numfact<N>(A, true);
            if(remove)
                opt.remove("mumps_icntl_13");
            return id_->infog[11];
        }
        unsigned short deficiency() const { return id_->infog[27]; }
        void solve(K* const x, const unsigned short& n = 1) const {
            id_->icntl[20] = 0;
            id_->icntl[26] = n;
            id_->rhs = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(x);
            id_->nrhs = n;
            id_->job = 3;
            MUMPS_STRUC_C<K>::mumps_c(id_);
        }
        void solve(const K* const b, K* const x, const unsigned short& n = 1) const {
            std::copy_n(b, n * id_->n, x);
            solve(x, n);
        }
};
#endif // MUMPSSUB
} // HPDDM
#endif // HPDDM_MUMPS_HPP_
