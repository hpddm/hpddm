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

#ifndef HPDDM_COARSE_OPERATOR_HPP_
#define HPDDM_COARSE_OPERATOR_HPP_

#if HPDDM_PETSC
#define HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K) template<class Solver, class K>
#define HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K) Solver, K
#else
#define HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K) template<template<class> class Solver, char S, class K>
#define HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K) Solver, S, K
#endif

#if HPDDM_INEXACT_COARSE_OPERATOR
# if !defined(DMKL_PARDISO) && !defined(DMUMPS) && !HPDDM_PETSC
#  undef HPDDM_INEXACT_COARSE_OPERATOR
#  define HPDDM_INEXACT_COARSE_OPERATOR 0
#  pragma message("Inexact coarse operators require either: PARDISO or MUMPS as a distributed direct solver, or compilation with HPDDM_PETSC")
# else
#  include "HPDDM_inexact_coarse_operator.hpp"
# endif
#endif
#if !HPDDM_INEXACT_COARSE_OPERATOR
namespace HPDDM {
HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
class InexactCoarseOperator;
}
#endif
#if defined(DPASTIX) || defined(DMKL_PARDISO) || defined(DSUITESPARSE) || defined(DLAPACK) || defined(DHYPRE) || defined(DELEMENTAL) || HPDDM_INEXACT_COARSE_OPERATOR
# define HPDDM_CSR_CO
#endif
#if defined(DMKL_PARDISO) || defined(DSUITESPARSE) || defined(DLAPACK) || defined(DHYPRE) || defined(DELEMENTAL) || HPDDM_INEXACT_COARSE_OPERATOR
# define HPDDM_CONTIGUOUS
#endif

namespace HPDDM {
HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
using coarse_operator_type = typename std::conditional<HPDDM_INEXACT_COARSE_OPERATOR, InexactCoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>, Solver
#if !HPDDM_PETSC
                                               <K>
#endif
                                                  >::type;
/* Class: Coarse operator
 *
 *  A class for handling coarse corrections.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of the coarse operator.
 *    S              - 'S'ymmetric or 'G'eneral coarse operator.
 *    K              - Scalar type. */
HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
class CoarseOperator : public coarse_operator_type<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, downscaled_type<K>)> {
    public:
#if HPDDM_PETSC
        typedef PetscErrorCode return_type;
#else
        typedef std::pair<MPI_Request, const K*>* return_type;
#endif
    private:
        /* Variable: gatherComm
         *  Communicator used for assembling right-hand sides. */
        MPI_Comm               gatherComm_;
        /* Variable: scatterComm
         *  Communicator used for distributing solution vectors. */
        MPI_Comm              scatterComm_;
        /* Variable: rankWorld
         *  Rank of the current subdomain in the global communicator supplied as an argument of <Coarse operator::constructionCommunicator>. */
        int                     rankWorld_;
        /* Variable: sizeWorld
         *  Size of <Subdomain::communicator>. */
        int                     sizeWorld_;
        int                     sizeSplit_;
        /* Variable: local
         *  Local number of coarse degrees of freedom (usually set to <Eigensolver::nu> after a call to <Eigensolver::selectNu>). */
        int                         local_;
        /* Variable: sizeRHS
         *  Local size of right-hand sides and solution vectors. */
        unsigned int              sizeRHS_;
        bool                       offset_;
        /* Function: constructionCommunicator
         *  Builds both <Coarse operator::scatterComm> and <DMatrix::communicator>. */
        template<bool, class Operator>
        void constructionCommunicator(Operator&&, const MPI_Comm&);
        /* Function: constructionCollective
         *
         *  Builds the buffers <DMatrix::gatherCounts>, <DMatrix::displs>, <DMatrix::gatherSplitCounts>, and <DMatrix::displsSplit> for all collective communications involving coarse corrections.
         *
         * Template Parameters:
         *    U              - True if the distribution of the coarse operator is uniform, false otherwise.
         *    D              - <DMatrix::Distribution> of right-hand sides and solution vectors.
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise. */
        template<bool U, typename DMatrix::Distribution D, bool excluded>
        void constructionCollective(const unsigned short* = nullptr, unsigned short = 0, const unsigned short* = nullptr);
        /* Function: constructionMap
         *
         *  Builds the maps <DMatrix::ldistribution> and <DMatrix::idistribution> necessary for sending and receiving distributed right-hand sides or solution vectors.
         *
         * Template Parameters:
         *    T              - Coarse operator distribution topology.
         *    U              - True if the distribution of the coarse operator is uniform, false otherwise.
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise. */
        template<char T, bool U, bool excluded>
        void constructionMap(unsigned short, const unsigned short* = nullptr);
        /* Function: constructionMatrix
         *
         *  Builds and factorizes the coarse operator.
         *
         * Template Parameters:
         *    T              - Coarse operator distribution topology.
         *    U              - True if the distribution of the coarse operator is uniform, false otherwise.
         *    excluded       - True if the main processes are excluded from the domain decomposition, false otherwise.
         *    Operator       - Operator used in the definition of the Galerkin matrix. */
        template<char T, unsigned short U, unsigned short excluded, class Operator>
        return_type constructionMatrix(typename std::enable_if<Operator::pattern_ != 'u', Operator>::type&);
        template<char T, unsigned short U, unsigned short excluded, class Operator>
        return_type constructionMatrix(typename std::enable_if<Operator::pattern_ == 'u', Operator>::type&);
        template<char T, unsigned short U, unsigned short excluded, bool blocked>
        void finishSetup(unsigned short*&, const int, const unsigned short, unsigned short**&, const int);
        /* Function: constructionCommunicatorCollective
         *
         *  Builds both communicators <Coarse operator::gatherComm> and <DMatrix::scatterComm> needed for coarse corrections.
         *
         * Template Parameter:
         *    count          - True if the main processes must be taken into consideration, false otherwise. */
        template<bool count>
        void constructionCommunicatorCollective(const unsigned short* const pt, unsigned short size, MPI_Comm& in, MPI_Comm* const out = nullptr) {
            unsigned short sizeComm = std::count_if(pt, pt + size, [](const unsigned short& nu) { return nu != 0; });
            if(sizeComm != size && in != MPI_COMM_NULL) {
                MPI_Group oldComm, newComm;
                MPI_Comm_group(in, &oldComm);
                if(*pt == 0)
                    ++sizeComm;
                int* array = new int[sizeComm];
                array[0] = 0;
                for(unsigned short i = 1, j = 1, k = 0; j < sizeComm; ++i) {
                    if(pt[i] != 0)
                        array[j++] = i - k;
                    else if(count && super::ldistribution_[k + 1] == i)
                        ++k;
                }
                MPI_Group_incl(oldComm, sizeComm, array, &newComm);
                MPI_Group_free(&oldComm);
                if(out)
                    MPI_Comm_create(in, newComm, out);
                else {
                    MPI_Comm tmp;
                    MPI_Comm_create(in, newComm, &tmp);
                    MPI_Comm_free(&in);
                    if(tmp != MPI_COMM_NULL) {
                        MPI_Comm_dup(tmp, &in);
                        MPI_Comm_free(&tmp);
                    }
                    else
                        in = MPI_COMM_NULL;
                }
                MPI_Group_free(&newComm);
                delete [] array;
            }
            else if(out)
                MPI_Comm_dup(in, out);
        }
        /* Function: transfer
         *
         *  Transfers vectors from the fine grid to the coarse grid, and vice versa.
         *
         * Template Parameter:
         *    T              - True if fine to coarse, false otherwise.
         *
         * Parameters:
         *    counts         - Array of integers <DMatrix::gatherSplitCounts> or <DMatrix::gatherCounts> used for MPI collectives.
         *    n              - Number of vectors or size of the communicator <Coarse operator::gatherComm> for MPI collectives.
         *    m              - Size of the communicator <Coarse operator::gatherComm> for MPI collectives or number of vectors.
         *    ab             - Array to transfer. */
        template<bool T>
        void transfer(int* const counts, const int n, const int m, downscaled_type<K>* const ab) const {
            if(!T) {
                std::for_each(counts, counts + 2 * n, [&](int& i) { i *= m; });
                MPI_Gatherv(MPI_IN_PLACE, 0, Wrapper<downscaled_type<K>>::mpi_type(), ab, counts, counts + n, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_);
            }
            permute<T>(counts, n, m, ab);
            if(T) {
                MPI_Scatterv(ab, counts, counts + m, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_);
                std::for_each(counts, counts + 2 * m, [&](int& i) { i /= n; });
            }
        }
        template<bool T>
        void permute(int* const counts, const int n, const int m, downscaled_type<K>* const ab) const {
            if(n != 1 && m != 1) {
                int size = T ? m : n;
                downscaled_type<K>* ba = new downscaled_type<K>[counts[size - 1] + counts[2 * size - 1]];
                if(!T) {
                    for(int i = 0; i < size; ++i)
                        for(int j = 0; j < m; ++j)
                            std::copy_n(ab + counts[size + i] + j * (counts[i] / m), counts[i] / m, ba + counts[size + i] / m + j * ((counts[size - 1] + counts[2 * size - 1]) / m));
                }
                else {
                    for(int j = 0; j < n; ++j)
                        for(int i = 0; i < size; ++i)
                            std::copy_n(ab + counts[size + i] / n + j * ((counts[size - 1] + counts[2 * size - 1]) / n), counts[i] / n, ba + counts[size + i] + j * (counts[i] / n));
                }
                std::copy_n(ba, counts[size - 1] + counts[2 * size - 1], ab);
                delete [] ba;
            }
        }
        template<bool T>
        void Itransfer(int* const counts, const int n, const int m, downscaled_type<K>* const ab, MPI_Request* rq) const {
            if(!T) {
                std::for_each(counts, counts + 2 * n, [&](int& i) { i *= m; });
                MPI_Igatherv(MPI_IN_PLACE, 0, Wrapper<downscaled_type<K>>::mpi_type(), ab, counts, counts + n, Wrapper<downscaled_type<K>>::mpi_type(), 0, gatherComm_, rq);
                MPI_Wait(rq, MPI_STATUS_IGNORE);
            }
            permute<T>(counts, n, m, ab);
            if(T) {
                MPI_Iscatterv(ab, counts, counts + m, Wrapper<downscaled_type<K>>::mpi_type(), MPI_IN_PLACE, 0, Wrapper<downscaled_type<K>>::mpi_type(), 0, scatterComm_, rq);
                std::for_each(counts, counts + 2 * m, [&](int& i) { i /= n; });
            }
        }
    public:
        CoarseOperator() : gatherComm_(MPI_COMM_NULL), scatterComm_(MPI_COMM_NULL), rankWorld_(), sizeWorld_(), sizeSplit_(), local_(), sizeRHS_(), offset_(false) {
#if !HPDDM_PETSC
            static_assert(S == 'S' || S == 'G', "Unknown symmetry");
            static_assert(!Wrapper<K>::is_complex || S != 'S', "Symmetric complex coarse operators are not supported");
#endif
        }
        ~CoarseOperator() {
            int isFinalized;
            MPI_Finalized(&isFinalized);
            if(isFinalized)
                std::cerr << "Function " << __func__ << " in " << __FILE__ << ":" << __LINE__ << " should be called before MPI_Finalize()" << std::endl;
            else {
                if(gatherComm_ != scatterComm_ && gatherComm_ != MPI_COMM_NULL)
                    MPI_Comm_free(&gatherComm_);
                if(scatterComm_ != MPI_COMM_NULL)
                    MPI_Comm_free(&scatterComm_);
                gatherComm_ = scatterComm_ = MPI_COMM_NULL;
            }
        }
        /* Typedef: super
         *  Type of the immediate parent class <Solver>. */
        typedef coarse_operator_type<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, downscaled_type<K>)> super;
        /* Function: construction
         *  Wrapper function to call all needed subroutines. */
        template<unsigned short, unsigned short, class Operator>
        return_type construction(Operator&&, const MPI_Comm&);
        /* Function: callSolver
         *
         *  Solves a coarse system.
         *
         * Parameter:
         *    rhs            - Input right-hand side, solution vector is stored in-place. */
        template<bool>
        void callSolver(K* const, const unsigned short& = 1);
#if HPDDM_ICOLLECTIVE
        template<bool>
        void IcallSolver(K* const, const unsigned short&, MPI_Request*);
#endif
        /* Function: getRank
         *  Simple accessor that returns <Coarse operator::rankWorld>. */
        constexpr int getRank() const { return rankWorld_; }
        /* Function: getLocal
         *  Returns the value of <Coarse operator::local>. */
        constexpr int getLocal() const { return local_; }
        /* Function: getAddrLocal
         *  Returns the address of <Coarse operator::local>. */
        const int* getAddrLocal() const { return &local_; }
        /* Function: setLocal
         *  Sets the value of <Coarse operator::local>. */
        void setLocal(int l) { local_ = l; }
        /* Function: getSizeRHS
         *  Returns the value of <Coarse operator::sizeRHS>. */
        constexpr unsigned int getSizeRHS() const { return sizeRHS_; }
        const MPI_Comm& getCommunicator() const { return scatterComm_; }
};
} // HPDDM
#endif // HPDDM_COARSE_OPERATOR_HPP_
