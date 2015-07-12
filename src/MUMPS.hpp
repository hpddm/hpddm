/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@inf.ethz.ch>
        Date: 2012-10-04

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

#ifndef _MUMPS_
#define _MUMPS_

#include <smumps_c.h>
#include <dmumps_c.h>
#include <cmumps_c.h>
#include <zmumps_c.h>
#ifndef MUMPS_VERSION
#define MUMPS_VERSION "0.0.0"
#endif

namespace HPDDM {
template<class>
struct MUMPS_STRUC_C {
};
template<>
struct MUMPS_STRUC_C<float> {
    typedef SMUMPS_STRUC_C trait;
    typedef float mumps_type;
    static inline void mumps_c(SMUMPS_STRUC_C* id) {
        smumps_c(id);
    }
};
template<>
struct MUMPS_STRUC_C<double> {
    typedef DMUMPS_STRUC_C trait;
    typedef double mumps_type;
    static inline void mumps_c(DMUMPS_STRUC_C* id) {
        dmumps_c(id);
    }
};
template<>
struct MUMPS_STRUC_C<std::complex<float>> {
    typedef CMUMPS_STRUC_C trait;
    typedef mumps_complex mumps_type;
    static inline void mumps_c(CMUMPS_STRUC_C* id) {
        cmumps_c(id);
    }
};
template<>
struct MUMPS_STRUC_C<std::complex<double>> {
    typedef ZMUMPS_STRUC_C trait;
    typedef mumps_double_complex mumps_type;
    static inline void mumps_c(ZMUMPS_STRUC_C* id) {
        zmumps_c(id);
    }
};

#ifdef DMUMPS
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
        typename MUMPS_STRUC_C<K>::trait* _id;
        /* Variable: strategy
         *  Ordering of the matrix during analysis phase. */
        char                        _strategy;
        static const std::string  _analysis[];
    protected:
        /* Variable: numbering
         *  1-based indexing. */
        static constexpr char _numbering = 'F';
    public:
        Mumps() : _id(), _strategy(3) { }
        ~Mumps() {
            if(_id) {
                _id->job = -2;
                MUMPS_STRUC_C<K>::mumps_c(_id);
                delete _id;
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
        inline void numfact(unsigned int nz, int* I, int* J, K* C) {
            _id = new typename MUMPS_STRUC_C<K>::trait();
            _id->job = -1;
            _id->par = 1;
            _id->comm_fortran = MPI_Comm_c2f(DMatrix::_communicator);
            if(S == 'S')
                _id->sym = 1;
            else
                _id->sym = 0;
            MUMPS_STRUC_C<K>::mumps_c(_id);
            _id->n = DMatrix::_n;
            _id->nz_loc = nz;
            _id->irn_loc = I;
            _id->jcn_loc = J;
            _id->a_loc = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(C);
            _id->nrhs = 1;
            _id->icntl[4] = 0;
            if(_strategy > 0 && _strategy < 9 && _strategy != 2) {
                _id->icntl[27] = 1;             // 1: sequential analysis
                _id->icntl[6]  = _strategy - 1; //     0: AMD
            }                                   //     1:
                                                //     2: AMF
                                                //     3: SCOTCH
                                                //     4: PORD
                                                //     5: METIS
                                                //     6: QAMD
                                                //     7: automatic
            else if(_strategy > 8 && _strategy < 12) {
                _id->icntl[27] = 2;             // 2: parallel analysis
                _id->icntl[28] = _strategy - 9; //     0: automatic
            }                                   //     1: PT-STOCH
                                                //     2: ParMetis
            _id->icntl[8]  = 1;
            _id->icntl[10] = 0;                 // verbose level
            _id->icntl[17] = 3;                 // distributed matrix input
            _id->icntl[19] = 0;                 // dense RHS
            _id->icntl[13] = 75;                // percentage increase in the estimated working space
            _id->job = 4;
            MUMPS_STRUC_C<K>::mumps_c(_id);
            if(DMatrix::_rank == 0) {
                if(_id->infog[31] == 1 || _id->infog[31] == 2)
                    std::cout << "                 (memory: " << _id->infog[20] << "MB -- ordering tool: " << _analysis[_id->infog[6] + (_id->infog[31] == 1 ? 0 : 8)] << ")" << std::endl;
                else if(_id->infog[0] != 0)
                    std::cerr << "BUG MUMPS, INFOG(1) = " << _id->infog[0] << std::endl;
            }
            _id->icntl[2] = 0;
            delete [] I;
        }
        /* Function: solve
         *
         *  Solves the system in-place.
         *
         * Template Parameter:
         *    D              - Distribution of right-hand sides and solution vectors.
         *
         * Parameter:
         *    rhs            - Input right-hand side, solution vector is stored in-place. */
        template<DMatrix::Distribution D>
        inline void solve(K* rhs) {
            if(D == DMatrix::DISTRIBUTED_SOL) {
                _id->icntl[20] = 1;
                int info = _id->info[22];
                int* isol_loc = new int[info];
                K* sol_loc = new K[info];
                _id->sol_loc = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(sol_loc);
                _id->lsol_loc = info;
                _id->isol_loc = isol_loc;
                _id->rhs = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(rhs);
                _id->job = 3;
                MUMPS_STRUC_C<K>::mumps_c(_id);
                if(!DMatrix::_mapOwn && !DMatrix::_mapRecv)
                    DMatrix::initializeMap<0>(info, _id->isol_loc, sol_loc, rhs);
                else
                    DMatrix::redistribute<0>(sol_loc, rhs);
                delete [] sol_loc;
                delete [] isol_loc;
            }
            else {
                _id->icntl[20] = 0;
                _id->rhs = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(rhs);
                _id->job = 3;
                MUMPS_STRUC_C<K>::mumps_c(_id);
            }
        }
        /* Function: initialize
         *
         *  Initializes <Mumps::strategy>, <DMatrix::rank>, and <DMatrix::distribution>.
         *
         * Parameter:
         *    parm           - Vector of parameters. */
        template<class Container>
        inline void initialize(Container& parm) {
            if(DMatrix::_communicator != MPI_COMM_NULL)
                MPI_Comm_rank(DMatrix::_communicator, &(DMatrix::_rank));
            if(parm[DISTRIBUTION] != DMatrix::DISTRIBUTED_SOL && parm[DISTRIBUTION] != DMatrix::NON_DISTRIBUTED) {
                if(DMatrix::_communicator != MPI_COMM_NULL && DMatrix::_rank == 0)
                    std::cout << "WARNING -- only distributed solution and nondistributed solution and RHS supported by the MUMPS interface, forcing the distribution to NON_DISTRIBUTED" << std::endl;
                DMatrix::_distribution = DMatrix::NON_DISTRIBUTED;
                parm[DISTRIBUTION] = DMatrix::NON_DISTRIBUTED;
            }
            else
                DMatrix::_distribution = static_cast<DMatrix::Distribution>(parm[DISTRIBUTION]);
            _strategy = parm[STRATEGY];
        }
};

template<class K>
const std::string Mumps<K>::_analysis[] { "AMD", "", "AMF", "SCOTCH", "PORD", "METIS", "QAMD", "automatic sequential", "automatic parallel", "PT-SCOTCH", "ParMetis" };
#endif // DMUMPS

#ifdef MUMPSSUB
#define SUBDOMAIN HPDDM::MumpsSub
template<class K>
class MumpsSub {
    private:
        typename MUMPS_STRUC_C<K>::trait* _id;
        int*                               _I;
    public:
        MumpsSub() : _id(), _I() { }
        MumpsSub(const MumpsSub&) = delete;
        ~MumpsSub() {
            if(_id) {
                _id->job = -2;
                MUMPS_STRUC_C<K>::mumps_c(_id);
                delete _id;
                _id = nullptr;
            }
            delete [] _I;
        }
        inline void numfact(MatrixCSR<K>* const& A, bool detection = false, K* const& schur = nullptr) {
            if(!_id) {
                _id = new typename MUMPS_STRUC_C<K>::trait();
                _id->job = -1;
                _id->par = 1;
                _id->comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);
                _id->sym = A->_sym ? 1 + detection : 0;
                MUMPS_STRUC_C<K>::mumps_c(_id);
            }
            _id->icntl[23] = detection;
            _id->cntl[2] = -1.0e-6;
            std::for_each(A->_ja, A->_ja + A->_nnz, [](int& i) { ++i; });
            _id->jcn = A->_ja;
            _id->a = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(A->_a);
            int* listvar = nullptr;
            if(_id->job == -1) {
                char strategy = (5 <= sto<int>(std::string(MUMPS_VERSION).substr(0, std::string(MUMPS_VERSION).find_first_of("."))) ? 4 : 3);
                _id->nrhs = 1;
                std::fill_n(_id->icntl, 5, 0);
                if(strategy > 0 && strategy < 9 && strategy != 2) {
                    _id->icntl[27] = 1;             // 1: sequential analysis
                    _id->icntl[6]  = strategy - 1;  //     0: AMD
                }                                   //     1:
                                                    //     2: AMF
                                                    //     3: SCOTCH
                                                    //     4: PORD
                                                    //     5: METIS
                                                    //     6: QAMD
                                                    //     7: automatic
                else {
                    _id->icntl[27] = 1;
                    _id->icntl[6]  = 7;
                }
                _id->icntl[8]  = 1;
                _id->icntl[10] = 0;
                _id->icntl[17] = 0;
                _id->icntl[19] = 0;
                _id->icntl[13] = 80;
                _id->n = A->_n;
                _id->lrhs = A->_n;
                _I = new int[A->_nnz];
                _id->nz = A->_nnz;
                for(int i = 0; i < A->_n; ++i)
                    std::fill(_I + A->_ia[i], _I + A->_ia[i + 1], i + 1);
                _id->irn = _I;
                if(schur != nullptr) {
                    listvar = new int[static_cast<int>(std::real(schur[0]))];
                    std::iota(listvar, listvar + static_cast<int>(std::real(schur[0])), static_cast<int>(std::real(schur[1])));
                    _id->size_schur = _id->schur_lld = static_cast<int>(std::real(schur[0]));
                    _id->icntl[18] = 2;
                    _id->icntl[25] = 0;
                    _id->listvar_schur = listvar;
                    _id->nprow = _id->npcol = 1;
                    _id->mblock = _id->nblock = 100;
                    _id->schur = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(schur);
                }
                _id->job = 4;
            }
            else
                _id->job = 2;
            MUMPS_STRUC_C<K>::mumps_c(_id);
            delete [] listvar;
            if(_id->infog[0] != 0)
                std::cerr << "BUG MUMPS, INFOG(1) = " << _id->infog[0] << std::endl;
            std::for_each(A->_ja, A->_ja + A->_nnz, [](int& i) { --i; });
        }
        inline unsigned short deficiency() const { return _id->infog[27]; }
        inline void solve(K* const x, const unsigned short& n = 1) const {
            _id->icntl[20] = 0;
            _id->rhs = reinterpret_cast<typename MUMPS_STRUC_C<K>::mumps_type*>(x);
            _id->nrhs = n;
            _id->job = 3;
            MUMPS_STRUC_C<K>::mumps_c(_id);
        }
        inline void solve(const K* const b, K* const x, const unsigned short& n = 1) const {
            std::copy_n(b, n * _id->n, x);
            solve(x, n);
        }
};
#endif // MUMPSSUB
} // HPDDM
#endif // _MUMPS_
