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

#ifndef _PASTIX_
#define _PASTIX_

extern "C" {
#include <pastix.h>
#include <cscd_utils.h>
}

#define HPDDM_GENERATE_PASTIX(C, T)                                                                          \
template<>                                                                                                   \
struct pstx<T> {                                                                                             \
    static inline void dist(pastix_data_t** pastix_data, MPI_Comm pastix_comm,                               \
                            pastix_int_t n, pastix_int_t* colptr, pastix_int_t* row,                         \
                            T* avals, pastix_int_t* loc2glob, pastix_int_t* perm, pastix_int_t* invp,        \
                            T* b, pastix_int_t rhs, pastix_int_t* iparm, double* dparm) {                    \
        C ## _dpastix(pastix_data, pastix_comm,                                                              \
                      n, colptr, row, avals, loc2glob, perm, invp, b, rhs, iparm, dparm);                    \
    }                                                                                                        \
    static inline void seq(pastix_data_t** pastix_data, MPI_Comm pastix_comm,                                \
                           pastix_int_t n, pastix_int_t* colptr, pastix_int_t* row,                          \
                           T* avals, pastix_int_t* perm, pastix_int_t* invp,                                 \
                           T* b, pastix_int_t rhs, pastix_int_t* iparm, double* dparm) {                     \
        C ## _pastix(pastix_data, pastix_comm, n, colptr, row, avals, perm, invp, b, rhs, iparm, dparm);     \
    }                                                                                                        \
    static inline pastix_int_t cscd_redispatch(pastix_int_t n, pastix_int_t* ia, pastix_int_t* ja, T* a,     \
                                               T* rhs, pastix_int_t nrhs, pastix_int_t* l2g,                 \
                                               pastix_int_t dn, pastix_int_t** dia,                          \
                                               pastix_int_t** dja, T** da,                                   \
                                               T** drhs, pastix_int_t* dl2g,                                 \
                                               MPI_Comm comm, pastix_int_t dof) {                            \
        return C ## _cscd_redispatch(n, ia, ja, a, rhs, nrhs, l2g, dn, dia,                                  \
                                     dja, da, drhs, dl2g, comm, dof);                                        \
    }                                                                                                        \
    static inline void initParam(pastix_int_t* iparm, double* dparm) {                                       \
        C ## _pastix_initParam(iparm, dparm);                                                                \
    }                                                                                                        \
    static inline pastix_int_t getLocalNodeNbr(pastix_data_t** pastix_data) {                                \
        return C ## _pastix_getLocalNodeNbr(pastix_data);                                                    \
    }                                                                                                        \
    static inline pastix_int_t getLocalNodeLst(pastix_data_t** pastix_data, pastix_int_t* nodelst) {         \
        return C ## _pastix_getLocalNodeLst(pastix_data, nodelst);                                           \
    }                                                                                                        \
    static inline pastix_int_t setSchurUnknownList(pastix_data_t* pastix_data, pastix_int_t n,               \
                                                   pastix_int_t* list) {                                     \
        return C ## _pastix_setSchurUnknownList(pastix_data, n, list);                                       \
    }                                                                                                        \
    static inline pastix_int_t setSchurArray(pastix_data_t* pastix_data, T* array) {                         \
        return C ## _pastix_setSchurArray(pastix_data, array);                                               \
    }                                                                                                        \
    static constexpr API_FACT LLT = API_FACT_LLT;                                                            \
    static constexpr API_FACT LDLT = API_FACT_LDLT;                                                          \
};

namespace HPDDM {
template<class>
struct pstx {
};
HPDDM_GENERATE_PASTIX(s, float)
HPDDM_GENERATE_PASTIX(d, double)
#if defined(PASTIX_HAS_COMPLEX)
HPDDM_GENERATE_PASTIX(c, std::complex<float>)
HPDDM_GENERATE_PASTIX(z, std::complex<double>)
#endif

#ifdef DPASTIX
#define COARSEOPERATOR HPDDM::Pastix
/* Class: Pastix
 *
 *  A class inheriting from <DMatrix> to use <Pastix>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Pastix : public DMatrix {
    private:
        /* Variable: data
         *  Internal data pointer. */
        pastix_data_t*      _data;
        /* Variable: values2
         *  Array of data. */
        K*               _values2;
        /* Variable: dparm
         *  Array of double-precision floating-point parameters. */
        double*            _dparm;
        /* Variable: ncol2
         *  Number of local rows. */
        pastix_int_t       _ncol2;
        /* Variable: colptr2
         *  Array of row pointers. */
        pastix_int_t*    _colptr2;
        /* Variable: rows2
         *  Array of column indices. */
        pastix_int_t*      _rows2;
        /* Variable: loc2glob2
         *  Local to global numbering. */
        pastix_int_t*  _loc2glob2;
        /* Variable: iparm
         *  Array of integer parameters. */
        pastix_int_t*      _iparm;
    protected:
        /* Variable: numbering
         *  1-based indexing. */
        static constexpr char _numbering = 'F';
    public:
        Pastix() : _data(), _values2(), _dparm(), _colptr2(), _rows2(), _loc2glob2(), _iparm() { }
        ~Pastix() {
            free(_rows2);
            free(_values2);
            delete [] _loc2glob2;
            if(_iparm) {
                _iparm[IPARM_START_TASK]          = API_TASK_CLEAN;
                _iparm[IPARM_END_TASK]            = API_TASK_CLEAN;

                pstx<K>::dist(&_data, DMatrix::_communicator,
                              0, NULL, NULL, NULL, NULL,
                              NULL, NULL, NULL, 1, _iparm, _dparm);
                delete [] _iparm;
                delete [] _dparm;
            }
        }
        /* Function: numfact
         *
         *  Initializes <Pastix::iparm> and <Pastix::dparm>, and factorizes the supplied matrix.
         *
         * Template Parameter:
         *    S              - 'S'ymmetric or 'G'eneral factorization.
         *
         * Parameters:
         *    ncol           - Number of local rows.
         *    I              - Array of row pointers.
         *    loc2glob       - Local to global numbering.
         *    J              - Array of column indices.
         *    C              - Array of data. */
        template<char S>
        inline void numfact(unsigned int ncol, int* I, int* loc2glob, int* J, K* C) {
            _iparm = new pastix_int_t[IPARM_SIZE];
            _dparm = new double[DPARM_SIZE];

            pstx<K>::initParam(_iparm, _dparm);
            _iparm[IPARM_VERBOSE]             = API_VERBOSE_NO;
            _iparm[IPARM_MATRIX_VERIFICATION] = API_NO;
            _iparm[IPARM_START_TASK]          = API_TASK_INIT;
            _iparm[IPARM_END_TASK]            = API_TASK_INIT;
            if(S == 'S') {
                _iparm[IPARM_SYM]             = API_SYM_YES;
                _iparm[IPARM_FACTORIZATION]   = pstx<K>::LLT;
            }
            else {
                _iparm[IPARM_SYM]             = API_SYM_NO;
                _iparm[IPARM_FACTORIZATION]   = API_FACT_LU;
                if(!std::is_same<K, typename Wrapper<K>::ul_type>::value)
                    _iparm[IPARM_TRANSPOSE_SOLVE] = API_YES;
            }
            _iparm[IPARM_RHSD_CHECK]          = API_NO;
            pastix_int_t* perm = new pastix_int_t[ncol];
            pstx<K>::dist(&_data, DMatrix::_communicator,
                          ncol, I, J, NULL, loc2glob,
                          perm, NULL, NULL, 1, _iparm, _dparm);

            _iparm[IPARM_START_TASK]      = API_TASK_ORDERING;
            _iparm[IPARM_END_TASK]        = API_TASK_ANALYSE;

            pstx<K>::dist(&_data, DMatrix::_communicator,
                          ncol, I, J, NULL, loc2glob,
                          perm, NULL, NULL, 1, _iparm, _dparm);
            delete [] perm;

            _iparm[IPARM_VERBOSE]             = API_VERBOSE_NOT;

            _ncol2 = pstx<K>::getLocalNodeNbr(&_data);

            _loc2glob2 = new pastix_int_t[_ncol2];
            pstx<K>::getLocalNodeLst(&_data, _loc2glob2);

            pstx<K>::cscd_redispatch(ncol, I, J, C, NULL, 0, loc2glob,
                                     _ncol2, &_colptr2, &_rows2, &_values2, NULL, _loc2glob2,
                                     DMatrix::_communicator, 1);

            _iparm[IPARM_START_TASK]   = API_TASK_NUMFACT;
            _iparm[IPARM_END_TASK]     = API_TASK_NUMFACT;

            pstx<K>::dist(&_data, DMatrix::_communicator,
                          _ncol2, _colptr2, _rows2, _values2, _loc2glob2,
                          NULL, NULL, NULL, 1, _iparm, _dparm);

            _iparm[IPARM_CSCD_CORRECT] = API_YES;
            delete [] I;
            delete [] loc2glob;
        }
        /* Function: solve
         *
         *  Solves the system in-place.
         *
         * Template Parameter:
         *    D              - Distribution of right-hand sides and solution vectors.
         *
         * Parameters:
         *    rhs            - Input right-hand side, solution vector is stored in-place.
         *    fuse           - Number of fused reductions (optional). */
        template<DMatrix::Distribution D>
        inline void solve(K* rhs, const unsigned short& fuse = 0) {
            K* rhs2 = new K[_ncol2];
            if(!DMatrix::_mapOwn && !DMatrix::_mapRecv)
                DMatrix::initializeMap<1>(_ncol2, _loc2glob2, rhs2, rhs);
            else
                DMatrix::redistribute<1>(rhs2, rhs, fuse);

            _iparm[IPARM_START_TASK] = API_TASK_SOLVE;
            _iparm[IPARM_END_TASK]   = API_TASK_SOLVE;
            pstx<K>::dist(&_data, DMatrix::_communicator,
                          _ncol2, _colptr2, _rows2, _values2, _loc2glob2,
                          NULL, NULL, rhs2, 1, _iparm, _dparm);

            DMatrix::redistribute<2>(rhs, rhs2, fuse);
            delete [] rhs2;
        }
        /* Function: initialize
         *
         *  Initializes <DMatrix::rank> and <DMatrix::distribution>.
         *
         * Parameter:
         *    parm           - Vector of parameters. */
        template<class Container>
        inline void initialize(Container& parm) {
            if(DMatrix::_communicator != MPI_COMM_NULL)
                MPI_Comm_rank(DMatrix::_communicator, &(DMatrix::_rank));
            if(parm[DISTRIBUTION] != DMatrix::DISTRIBUTED_SOL_AND_RHS) {
                if(DMatrix::_communicator != MPI_COMM_NULL && DMatrix::_rank == 0)
                    std::cout << "WARNING -- only distributed solution and RHS supported by the PaStiX interface, forcing the distribution to DISTRIBUTED_SOL_AND_RHS" << std::endl;
                parm[DISTRIBUTION] = DMatrix::DISTRIBUTED_SOL_AND_RHS;
            }
            DMatrix::_distribution = DMatrix::DISTRIBUTED_SOL_AND_RHS;
        }
};
#endif // DPASTIX

#ifdef PASTIXSUB
#define SUBDOMAIN HPDDM::PastixSub
template<class K>
class PastixSub {
    private:
        pastix_data_t*    _data;
        K*              _values;
        double*          _dparm;
        pastix_int_t      _ncol;
        pastix_int_t*   _colptr;
        pastix_int_t*     _rows;
        pastix_int_t*    _iparm;
    public:
        PastixSub() : _data(), _values(), _dparm(), _colptr(), _rows(), _iparm() { }
        PastixSub(const PastixSub&) = delete;
        ~PastixSub() {
            if(_iparm) {
                if(_iparm[IPARM_SYM] == API_SYM_YES || _iparm[IPARM_SYM] == API_SYM_HER) {
                    delete [] _rows;
                    delete [] _colptr;
                    delete [] _values;
                }
                _iparm[IPARM_START_TASK]          = API_TASK_CLEAN;
                _iparm[IPARM_END_TASK]            = API_TASK_CLEAN;
                pstx<K>::seq(&_data, MPI_COMM_SELF,
                             0, NULL, NULL, NULL,
                             NULL, NULL, NULL, 1, _iparm, _dparm);
                delete [] _iparm;
                delete [] _dparm;
            }
        }
        inline void numfact(MatrixCSR<K>* const& A, bool detection = false, K* const& schur = nullptr) {
            if(!_iparm) {
                _iparm = new pastix_int_t[IPARM_SIZE];
                _dparm = new double[DPARM_SIZE];
                _ncol = A->_n;
                pstx<K>::initParam(_iparm, _dparm);
                _iparm[IPARM_VERBOSE]             = API_VERBOSE_NOT;
                _iparm[IPARM_MATRIX_VERIFICATION] = API_NO;
                _iparm[IPARM_START_TASK]          = API_TASK_INIT;
                _iparm[IPARM_END_TASK]            = API_TASK_INIT;
                _iparm[IPARM_SCHUR]               = schur ? API_YES : API_NO;
                _iparm[IPARM_RHSD_CHECK]          = API_NO;
                _dparm[DPARM_EPSILON_MAGN_CTRL]   = -1.0 / HPDDM_PEN;
                if(A->_sym) {
                    _values = new K[A->_nnz];
                    _colptr = new int[_ncol + 1];
                    _rows = new int[A->_nnz];
                    _iparm[IPARM_SYM]             = std::is_same<K, typename Wrapper<K>::ul_type>::value ? API_SYM_YES : API_SYM_HER;
                }
                else  {
                    _iparm[IPARM_SYM]             = API_SYM_NO;
                    _iparm[IPARM_FACTORIZATION]   = API_FACT_LU;
                }
            }
            if(A->_sym) {
                _iparm[IPARM_FACTORIZATION]       = detection ? pstx<K>::LDLT : pstx<K>::LLT;
                Wrapper<K>::template csrcsc<'F'>(&_ncol, A->_a, A->_ja, A->_ia, _values, _rows, _colptr);
            }
            else {
                _values = A->_a;
                _colptr = A->_ia;
                _rows = A->_ja;
                std::for_each(_colptr, _colptr + _ncol + 1, [](int& i) { ++i; });
                std::for_each(_rows, _rows + A->_nnz, [](int& i) { ++i; });
            }
            pastix_int_t* perm = new pastix_int_t[2 * _ncol];
            pastix_int_t* iperm = perm + _ncol;
            int* listvar = nullptr;
            if(_iparm[IPARM_START_TASK] == API_TASK_INIT) {
                pstx<K>::seq(&_data, MPI_COMM_SELF,
                             _ncol, _colptr, _rows, NULL,
                             NULL, NULL, NULL, 1, _iparm, _dparm);
                if(schur != nullptr) {
                    listvar = new int[static_cast<int>(std::real(schur[0]))];
                    std::iota(listvar, listvar + static_cast<int>(std::real(schur[0])), static_cast<int>(std::real(schur[1])));
                    pstx<K>::setSchurUnknownList(_data, static_cast<int>(std::real(schur[0])), listvar);
                    pstx<K>::setSchurArray(_data, schur);
                }
                _iparm[IPARM_START_TASK]          = API_TASK_ORDERING;
                _iparm[IPARM_END_TASK]            = API_TASK_NUMFACT;
            }
            else {
                _iparm[IPARM_START_TASK]          = API_TASK_NUMFACT;
                _iparm[IPARM_END_TASK]            = API_TASK_NUMFACT;
            }
            pstx<K>::seq(&_data, MPI_COMM_SELF,
                         _ncol, _colptr, _rows, _values,
                         perm, iperm, NULL, 1, _iparm, _dparm);
            delete [] listvar;
            delete [] perm;
            if(_iparm[IPARM_SYM] == API_SYM_NO) {
                std::for_each(_colptr, _colptr + _ncol + 1, [](int& i) { --i; });
                std::for_each(_rows, _rows + A->_nnz, [](int& i) { --i; });
            }
        }
        inline void solve(K* const x, const unsigned short& n = 1) const {
            _iparm[IPARM_START_TASK] = API_TASK_SOLVE;
            _iparm[IPARM_END_TASK]   = API_TASK_SOLVE;
            pstx<K>::seq(const_cast<pastix_data_t**>(&_data), MPI_COMM_SELF,
                         // _ncol, _colptr, _rows, _values,
                         _ncol, NULL, NULL, NULL,
                         NULL, NULL, x, n, _iparm, _dparm);
        }
        inline void solve(const K* const b, K* const x) const {
            std::copy_n(b, _ncol, x);
            solve(x);
        }
};
#endif // PASTIXSUB
} // HPDDM
#endif // _PASTIX_
