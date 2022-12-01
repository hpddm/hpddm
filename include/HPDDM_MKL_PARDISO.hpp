/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2012-10-07

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

#ifndef _HPDDM_MKL_PARDISO_
#define _HPDDM_MKL_PARDISO_

#ifdef DMKL_PARDISO
#include <mkl_cluster_sparse_solver.h>
#endif
#ifdef MKL_PARDISOSUB
#include <mkl_pardiso.h>
#endif

namespace HPDDM {
template<class K>
struct prds {
    static constexpr int SPD = !Wrapper<K>::is_complex ? 2 : 4;
    static constexpr int SYM = !Wrapper<K>::is_complex ? -2 : 6;
    static constexpr int SSY = !Wrapper<K>::is_complex ? 1 : 3;
    static constexpr int UNS = !Wrapper<K>::is_complex ? 11 : 13;
};

#ifdef DMKL_PARDISO
#undef HPDDM_CHECK_SUBDOMAIN
#define HPDDM_CHECK_COARSEOPERATOR
#include "HPDDM_preprocessor_check.hpp"
#define COARSEOPERATOR HPDDM::MklPardiso
/* Class: MKL Pardiso
 *
 *  A class inheriting from <DMatrix> to use <MKL Pardiso>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class MklPardiso : public DMatrix {
    private:
        /* Variable: pt
         *  Internal data pointer. */
        void*          _pt[64];
        /* Variable: a
         *  Array of data. */
        K*                  _C;
        /* Variable: I
         *  Array of row pointers. */
        int*                _I;
        /* Variable: J
         *  Array of column indices. */
        int*                _J;
#if !HPDDM_INEXACT_COARSE_OPERATOR
        /* Variable: w
         *  Workspace array. */
        K*                  _w;
#endif
        /* Variable: mtype
         *  Matrix type. */
        int             _mtype;
        /* Variable: iparm
         *  Array of parameters. */
        mutable int _iparm[64];
        /* Variable: comm
         *  MPI communicator. */
        int              _comm;
    protected:
        /* Variable: numbering
         *  0-based indexing. */
        static constexpr char _numbering = 'F';
    public:
        MklPardiso() : _pt(), _C(), _I(), _J(),
#if !HPDDM_INEXACT_COARSE_OPERATOR
            _w(),
#endif
            _mtype(), _iparm(), _comm(-1) { }
        ~MklPardiso() {
#if !HPDDM_INEXACT_COARSE_OPERATOR
            delete [] _w;
#endif
            int phase = -1;
            int error;
            K ddum;
            int idum;
            if(_comm != -1) {
                int i__0 = 0;
                int i__1 = 1;
                CLUSTER_SPARSE_SOLVER(_pt, &i__1, &i__1, &_mtype, &phase, &(DMatrix::_n), &ddum, &idum, &idum, &i__1, &i__1, _iparm, &i__0, &ddum, &ddum, const_cast<int*>(&_comm), &error);
                _comm = -1;
            }
            delete [] _I;
            delete [] _C;
        }
        /* Function: numfact
         *
         *  Initializes <MKL Pardiso::pt> and <MKL Pardiso::iparm>, and factorizes the supplied matrix.
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
            if(DMatrix::_communicator != MPI_COMM_NULL && _comm == -1)
                _comm = MPI_Comm_c2f(DMatrix::_communicator);
            _I = I;
            _J = J;
            _C = C;
            const Option& opt = *Option::get();
            if(S == 'S')
                _mtype = opt.val<char>("operator_spd", 0) ? prds<K>::SPD : prds<K>::SYM;
            else
                _mtype = prds<K>::SSY;
            int phase, error;
            K ddum;
            std::fill_n(_iparm, 64, 0);
            _iparm[0]  = 1;
            _iparm[1]  = opt.val<int>("mkl_pardiso_iparm_2", 2);
#if !HPDDM_INEXACT_COARSE_OPERATOR
            _iparm[5]  = 1;
#else
            _iparm[5]  = 0;
#endif
            _iparm[9]  = opt.val<int>("mkl_pardiso_iparm_10", S != 'S' ? 13 : 8);
            _iparm[10] = opt.val<int>("mkl_pardiso_iparm_11", S != 'S' ? 1 : 0);
            _iparm[12] = opt.val<int>("mkl_pardiso_iparm_13", S != 'S' ? 1 : 0);
            _iparm[20] = opt.val<int>("mkl_pardiso_iparm_21", 1);
            _iparm[26] = opt.val<int>("mkl_pardiso_iparm_27", 0);
            _iparm[27] = std::is_same<double, underlying_type<K>>::value ? 0 : 1;
            _iparm[34] = (_numbering == 'C');
            _iparm[36] = bs;
            _iparm[39] = 2;
            _iparm[40] = loc2glob[0];
            _iparm[41] = loc2glob[1];
            phase = 12;
            *loc2glob = DMatrix::_n / bs;

            int i__0 = 0;
            int i__1 = 1;
            CLUSTER_SPARSE_SOLVER(_pt, &i__1, &i__1, &_mtype, &phase, loc2glob, C, _I, _J, &i__1, &i__1, _iparm, opt.val<char>("verbosity", 0) < 3 ? &i__0 : &i__1, &ddum, &ddum, const_cast<int*>(&_comm), &error);
#if !HPDDM_INEXACT_COARSE_OPERATOR
            _w = new K[(_iparm[41] - _iparm[40] + 1) * bs];
#endif
            C = nullptr;
            delete [] loc2glob;
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
            int error;
            int phase = 33;
            int nrhs = n;
            int i__0 = 0;
            int i__1 = 1;
#if !HPDDM_INEXACT_COARSE_OPERATOR
            if(n != 1) {
                delete [] _w;
                K** ptr = const_cast<K**>(&_w);
                *ptr = new K[(_iparm[41] - _iparm[40] + 1) * _iparm[36] * n];
            }
            CLUSTER_SPARSE_SOLVER(const_cast<void**>(_pt), &i__1, &i__1, &_mtype, &phase, &(DMatrix::_n), _C, _I, _J, &i__1, &nrhs, _iparm, &i__0, rhs, _w, const_cast<int*>(&_comm), &error);
#else
            CLUSTER_SPARSE_SOLVER(const_cast<void**>(_pt), &i__1, &i__1, &_mtype, &phase, &(DMatrix::_n), _C, _I, _J, &i__1, &nrhs, _iparm, &i__0, const_cast<K*>(rhs), x, const_cast<int*>(&_comm), &error);
#endif
        }
};
#endif // DMKL_PARDISO

#ifdef MKL_PARDISOSUB
#undef HPDDM_CHECK_COARSEOPERATOR
#define HPDDM_CHECK_SUBDOMAIN
#include "HPDDM_preprocessor_check.hpp"
#define SUBDOMAIN HPDDM::MklPardisoSub
template<class K>
class MklPardisoSub {
    private:
        void*          _pt[64];
        K*                  _C;
        int*                _I;
        int*                _J;
        K*                  _w;
        int             _mtype;
        mutable int _iparm[64];
        int                 _n;
        int           _partial;
    public:
        MklPardisoSub() : _pt(), _C(), _I(), _J(), _w(), _mtype(), _iparm(), _n(), _partial() { }
        MklPardisoSub(const MklPardisoSub&) = delete;
        ~MklPardisoSub() { dtor(); }
        static constexpr char _numbering = 'F';
        void dtor() {
            delete [] _w;
            _w = nullptr;
            int phase = -1;
            int error;
            int idum;
            K ddum;
            _n = 1;
            PARDISO(_pt, const_cast<int*>(&i__1), const_cast<int*>(&i__1), &_mtype, &phase, &_n, &ddum, &idum, &idum, const_cast<int*>(&i__1), const_cast<int*>(&i__1), _iparm, const_cast<int*>(&i__0), &ddum, &ddum, &error);
            if(_mtype == prds<K>::SPD || _mtype == prds<K>::SYM) {
                delete [] _I;
                delete [] _J;
                _I = nullptr;
                _J = nullptr;
            }
            if(_mtype == prds<K>::SYM) {
                delete [] _C;
                _C = nullptr;
            }
        }
        template<char N = HPDDM_NUMBERING>
        void numfact(MatrixCSR<K>* const& A, bool detection = false, K* const& schur = nullptr) {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            int* perm = nullptr;
            int phase, error;
            K ddum;
            const Option& opt = *Option::get();
            if(!_w) {
                _n = A->_n;
                std::fill_n(_iparm, 64, 0);
                _iparm[0]  = 1;
                _iparm[1]  = opt.val<int>("mkl_pardiso_iparm_2", 2);
                _iparm[5]  = 1;
                _iparm[9]  = opt.val<int>("mkl_pardiso_iparm_10", !A->_sym ? 13 : 8);
                _iparm[10] = opt.val<int>("mkl_pardiso_iparm_11", !A->_sym ? 1 : 0);
                _iparm[12] = opt.val<int>("mkl_pardiso_iparm_13", !A->_sym ? 1 : 0);
                _iparm[20] = opt.val<int>("mkl_pardiso_iparm_21", 1);
                _iparm[23] = opt.val<int>("mkl_pardiso_iparm_24", 0);
                _iparm[24] = opt.val<int>("mkl_pardiso_iparm_25", 0);
                _iparm[26] = opt.val<int>("mkl_pardiso_iparm_27", 0);
                _iparm[27] = std::is_same<double, underlying_type<K>>::value ? 0 : 1;
                _iparm[34] = (N == 'C');
                phase = 12;
                if(A->_sym) {
                    _I = new int[_n + 1];
                    _J = new int[A->_nnz];
                    _C = new K[A->_nnz];
                }
                else
                    _mtype = A->template structurallySymmetric<N>() ? prds<K>::SSY : prds<K>::UNS;
                if(schur) {
                    _iparm[35] = 2;
                    perm = new int[_n];
                    _partial = static_cast<int>(std::real(schur[1]));
                    std::fill_n(perm, _partial, 0);
                    std::fill(perm + _partial, perm + _n, 1);
                }
                _w = new K[_n];
            }
            else {
                if(_mtype == prds<K>::SPD)
                    _C = new K[A->_nnz];
                phase = 22;
            }
            if(A->_sym) {
                _mtype = (opt.val<char>("operator_spd", 0) && !detection) ? prds<K>::SPD : prds<K>::SYM;
                Wrapper<K>::template csrcsc<N, N>(&_n, A->_a, A->_ja, A->_ia, _C, _J, _I);
            }
            else {
                _I = A->_ia;
                _J = A->_ja;
                _C = A->_a;
            }
            PARDISO(_pt, const_cast<int*>(&i__1), const_cast<int*>(&i__1), &_mtype, &phase,
                    const_cast<int*>(&_n), _C, _I, _J, perm, const_cast<int*>(&i__1), _iparm, opt.val<char>("verbosity", 0) >= 4 ? const_cast<int*>(&i__1) : const_cast<int*>(&i__0), &ddum, schur, &error);
            delete [] perm;
            if(_mtype == prds<K>::SPD)
                delete [] _C;
        }
        template<char N = HPDDM_NUMBERING>
        int inertia(MatrixCSR<K>* const& A) {
            numfact<N>(A, true);
            return _iparm[22];
        }
        void solve(K* x) const {
            int error;
            _iparm[5] = 1;
            if(!_partial) {
                int phase = 33;
                PARDISO(const_cast<void**>(_pt), const_cast<int*>(&i__1), const_cast<int*>(&i__1), const_cast<int*>(&_mtype), &phase, const_cast<int*>(&_n), _C, _I, _J, const_cast<int*>(&i__1), const_cast<int*>(&i__1), _iparm, const_cast<int*>(&i__0), x, const_cast<K*>(_w), &error);
            }
            else {
                int phase = 331;
                PARDISO(const_cast<void**>(_pt), const_cast<int*>(&i__1), const_cast<int*>(&i__1), const_cast<int*>(&_mtype), &phase, const_cast<int*>(&_n), _C, _I, _J, const_cast<int*>(&i__1), const_cast<int*>(&i__1), _iparm, const_cast<int*>(&i__0), x, const_cast<K*>(_w), &error);
                std::fill(x + _partial, x + _n, K());
                phase = 333;
                PARDISO(const_cast<void**>(_pt), const_cast<int*>(&i__1), const_cast<int*>(&i__1), const_cast<int*>(&_mtype), &phase, const_cast<int*>(&_n), _C, _I, _J, const_cast<int*>(&i__1), const_cast<int*>(&i__1), _iparm, const_cast<int*>(&i__0), x, const_cast<K*>(_w), &error);
            }
        }
        void solve(const K* const b, K* const x) const {
            int error;
            if(!_partial) {
                _iparm[5] = 0;
                int phase = 33;
                PARDISO(const_cast<void**>(_pt), const_cast<int*>(&i__1), const_cast<int*>(&i__1), const_cast<int*>(&_mtype), &phase, const_cast<int*>(&_n), _C, _I, _J, const_cast<int*>(&i__1), const_cast<int*>(&i__1), _iparm, const_cast<int*>(&i__0), const_cast<K*>(b), x, &error);
            }
            else {
                _iparm[5] = 1;
                int phase = 331;
                std::copy_n(b, _partial, x);
                PARDISO(const_cast<void**>(_pt), const_cast<int*>(&i__1), const_cast<int*>(&i__1), const_cast<int*>(&_mtype), &phase, const_cast<int*>(&_n), _C, _I, _J, const_cast<int*>(&i__1), const_cast<int*>(&i__1), _iparm, const_cast<int*>(&i__0), x, const_cast<K*>(_w), &error);
                std::fill(x + _partial, x + _n, K());
                phase = 333;
                PARDISO(const_cast<void**>(_pt), const_cast<int*>(&i__1), const_cast<int*>(&i__1), const_cast<int*>(&_mtype), &phase, const_cast<int*>(&_n), _C, _I, _J, const_cast<int*>(&i__1), const_cast<int*>(&i__1), _iparm, const_cast<int*>(&i__0), x, const_cast<K*>(_w), &error);
            }
        }
        void solve(K* const x, const unsigned short& n) const {
            int error;
            int phase = 33;
            int nrhs = n;
            _iparm[5] = 1;
            K* w = new K[_n * n];
            PARDISO(const_cast<void**>(_pt), const_cast<int*>(&i__1), const_cast<int*>(&i__1), const_cast<int*>(&_mtype), &phase, const_cast<int*>(&_n), _C, _I, _J, const_cast<int*>(&i__1), &nrhs, _iparm, const_cast<int*>(&i__0), x, w, &error);
            delete [] w;
        }
        void solve(const K* const b, K* const x, const unsigned short& n) const {
            int error;
            int phase = 33;
            int nrhs = n;
            _iparm[5] = 0;
            PARDISO(const_cast<void**>(_pt), const_cast<int*>(&i__1), const_cast<int*>(&i__1), const_cast<int*>(&_mtype), &phase, const_cast<int*>(&_n), _C, _I, _J, const_cast<int*>(&i__1), &nrhs, _iparm, const_cast<int*>(&i__0), const_cast<K*>(b), x, &error);
        }
};
#endif // MKL_PARDISOSUB
} // HPDDM
#endif // _HPDDM_MKL_PARDISO_
