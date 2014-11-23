/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2013-03-10

   Copyright (C) 2011-2014 Université de Grenoble

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

#ifndef _SCHUR_
#define _SCHUR_

namespace HPDDM {
/* Class: Schur
 *
 *  A class from which derives <Bdd> and <Feti> that inherits from <Preconditioner>.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of local matrices.
 *    CoarseOperator - Class of the coarse operator.
 *    K              - Scalar type. */
template<template<class> class Solver, class CoarseOperator, class K>
class Schur : public Preconditioner<Solver, CoarseOperator, K> {
    private:
        /* Function: exchangeSchurComplement
         *
         *  Exchanges the local Schur complements <Schur::schur> to form an explicit restriction of the global Schur complement.
         *
         * Template Parameter:
         *    L              - 'S'ymmetric or 'G'eneral transfer of the local Schur complements.
         *
         * Parameters:
         *    rq             - MPI request to check completion of the MPI transfers.
         *    send           - Buffer for sending the local Schur complement.
         *    recv           - Buffer for receiving the local Schur complement of each neighboring subdomains.
         *    res            - Restriction of the global Schur complement. */
        template<char L>
        inline void exchangeSchurComplement(MPI_Request* const& rq, K* const* const& send, K* const* const& recv, K* const& res) const {
            if(send && recv && res) {
                if(L == 'S')
                    for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                        MPI_Irecv(recv[i], (Subdomain<K>::_map[i].second.size() * (Subdomain<K>::_map[i].second.size() + 1)) / 2, Wrapper<K>::mpi_type(), Subdomain<K>::_map[i].first, 1, Subdomain<K>::_communicator, rq + i);
                        for(unsigned int j = 0; j < Subdomain<K>::_map[i].second.size(); ++j)
                            for(unsigned int k = j; k < Subdomain<K>::_map[i].second.size(); ++k) {
                                if(Subdomain<K>::_map[i].second[j] < Subdomain<K>::_map[i].second[k])
                                    send[i][Subdomain<K>::_map[i].second.size() * j - (j * (j + 1)) / 2 + k] = _schur[Subdomain<K>::_map[i].second[j] * Subdomain<K>::_dof + Subdomain<K>::_map[i].second[k]];
                                else
                                    send[i][Subdomain<K>::_map[i].second.size() * j - (j * (j + 1)) / 2 + k] = _schur[Subdomain<K>::_map[i].second[k] * Subdomain<K>::_dof + Subdomain<K>::_map[i].second[j]];
                            }
                        MPI_Isend(send[i], (Subdomain<K>::_map[i].second.size() * (Subdomain<K>::_map[i].second.size() + 1)) / 2, Wrapper<K>::mpi_type(), Subdomain<K>::_map[i].first, 1, Subdomain<K>::_communicator, rq + Subdomain<K>::_map.size() + i);
                    }
                else
                    for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                        MPI_Irecv(recv[i], Subdomain<K>::_map[i].second.size() * Subdomain<K>::_map[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::_map[i].first, 1, Subdomain<K>::_communicator, rq + i);
                        for(unsigned int j = 0; j < Subdomain<K>::_map[i].second.size(); ++j)
                            for(unsigned int k = 0; k < Subdomain<K>::_map[i].second.size(); ++k) {
                                if(Subdomain<K>::_map[i].second[j] < Subdomain<K>::_map[i].second[k])
                                    send[i][j * Subdomain<K>::_map[i].second.size() + k] = _schur[Subdomain<K>::_map[i].second[j] * Subdomain<K>::_dof + Subdomain<K>::_map[i].second[k]];
                                else
                                    send[i][j * Subdomain<K>::_map[i].second.size() + k] = _schur[Subdomain<K>::_map[i].second[k] * Subdomain<K>::_dof + Subdomain<K>::_map[i].second[j]];
                            }
                        MPI_Isend(send[i], Subdomain<K>::_map[i].second.size() * Subdomain<K>::_map[i].second.size(), Wrapper<K>::mpi_type(), Subdomain<K>::_map[i].first, 1, Subdomain<K>::_communicator, rq + Subdomain<K>::_map.size() + i);
                    }
                Wrapper<K>::lacpy(&uplo, &(Subdomain<K>::_dof), &(Subdomain<K>::_dof), _schur, &(Subdomain<K>::_dof), res, &(Subdomain<K>::_dof));
                if(L == 'S')
                    for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                        int index;
                        MPI_Waitany(Subdomain<K>::_map.size(), rq, &index, MPI_STATUS_IGNORE);
                        for(unsigned int j = 0; j < Subdomain<K>::_map[index].second.size(); ++j) {
                            for(unsigned int k = 0; k < j; ++k)
                                if(Subdomain<K>::_map[index].second[j] <= Subdomain<K>::_map[index].second[k])
                                    res[Subdomain<K>::_map[index].second[j] * Subdomain<K>::_dof + Subdomain<K>::_map[index].second[k]] += recv[index][Subdomain<K>::_map[index].second.size() * k - (k * (k + 1)) / 2 + j];
                            for(unsigned int k = j; k < Subdomain<K>::_map[index].second.size(); ++k)
                                if(Subdomain<K>::_map[index].second[j] <= Subdomain<K>::_map[index].second[k])
                                    res[Subdomain<K>::_map[index].second[j] * Subdomain<K>::_dof + Subdomain<K>::_map[index].second[k]] += recv[index][Subdomain<K>::_map[index].second.size() * j - (j * (j + 1)) / 2 + k];
                        }
                    }
                else
                    for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i) {
                        int index;
                        MPI_Waitany(Subdomain<K>::_map.size(), rq, &index, MPI_STATUS_IGNORE);
                        for(unsigned int j = 0; j < Subdomain<K>::_map[index].second.size(); ++j)
                            for(unsigned int k = 0; k < Subdomain<K>::_map[index].second.size(); ++k)
                                if(Subdomain<K>::_map[index].second[j] <= Subdomain<K>::_map[index].second[k])
                                    res[Subdomain<K>::_map[index].second[j] * Subdomain<K>::_dof + Subdomain<K>::_map[index].second[k]] += recv[index][j * Subdomain<K>::_map[index].second.size() + k];
                    }
            }
        }
    protected:
        /* Variable: p
         *  Solver used in <Schur::callNumfact> for pseudo-factorizing <Subdomain::a>. */
        Solver<K>                     _p;
        /* Variable: bb
         *  Local matrix assembled on boundary degrees of freedom. */
        MatrixCSR<K, Wrapper<K>::I>* _bb;
        /* Variable: ii
         *  Local matrix assembled on interior degrees of freedom. */
        MatrixCSR<K>*                _ii;
        /* Variable: bi
         *  Local matrix assembled on boundary and interior degrees of freedom. */
        MatrixCSR<K, Wrapper<K>::I>* _bi;
        /* Variable: schur
         *  Explicit local Schur complement. */
        K*                        _schur;
        /* Variable: work
         *  Workspace array. */
        K*                         _work;
        /* Variable: structure
         *  Workspace array of size lower than or equal to <Subdomain::dof>. */
        K*                    _structure;
        /* Variable: rankWorld
         *  Rank of the current subdomain in <Subdomain::communicator>. */
        int                   _rankWorld;
        /* Variable: mult
         *  Number of local Lagrange multipliers. */
        int                        _mult;
        /* Variable: signed
         *  Number of neighboring subdomains in <Subdomain::communicator> with ranks lower than <rankWorld>. */
        unsigned short           _signed;
        /* Variable: deficiency
         *  Dimension of the kernel of <Subdomain::a>. */
        unsigned short       _deficiency;
        /* Function: solveGEVP
         *
         *  Solves the GenEO problem.
         *
         * Template Parameter:
         *    L              - 'S'ymmetric or 'G'eneral transfer of the local Schur complements.
         *
         * Parameters:
         *    d              - Constant pointer to a partition of unity of the primal unknowns, cf. <Bdd::m>.
         *    nu             - Number of eigenvectors requested.
         *    threshold      - Criterion for selecting the eigenpairs (optional). */
        template<char L>
        inline void solveGEVP(const double* const d, unsigned short& nu, const double& threshold) {
            if(_schur) {
                MPI_Request* rq = new MPI_Request[2 * Subdomain<K>::_map.size()];
                K** send = new K*[2 * Subdomain<K>::_map.size()];
                unsigned int size = 0;
                if(L == 'S')
                    for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i)
                        size += (Subdomain<K>::_map[i].second.size() * (Subdomain<K>::_map[i].second.size() + 1)) / 2;
                else
                    for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i)
                        size += Subdomain<K>::_map[i].second.size() * Subdomain<K>::_map[i].second.size();
                *send = new K[2 * size];
                K** recv = send + Subdomain<K>::_map.size();
                *recv = *send + size;
                if(L == 'S')
                    for(unsigned short i = 1; i < Subdomain<K>::_map.size(); ++i) {
                        send[i] = send[i - 1] + (Subdomain<K>::_map[i - 1].second.size() * (Subdomain<K>::_map[i - 1].second.size() + 1)) / 2;
                        recv[i] = recv[i - 1] + (Subdomain<K>::_map[i - 1].second.size() * (Subdomain<K>::_map[i - 1].second.size() + 1)) / 2;
                    }
                else
                    for(unsigned short i = 1; i < Subdomain<K>::_map.size(); ++i) {
                        send[i] = send[i - 1] + Subdomain<K>::_map[i - 1].second.size() * Subdomain<K>::_map[i - 1].second.size();
                        recv[i] = recv[i - 1] + Subdomain<K>::_map[i - 1].second.size() * Subdomain<K>::_map[i - 1].second.size();
                    }
                K* res = new K[Subdomain<K>::_dof * Subdomain<K>::_dof];
                exchangeSchurComplement<L>(rq, send, recv, res);

                Lapack<K> evp(nu >= 10 ? (nu >= 40 ? 1e-14 : 1e-12) : 1e-8, threshold, Subdomain<K>::_dof, nu);
                K* A;
                if(size < Subdomain<K>::_dof * Subdomain<K>::_dof)
                    A = new K[Subdomain<K>::_dof * Subdomain<K>::_dof];
                else
                    A = *recv;
                Wrapper<K>::lacpy(&uplo, &(Subdomain<K>::_dof), &(Subdomain<K>::_dof), _schur, &(Subdomain<K>::_dof), A, &(Subdomain<K>::_dof));
                if(d)
                    for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i)
                        for(unsigned int j = i; j < Subdomain<K>::_dof; ++j)
                            res[j + i * Subdomain<K>::_dof] *= d[i] * d[j];
                evp.reduce(A, res);
                int flag;
                int lwork = 64 * Subdomain<K>::_dof;
                MPI_Testall(Subdomain<K>::_map.size(), rq + Subdomain<K>::_map.size(), &flag, MPI_STATUSES_IGNORE);
                K* work;
                const int storage = std::is_same<K, double>::value ? 4 * Subdomain<K>::_dof - 1 : 2 * Subdomain<K>::_dof;
                if(flag) {
                    if((lwork + storage) <= size || (A != *recv && (lwork + storage) <= 2 * size))
                        work = *send;
                    else
                        work = new K[lwork + storage];
                }
                else {
                    if(A != *recv && (lwork + storage) <= size)
                        work = *recv;
                    else
                        work = new K[lwork + storage];
                }
                evp.solve(A, super::_ev, work, lwork, Subdomain<K>::_communicator);
                nu = evp.getNu();
                _deficiency = std::distance(reinterpret_cast<double*>(work) + lwork, std::upper_bound(reinterpret_cast<double*>(work) + lwork, reinterpret_cast<double*>(work) + lwork + nu, evp.getTol()));
                if(A != *recv)
                    delete [] A;
                evp.expand(res, super::_ev);
                if(nu == 0 && super::_ev) {
                    delete [] *super::_ev;
                    delete []  super::_ev;
                    super::_ev = nullptr;
                }
                if(work != *recv && work != *send)
                    delete [] work;
                if(!flag)
                    MPI_Waitall(Subdomain<K>::_map.size(), rq + Subdomain<K>::_map.size(), MPI_STATUSES_IGNORE);
                delete [] res;
                delete [] rq;
                delete [] *send;
                delete [] send;
            }
            else
                nu = 0;
        }
    public:
        Schur() : _bb(), _ii(), _bi(), _schur(), _work(), _structure(), _mult(), _signed(), _deficiency() { }
        Schur(const Schur&) = delete;
        ~Schur() {
            delete _bb;
            if(!_schur)
                delete _ii;
            delete _bi;
            delete [] _schur;
            delete [] _work;
        }
        /* Typedef: super
         *  Type of the immediate parent class <Preconditioner>. */
        typedef Preconditioner<Solver, CoarseOperator, K> super;
        /* Function: initialize
         *  Sets <Schur::rankWorld> and <Schur::signed>, and allocates <Schur::mult>, <Schur::work>, and <Schur::structure>. */
        template<bool m>
        inline void initialize() {
            MPI_Comm_rank(Subdomain<K>::_communicator, &_rankWorld);
            for(const pairNeighbor& neighbor : Subdomain<K>::_map) {
                _mult += neighbor.second.size();
                if(neighbor.first < _rankWorld)
                    ++_signed;
            }
            if(m) {
                _work = new K[_mult + Subdomain<K>::_a->_n];
                _structure = _work + _mult;
            }
            else {
                _work = new K[Subdomain<K>::_dof + Subdomain<K>::_a->_n];
                _structure = _work + Subdomain<K>::_dof;
            }
        }
        /* Function: callNumfact
         *  Factorizes <Subdomain::a>. */
        inline void callNumfact() {
            if(Subdomain<K>::_a) {
                if(_deficiency) {
#if defined(MUMPSSUB) || defined(PASTIXSUB)
                    _p.numfact(Subdomain<K>::_a, true);
#else
                    for(unsigned short i = 0; i < _deficiency; ++i)
                        _ii->_a[_ii->_ia[((i + 1) * _ii->_n) / (_deficiency + 1)] - 1] += HPDDM_PEN;
                    _p.numfact(Subdomain<K>::_a);
                    for(unsigned short i = 0; i < _deficiency; ++i)
                        _ii->_a[_ii->_ia[((i + 1) * _ii->_n) / (_deficiency + 1)] - 1] -= HPDDM_PEN;
#endif
                }
                else
                    _p.numfact(Subdomain<K>::_a);
            }
            else
                std::cerr << "The matrix '_a' has not been allocated => impossible to build the Neumann preconditioner" << std::endl;
        }
        /* Function: computeSchurComplement
         *  Computes the explicit Schur complement <Schur::schur>. */
        inline void computeSchurComplement() {
#if defined(MUMPSSUB) || defined(PASTIXSUB) || defined(MKL_PARDISOSUB)
            if(Subdomain<K>::_a) {
                _schur = new K[Subdomain<K>::_dof * Subdomain<K>::_dof];
                _schur[0] = Subdomain<K>::_dof;
#if defined(MKL_PARDISOSUB)
                _schur[1] = _bi->_m;
#else
                _schur[1] = _bi->_m + 1.0;
#endif
                if(_ii)
                    delete _ii;
                super::_s.numfact(Subdomain<K>::_a, true, _schur);
            }
            else
                std::cerr << "The matrix '_a' has not been allocated => impossible to build the Schur complement" << std::endl;
#else
#warning Consider changing your linear solver if you need to compute Schur complements
#endif
        }
        /* Function: callNumfactPreconditioner
         *  Factorizes <Schur::ii> if <Schur::schur> is not available. */
        inline void callNumfactPreconditioner() {
            if(_ii) {
                if(!_schur)
                    super::_s.numfact(_ii);
            }
            else
                std::cerr << "The matrix '_ii' has not been allocated => impossible to build the Dirichlet preconditioner" << std::endl;
        }
        /* Function: originalNumbering
         *
         *  Renumbers a vector according to the numbering of the user.
         *
         * Parameters:
         *    interface      - Numbering of the interface.
         *    in             - Input vector. */
        template<class Container>
        inline void originalNumbering(const Container& interface, K* const in) const {
            unsigned int end = Subdomain<K>::_a->_n;
            std::vector<K> backup(in + _bi->_m, in + end);
            for(unsigned int j = Subdomain<K>::_dof; j-- > 0; ) {
                std::copy_backward(in + interface[j] - j - 1, in + end - j - 1, in + end);
                in[interface[j]] = backup[j];
                end = interface[j];
            }
        }
        /* Function: renumber
         *
         *  Renumbers <Subdomain::a> and <Preconditioner::ev> to easily assemble <Schur::bb>, <Schur::ii>, and <Schur::bi>.
         *
         * Parameters:
         *    interface      - Numbering of the interface.
         *    f              - Right-hand side to renumber (optional). */
        template<class Container>
        inline void renumber(const Container& interface, K* const& f = nullptr) {
            if(!interface.empty()) {
                if(!_ii) {
                    Subdomain<K>::_dof = Subdomain<K>::_a->_n;
                    std::vector<signed int> vec;
                    vec.reserve(Subdomain<K>::_dof);
                    unsigned int i = 0, j = 0;
                    std::vector<std::vector<K>> deflationBoundary(super::getLocal());
                    for(std::vector<K>& deflation : deflationBoundary)
                        deflation.reserve(interface.size());
                    for(unsigned int k = 0; i < interface.size(); ++k) {
                        if(k == interface[i]) {
                            vec.emplace_back(++i);
                            for(unsigned short l = 0; l < deflationBoundary.size(); ++l)
                                deflationBoundary[l].emplace_back(super::_ev[l][k]);
                        }
                        else {
                            for(unsigned short l = 0; l < deflationBoundary.size(); ++l)
                                super::_ev[l][j] = super::_ev[l][k];
                            vec.emplace_back(-(++j));
                        }
                    }
                    for(unsigned int k = interface.back() + 1; k < Subdomain<K>::_dof; ++k) {
                        for(unsigned short l = 0; l < deflationBoundary.size(); ++l)
                            super::_ev[l][j] = super::_ev[l][k];
                        vec.emplace_back(-(++j));
                    }
                    for(unsigned short l = 0; l < deflationBoundary.size(); ++l)
                        std::copy(deflationBoundary[l].cbegin(), deflationBoundary[l].cend(), super::_ev[l] + Subdomain<K>::_dof - interface.size());
                    std::vector<std::pair<unsigned int, K>> tmpInterior;
                    std::vector<std::pair<unsigned int, K>> tmpBoundary;
                    std::vector<std::vector<std::pair<unsigned int, K>>> tmpInteraction(interface.size());
                    tmpInterior.reserve(Subdomain<K>::_a->_nnz * (Subdomain<K>::_dof - interface.size()) / Subdomain<K>::_dof);
                    tmpBoundary.reserve(Subdomain<K>::_a->_nnz * interface.size() / Subdomain<K>::_dof);
                    for(i = 0; i < interface.size(); ++i)
                        tmpInteraction.reserve(Subdomain<K>::_a->_ia[interface.back()] - Subdomain<K>::_a->_ia[std::max(interface.back() - 2, static_cast<typename Container::value_type>(0))]);
                    _bb = new MatrixCSR<K, Wrapper<K>::I>(interface.size(), interface.size(), true);
                    int* ii = new int[Subdomain<K>::_dof + 1];
                    ii[0] = 0;
                    _bb->_ia[0] = (Wrapper<K>::I == 'F');
                    std::vector<int> boundaryCond;
                    if(!Subdomain<K>::_a->_sym)
                        boundaryCond.reserve(Subdomain<K>::_dof - interface.size());
                    for(i = 0; i < Subdomain<K>::_dof; ++i) {
                        signed int row = vec[i];
                        unsigned int stop;
                        if(!Subdomain<K>::_a->_sym) {
                            stop = std::distance(Subdomain<K>::_a->_ja, std::upper_bound(Subdomain<K>::_a->_ja + Subdomain<K>::_a->_ia[i], Subdomain<K>::_a->_ja + Subdomain<K>::_a->_ia[i + 1], i));
                            if(row < 0) {
                                bool isBoundaryCond = true;
                                for(j = Subdomain<K>::_a->_ia[i]; j < Subdomain<K>::_a->_ia[i + 1] && isBoundaryCond; ++j) {
                                    if(i != Subdomain<K>::_a->_ja[j] && std::abs(Subdomain<K>::_a->_a[j]) > HPDDM_EPS)
                                        isBoundaryCond = false;
                                    else if(i == Subdomain<K>::_a->_ja[j] && std::abs(Subdomain<K>::_a->_a[j] - 1.0) > HPDDM_EPS)
                                        isBoundaryCond = false;
                                }
                                if(isBoundaryCond)
                                    boundaryCond.push_back(-row);
                            }
                        }
                        else
                            stop = Subdomain<K>::_a->_ia[i + 1];
                        for(j = Subdomain<K>::_a->_ia[i]; j < stop; ++j) {
                            const K val = Subdomain<K>::_a->_a[j];
                            if(std::abs(val) > HPDDM_EPS) {
                                const int col = vec[Subdomain<K>::_a->_ja[j]];
                                if(col > 0) {
                                    if(row < 0)
                                        tmpInteraction[col - 1].emplace_back(-row - (Wrapper<K>::I != 'F'), val);
                                    else
                                        tmpBoundary.emplace_back(col - (Wrapper<K>::I != 'F'), val);
                                }
                                else if(Subdomain<K>::_a->_sym || col == row || !std::binary_search(boundaryCond.cbegin(), boundaryCond.cend(), -col)) {
                                    if(row < 0)
                                        tmpInterior.emplace_back(-col - 1, val);
                                    else
                                        tmpInteraction[row - 1].emplace_back(-col - (Wrapper<K>::I != 'F'), val);
                                }
                            }
                        }
                        if(row < 0)
                            ii[-row] = tmpInterior.size();
                        else
                            _bb->_ia[row] = tmpBoundary.size() + (Wrapper<K>::I == 'F');
                    }
                    for(i = 0; i < tmpInterior.size(); ++i) {
                        Subdomain<K>::_a->_ja[i] = tmpInterior[i].first;
                        Subdomain<K>::_a->_a[i] = tmpInterior[i].second;
                    }
                    for(i = 0, j = 0; i < tmpInteraction.size(); ++i) {
                        std::sort(tmpInteraction[i].begin(), tmpInteraction[i].end(), [](const std::pair<unsigned int, K>& lhs, const std::pair<unsigned int, K>& rhs) { return lhs.first < rhs.first; });
                        j += tmpInteraction[i].size();
                    }
                    _bi = new MatrixCSR<K, Wrapper<K>::I>(interface.size(), Subdomain<K>::_dof - interface.size(), j, false);
                    _bi->_ia[0] = (Wrapper<K>::I == 'F');
                    for(i = 0, j = 0; i < tmpInteraction.size(); ++i) {
                        for(const std::pair<unsigned int, K>& p : tmpInteraction[i]) {
                            _bi->_ja[j] = p.first;
                            _bi->_a[j++]  = p.second;
                        }
                        _bi->_ia[i + 1] = j + (Wrapper<K>::I == 'F');
                    }
                    _bb->_nnz = tmpBoundary.size();
                    _bb->_a = new K[_bb->_nnz];
                    _bb->_ja = new int[_bb->_nnz];
                    for(i = 0; i < tmpBoundary.size(); ++i) {
                        _bb->_ja[i] = tmpBoundary[i].first;
                        _bb->_a[i]  = tmpBoundary[i].second;
                    }
                    for(i = 0; i < _bb->_n; ++i) {
                        if(Wrapper<K>::I == 'F')
                            for(j = 0; j < _bi->_ia[i + 1] - _bi->_ia[i]; ++j)
                                Subdomain<K>::_a->_ja[ii[_bi->_m + i] + j] = *(_bi->_ja + _bi->_ia[i] - 1 + j) - 1;
                        else
                            std::copy(_bi->_ja + _bi->_ia[i], _bi->_ja + _bi->_ia[i + 1], Subdomain<K>::_a->_ja + ii[_bi->_m + i]);
                        std::copy(_bi->_a + _bi->_ia[i] - (Wrapper<K>::I == 'F'), _bi->_a + _bi->_ia[i + 1] - (Wrapper<K>::I == 'F'), Subdomain<K>::_a->_a + ii[_bi->_m + i]);
                        ii[_bi->_m + i + 1] = ii[_bi->_m + i] + _bi->_ia[i + 1] - _bi->_ia[i] - (_bb->_ia[i] - (Wrapper<K>::I == 'F'));
                        for(j = _bb->_ia[i] - (Wrapper<K>::I == 'F'); j < _bb->_ia[i + 1] - (Wrapper<K>::I == 'F'); ++j)
                            Subdomain<K>::_a->_ja[ii[_bi->_m + i + 1] + j] = _bb->_ja[j] - (Wrapper<K>::I == 'F') + _bi->_m;
                        std::copy(_bb->_a + _bb->_ia[i] - (Wrapper<K>::I == 'F'), _bb->_a + _bb->_ia[i + 1] - (Wrapper<K>::I == 'F'), Subdomain<K>::_a->_a + ii[_bi->_m + i + 1] + _bb->_ia[i] - (Wrapper<K>::I == 'F'));
                        ii[_bi->_m + i + 1] += _bb->_ia[i + 1] - (Wrapper<K>::I == 'F');
                    }
                    delete [] Subdomain<K>::_a->_ia;
                    Subdomain<K>::_a->_ia = ii;
                    Subdomain<K>::_a->_nnz = ii[Subdomain<K>::_dof];
                    Subdomain<K>::_a->_sym = true;
                    _ii = new MatrixCSR<K>(_bi->_m, _bi->_m, Subdomain<K>::_a->_ia[_bi->_m], Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, true);
                    Subdomain<K>::_dof = _bb->_n;
                }
                if(f) {
                    unsigned int i = 0, j = 0;
                    std::vector<K> fBoundary;
                    fBoundary.reserve(interface.size());
                    for(unsigned int k = 0; i < interface.size(); ++k) {
                        if(k == interface[i]) {
                            ++i;
                            fBoundary.emplace_back(f[k]);
                        }
                        else {
                            f[j] = f[k];
                            ++j;
                        }
                    }
                    std::copy(f + interface.back() + 1, f + _bi->_n + _bi->_m, f + j);
                    std::copy(fBoundary.cbegin(), fBoundary.cend(), f + _bi->_m);
                }
            }
            else {
                std::cerr << "The container of the interface is empty => no static condensation" << std::endl;
                Subdomain<K>::_dof = 0;
            }
        }
        /* Function: stiffnessScaling
         *
         *  Builds the stiffness scaling, cf. <Bdd::buildScaling> and <Feti::buildScaling>.
         *
         * Parameter:
         *    pt             - Reference to the array in which to store the values. */
        inline void stiffnessScaling(K* const& pt) {
            if(_bb) {
                for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i) {
                    unsigned int idx = _bb->_ia[i + 1] - (Wrapper<K>::I == 'F' ? 2 : 1);
                    if(_bb->_ja[idx] != i + (Wrapper<K>::I == 'F')) {
                        std::cerr << "The matrix '_bb' seems to be ill-formed" << std::endl;
                        pt[i] = 0;
                    }
                    else
                        pt[i] = _bb->_a[idx];
                }
            }
            else
                std::cerr << "The matrix '_bb' has not been allocated => impossible to build the stiffness scaling" << std::endl;
        }
        /* Function: getMult
         *  Returns the value of <Schur::mult>. */
        inline int getMult() const { return _mult; }
        /* Function: getSigned
         *  Returns the value of <Schur::signed>. */
        inline unsigned short getSigned() const { return _signed; }
        /* Function: applyLocalSchurComplement(n)
         *
         *  Applies the local Schur complement to multiple right-hand sides.
         *
         * Parameters:
         *    u              - Input vectors.
         *    n              - Number of input vectors.
         *
         * See also: <Feti::applyLocalPreconditioner(n)>. */
        inline void applyLocalSchurComplement(K*& in, const int& n) const {
            K* out = new K[n * Subdomain<K>::_dof]();
            if(!_schur) {
                K* tmp = new K[n * _bi->_m];
                Wrapper<K>::template csrgemm<Wrapper<K>::I>(&transb, &(Subdomain<K>::_dof), &n, &_bi->_m, &(Wrapper<K>::d__1), false, _bi->_a, _bi->_ia, _bi->_ja, in, &(Subdomain<K>::_dof), &(Wrapper<K>::d__0), tmp, &_bi->_m);
                super::_s.solve(tmp, n);
                Wrapper<K>::template csrgemm<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &n, &_bi->_m, &(Wrapper<K>::d__1), false, _bi->_a, _bi->_ia, _bi->_ja, tmp, &_bi->_m, &(Wrapper<K>::d__0), out, &(Subdomain<K>::_dof));
                Wrapper<K>::template csrgemm<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &n, &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), true, _bb->_a, _bb->_ia, _bb->_ja, in, &_bb->_m, &(Wrapper<K>::d__2), out, &(Subdomain<K>::_dof));
                delete [] tmp;
            }
            else
                Wrapper<K>::symm(&uplo, &uplo, &(Subdomain<K>::_dof), &n, &(Wrapper<K>::d__1), _schur, &(Subdomain<K>::_dof), in, &(Subdomain<K>::_dof), &(Wrapper<K>::d__0), out, &(Subdomain<K>::_dof));
            delete [] in;
            in = out;
        }
        /* Function: applyLocalSchurComplement
         *
         *  Applies the local Schur complement to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional).
         *
         * See also: <Feti::applyLocalPreconditioner> and <Bdd::apply>. */
        inline void applyLocalSchurComplement(K* const in, K* const& out = nullptr) const {
            if(!_schur) {
                Wrapper<K>::template csrgemv<Wrapper<K>::I>(&transb, &(Subdomain<K>::_dof), &_bi->_m, &(Wrapper<K>::d__1), false, _bi->_a, _bi->_ia, _bi->_ja, in, &(Wrapper<K>::d__0), _work);
                super::_s.solve(_work);
                if(out) {
                    Wrapper<K>::template csrgemv<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &_bi->_m, &(Wrapper<K>::d__1), false, _bi->_a, _bi->_ia, _bi->_ja, _work, &(Wrapper<K>::d__0), out);
                    Wrapper<K>::template csrgemv<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), true, _bb->_a, _bb->_ia, _bb->_ja, in, &(Wrapper<K>::d__2), out);
                }
                else {
                    Wrapper<K>::template csrgemv<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &_bi->_m, &(Wrapper<K>::d__1), false, _bi->_a, _bi->_ia, _bi->_ja, _work, &(Wrapper<K>::d__0), _work + _bi->_m);
                    Wrapper<K>::template csrgemv<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), true, _bb->_a, _bb->_ia, _bb->_ja, in, &(Wrapper<K>::d__2), _work + _bi->_m);
                    std::copy_n(_work + _bi->_m, Subdomain<K>::_dof, in);
                }
            }
            else if(out)
                Wrapper<K>::symv(&uplo, &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), _schur, &(Subdomain<K>::_dof), in, &i__1, &(Wrapper<K>::d__0), out, &i__1);
            else {
                Wrapper<K>::symv(&uplo, &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), _schur, &(Subdomain<K>::_dof), in, &i__1, &(Wrapper<K>::d__0), _work + _bi->_m, &i__1);
                std::copy_n(_work + _bi->_m, Subdomain<K>::_dof, in);
            }
        }
        /* Function: applyLocalLumpedMatrix(n)
         *
         *  Applies the local lumped matrix <Schur::bb> to multiple right-hand sides.
         *
         * Parameters:
         *    u              - Input vectors.
         *    n              - Number of input vectors.
         *
         * See also: <Feti::applyLocalPreconditioner(n)>. */
        inline void applyLocalLumpedMatrix(K*& in, const int& n) const {
            K* out = new K[n * Subdomain<K>::_dof];
            Wrapper<K>::template csrgemm<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &n, &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), true, _bb->_a, _bb->_ia, _bb->_ja, in, &(Subdomain<K>::_dof), &(Wrapper<K>::d__0), out, &(Subdomain<K>::_dof));
            delete [] in;
            in = out;
        }
        /* Function: applyLocalLumpedMatrix
         *
         *  Applies the local lumped matrix <Schur::bb> to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional).
         *
         * See also: <Feti::applyLocalPreconditioner>. */
        inline void applyLocalLumpedMatrix(K* const in) const {
            Wrapper<K>::template csrgemv<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &(Subdomain<K>::_dof), &(Wrapper<K>::d__1), true, _bb->_a, _bb->_ia, _bb->_ja, in, &(Wrapper<K>::d__0), _work + _bi->_m);
            std::copy_n(_work + _bi->_m, Subdomain<K>::_dof, in);
        }
        /* Function: applyLocalSuperlumpedMatrix(n)
         *
         *  Applies the local superlumped matrix diag(<Schur::bb>) to multiple right-hand sides.
         *
         * Parameters:
         *    u              - Input vectors.
         *    n              - Number of input vectors.
         *
         * See also: <Feti::applyLocalPreconditioner(n)>. */
        inline void applyLocalSuperlumpedMatrix(K*& in, const int& n) const {
            for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i) {
                K d = _bb->_a[_bb->_ia[i + 1] - (Wrapper<K>::I == 'F' ? 2 : 1)];
                for(int j = 0; j < n; ++j)
                    in[i + j * Subdomain<K>::_dof] *= d;
            }
        }
        /* Function: applyLocalSuperlumpedMatrix
         *
         *  Applies the local superlumped matrix diag(<Schur::bb>) to a single right-hand side.
         *
         * Parameters:
         *    in             - Input vector.
         *    out            - Output vector (optional).
         *
         * See also: <Feti::applyLocalPreconditioner>. */
        inline void applyLocalSuperlumpedMatrix(K* const in) const {
            for(unsigned int i = 0; i < Subdomain<K>::_dof; ++i)
                in[i] *= _bb->_a[_bb->_ia[i + 1] - (Wrapper<K>::I == 'F' ? 2 : 1)];
        }
        /* Function: getRank
         *  Returns the value of <Schur::rankWorld>. */
        inline int getRank() const { return _rankWorld; }
        /* Function: getLDR
         *  Returns the address of the leading dimension of <Preconditioner::ev>. */
        inline const int* getLDR() const {
            return _schur ? &_bi->_n : &(super::_a->_n);
        }
        /* Function: getEliminated
         *  Returns the number of eliminated unknowns of <Subdomain<K>::a>, i.e. the number of columns of <Schur::bi>. */
        inline unsigned int getEliminated() const {
            return _bi ? _bi->_m : 0;
        }
        /* Function: condensateEffort
         *
         *  Performs static condensation.
         *
         * Parameters:
         *    f              - Input right-hand side.
         *    b              - Condensed right-hand side. */
        inline void condensateEffort(const K* const f, K* const b) const {
            super::_s.solve(f, _structure);
            std::copy_n(f + _bi->_m, Subdomain<K>::_dof, b ? b : _structure + _bi->_m);
            Wrapper<K>::template csrgemv<Wrapper<K>::I>(&transa, &(Subdomain<K>::_dof), &_bi->_m, &(Wrapper<K>::d__2), false, _bi->_a, _bi->_ia, _bi->_ja, _structure, &(Wrapper<K>::d__1), b ? b : _structure + _bi->_m);
        }
        /* Function: computeError
         *
         *  Computes the Euclidean norm of a right-hand side and of the difference between a solution vector and a right-hand side.
         *
         * Parameters:
         *    x              - Solution vector.
         *    f              - Right-hand side.
         *    storage        - Array to store both values.
         *
         * See also: <Schwarz::computeError>. */
        inline void computeError(const K* const x, const K* const f, double* const residual) const {
            residual[0] = Wrapper<K>::dot(&(Subdomain<K>::_a->_n), f, &i__1, f, &i__1);
            K* tmp = new K[Subdomain<K>::_a->_n];
            std::copy(f, f + Subdomain<K>::_a->_n, tmp);
            Subdomain<K>::exchange(tmp + _bi->_m);
            for(unsigned short i = 0; i < Subdomain<K>::_map.size(); ++i)
                for(unsigned int j = 0; j < Subdomain<K>::_map[i].second.size(); ++j)
                    residual[0] += std::real(std::conj(f[_bi->_m + Subdomain<K>::_map[i].second[j]]) * Subdomain<K>::_rbuff[i][j]);
            Wrapper<K>::template csrmv<'C'>(Subdomain<K>::_a->_sym, &(Subdomain<K>::_a->_n), Subdomain<K>::_a->_a, Subdomain<K>::_a->_ia, Subdomain<K>::_a->_ja, x, _work);
            Subdomain<K>::exchange(_work + _bi->_m);
            Wrapper<K>::axpy(&(Subdomain<K>::_a->_n), &(Wrapper<K>::d__2), tmp, &i__1, _work, &i__1);
            residual[1] = Wrapper<K>::dot(&_bi->_m, _work, &i__1, _work, &i__1);
            std::fill(tmp, tmp + Subdomain<K>::_dof, 1.0);
            for(const pairNeighbor& neighbor : Subdomain<K>::_map)
                for(const pairNeighbor::second_type::value_type& val : neighbor.second)
                        tmp[val] /= (1.0 + tmp[val]);
            for(unsigned short i = 0; i < Subdomain<K>::_dof; ++i)
                    residual[1] += std::real(tmp[i]) * std::norm(_work[_bi->_m + i]);
            delete [] tmp;
            MPI_Allreduce(MPI_IN_PLACE, residual, 2, MPI_DOUBLE, MPI_SUM, Subdomain<K>::_communicator);
            residual[0] = std::sqrt(residual[0]);
            residual[1] = std::sqrt(residual[1]);
        }
        /* Function: getAllDof
         *  Returns the number of local interior and boundary degrees of freedom (with the right multiplicity). */
        inline unsigned int getAllDof() const {
            unsigned int dof = Subdomain<K>::_a->_n;
            for(unsigned int k = 0; k < Subdomain<K>::_dof; ++k) {
                bool exit = false;
                for(unsigned short i = 0; i < Subdomain<K>::_map.size() && Subdomain<K>::_map[i].first < _rankWorld && !exit; ++i)
                    for(unsigned int j = 0; j < Subdomain<K>::_map[i].second.size() && !exit; ++j)
                        if(Subdomain<K>::_map[i].second[j] == k) {
                            --dof;
                            exit = true;
                        }
            }
            return dof;
        }
        inline void distributedNumbering(unsigned int* in, unsigned int& first, unsigned int& last, unsigned int& global) const {
            Subdomain<K>::template globalMapping<'C'>(in, in + Subdomain<K>::_dof, first, last, global);
        }
        inline bool distributedCSR(unsigned int* num, unsigned int first, unsigned int last, int*& ia, int*& ja, K*& c) const {
            Subdomain<K>::distributedCSR(num, first, last, ia, ja, c, _bb);
        }
};
} // HPDDM
#endif // _SCHUR_
