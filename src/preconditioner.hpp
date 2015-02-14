/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2012-12-15

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

#ifndef _PRECONDITIONER_
#define _PRECONDITIONER_

#include "subdomain.hpp"

namespace HPDDM {
/* Class: Preconditioner
 *
 *  A base class from which <Schwarz> and <Schur> inherit.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of local matrices.
 *    CoarseOperator - Class of the coarse operator.
 *    K              - Scalar type. */
template<template<class> class Solver, class CoarseOperator, class K>
class Preconditioner : public Subdomain<K> {
    protected:
        /* Variable: s
         *  Solver used in <Schwarz::callNumfact> and <Schur::callNumfactPreconditioner> or <Schur::computeSchurComplement>. */
        Solver<K>           _s;
        /* Variable: co
         *  Pointer to a <Coarse operator>. */
        CoarseOperator*    _co;
        /* Variable: ev
         *  Array of deflation vectors as needed by <Preconditioner::co>. */
        K**                _ev;
        /* Variable: uc
         *  Workspace array of size <Coarse operator::local>. */
        K*                 _uc;
    public:
        Preconditioner() : _co(), _ev(), _uc() { }
        Preconditioner(const Preconditioner&) = delete;
        ~Preconditioner() {
            delete _co;
            if(_ev)
                delete [] *_ev;
            delete [] _ev;
            delete [] _uc;
        }
        /* Function: initialize
         *
         *  Initializes a two-level preconditioner.
         *
         * Parameter:
         *    deflation      - Number of local deflation vectors. */
        inline void initialize(const unsigned short& deflation) {
            if(!_co) {
                _co = new CoarseOperator;
                _co->setLocal(deflation);
            }
        }
        /* Function: callSolve
         *
         *  Applies <Preconditioner::s> to multiple right-hand sides in-place.
         *
         * Parameters:
         *    x              - Input right-hand sides, solution vectors are stored in-place.
         *    n              - Number of input right-hand sides. */
        inline void callSolve(K* const x, const unsigned short& n = 1) const {
            _s.solve(x, n);
        }
        /* Function: buildTwo
         *
         *  Assembles and factorizes the coarse operator.
         *
         * Template Parameter:
         *    excluded       - Greater than 0 if the master processes are excluded from the domain decomposition, equal to 0 otherwise.
         *
         * Parameters:
         *    A              - Operator used in the definition of the Galerkin matrix.
         *    comm           - Global MPI communicator.
         *    parm           - Vector of parameters. */
        template<unsigned short excluded, unsigned short N, class Operator, class Container>
        inline std::pair<MPI_Request, const K*>* buildTwo(Operator&& A, const MPI_Comm& comm, Container& parm) {
            static_assert(N == 2 || N == 3, "Wrong template parameter");
            std::pair<MPI_Request, const K*>* ret = nullptr;
            unsigned short allUniform[N + 1];
            allUniform[0] = Subdomain<K>::_map.size();
            allUniform[1] = parm[NU];
            allUniform[2] = static_cast<unsigned short>(~parm[NU]);
            if(N == 3)
                allUniform[3] = parm[NU] > 0 ? parm[NU] : std::numeric_limits<unsigned short>::max();
            {
                auto f = [](void* in, void* inout, int*, MPI_Datatype*) -> void {
                    unsigned short* input = static_cast<unsigned short*>(in);
                    unsigned short* output = static_cast<unsigned short*>(inout);
                    output[0] = std::max(output[0], input[0]);
                    output[1] = output[1] & input[1];
                    output[2] = output[2] & input[2];
                    if(N == 3)
                        output[3] = output[3] & input[3];
                };
                MPI_Op op;
                MPI_Op_create(f, 1, &op);
                MPI_Allreduce(MPI_IN_PLACE, allUniform, N + 1, MPI_UNSIGNED_SHORT, op, comm);
                MPI_Op_free(&op);
            }
            A.sparsity(allUniform[0]);
            if(parm[NU] > 0 || allUniform[1] != 0 || allUniform[2] != std::numeric_limits<unsigned short>::max()) {
                if(!_co)
                    _co = new CoarseOperator;

                _co->setLocal(parm[NU]);

                MPI_Barrier(comm);
                double construction = MPI_Wtime();
                if(allUniform[1] == parm[NU] && allUniform[2] == static_cast<unsigned short>(~parm[NU]))
                    ret = _co->template construction<1, excluded>(A, comm, parm);
                else if(N == 3 && allUniform[1] == 0 && allUniform[2] == static_cast<unsigned short>(~allUniform[3]))
                    ret = _co->template construction<2, excluded>(A, comm, parm);
                else
                    ret = _co->template construction<0, excluded>(A, comm, parm);
                construction = MPI_Wtime() - construction;
                if(_co->getRank() == 0) {
                    std::cout << "                 (" << parm[P] << " process" << (parm[P] > 1 ? "es" : "") << " -- topology = " << parm[TOPOLOGY] << " -- distribution = " << _co->getDistribution() << ")" << std::endl;
                    std::cout << std::scientific << " --- coarse operator transferred and factorized (in " << construction << ")" << std::endl;
                    std::cout << "                                     (criterion: " << (allUniform[1] == parm[NU] && allUniform[2] == static_cast<unsigned short>(~parm[NU]) ? parm[NU] : (N == 3 && allUniform[2] == static_cast<unsigned short>(~allUniform[3]) ? -_co->getLocal() : 0)) << ")" << std::endl;
                }
                _uc = new K[_co->getSizeRHS()];
            }
            return ret;
        }
        /* Function: getVectors
         *  Returns a constant pointer to <Preconditioner::ev>. */
        inline K** getVectors() const { return _ev; }
        /* Function: setVectors
         *  Sets the pointer <Preconditioner::ev>. */
        inline void setVectors(K** const& ev) { _ev = ev; }
        /* Function: getLocal
         *  Returns the value of <Coarse operator::local>. */
        inline unsigned short getLocal() const { return _co ? _co->getLocal() : 0; }
        /* Function: getAddrLocal
         *  Returns the address of <Coarse operator::local> or <i__0> if <Preconditioner::co> is not allocated. */
        inline const int* getAddrLocal() const { return _co ? _co->getAddrLocal() : &i__0; }
};
} // HPDDM
#endif // _PRECONDITIONER_
