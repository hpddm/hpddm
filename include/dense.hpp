/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2017-09-07

   Copyright (C) 2017-     Centre National de la Recherche Scientifique

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

#ifndef _HPDDM_DENSE_
#define _HPDDM_DENSE_

#include "schwarz.hpp"

namespace HPDDM {
template<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    template<class> class Solver, template<class> class CoarseSolver,
#endif
    char S, class K>
class Dense : public Schwarz<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
                Solver, CoarseSolver,
#endif
                S, K> {
    public:
        Dense() { }
        Dense(const Subdomain<K>& s) : super(s) { super::_d = nullptr; }
        ~Dense() { super::_d = nullptr; }
        /* Typedef: super
         *  Type of the immediate parent class <Preconditioner>. */
        typedef Schwarz<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
            Solver, CoarseSolver,
#endif
            S, K> super;
        template<class Neighbor, class Mapping>
        void initialize(const int& n, const bool sym, K* const& p, const Neighbor& o, const Mapping& r, MPI_Comm* const& comm = nullptr) {
            MatrixCSR<K>* A = new MatrixCSR<K>(n, n, n * n, p, nullptr, nullptr, sym);
            Subdomain<K>::initialize(A, o, r, comm);
        }
        template<char N = HPDDM_NUMBERING>
        void callNumfact(MatrixCSR<K>* const& A = nullptr) {
            super::_s.template numfact<N>(Subdomain<K>::_a);
        }
        void setMatrix(MatrixCSR<K>* const& A) {
            super::setMatrix(A);
            super::destroySolver();
            super::_s.numfact(A);
        }
        void setMatrix(const int& n, const bool sym, K* const& p) {
            MatrixCSR<K>* A = new MatrixCSR<K>(n, n, n * n, p, nullptr, nullptr, sym);
            super::setMatrix(A);
            super::destroySolver();
            super::_s.numfact(A);
        }
        template<bool excluded = false>
        void apply(const K* const in, K* const out, const unsigned short& mu = 1, K* work = nullptr) const {
            super::_s.solve(in, out, mu);
            super::scaledExchange(out, mu);
        }
        static constexpr std::unordered_map<unsigned int, K> boundaryConditions() { return std::unordered_map<unsigned int, K>(); }
    private:
        using super::GMV;
};
} // HPDDM

#endif // _HPDDM_DENSE_
