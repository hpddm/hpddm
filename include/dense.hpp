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
#if HPDDM_DENSE
    template<class> class Solver, template<class> class CoarseSolver,
#endif
    char S, class K>
class Dense : public Schwarz<
#if HPDDM_DENSE
                Solver, CoarseSolver,
#endif
                S, K> {
    public:
        Dense() : super() { }
        Dense(const Subdomain<K>& s) : super(s) { super::_d = nullptr; }
        ~Dense() { super::_d = nullptr; }
        /* Typedef: super
         *  Type of the immediate parent class <Preconditioner>. */
        typedef Schwarz<
#if HPDDM_DENSE
            Solver, CoarseSolver,
#endif
            S, K> super;
        template<class Neighbor, class Mapping>
        void initialize(const int& n, const bool sym, K* const& p, const Neighbor& o, const Mapping& r, MPI_Comm* const& comm = nullptr) {
            MatrixCSR<K>* A = new MatrixCSR<K>(n, n, n * n, p, nullptr, nullptr, sym);
            Subdomain<K>::initialize(A, o, r, comm);
        }
        void setMatrix(MatrixCSR<K>* const& A) {
            Option& opt = *Option::get();
            const bool resetPrefix = (opt.getPrefix().size() == 0 && super::prefix().size() != 0);
            if(resetPrefix)
                opt.setPrefix(super::prefix());
            super::setMatrix(A);
            super::destroySolver();
            super::_s.numfact(A);
            if(resetPrefix)
                opt.setPrefix("");
        }
        void setMatrix(const int& n, const bool sym, K* const& p) {
            MatrixCSR<K>* A = new MatrixCSR<K>(n, n, n * n, p, nullptr, nullptr, sym);
            setMatrix(A);
        }
        void solveEVP(const K* const A) {
            K* H = new K[std::max(2, Subdomain<K>::_dof * (Subdomain<K>::_dof + 1))]();
            int info;
            int lwork = -1;
            {
                Lapack<K>::gehrd(&(Subdomain<K>::_dof), &i__1, &(Subdomain<K>::_dof), nullptr, &(Subdomain<K>::_dof), nullptr, H, &lwork, &info);
                Lapack<K>::hseqr("E", "N", &(Subdomain<K>::_dof), &i__1, &(Subdomain<K>::_dof), nullptr, &(Subdomain<K>::_dof), nullptr, nullptr, nullptr, &i__1, H + 1, &lwork, &info);
                lwork = std::max(Subdomain<K>::_dof * (Subdomain<K>::_dof + (Wrapper<K>::is_complex ? 0 : 2)), static_cast<int>(std::max(std::real(H[0]), std::real(H[1]))));
            }
            K* work = new K[lwork]();
            std::copy_n(A, Subdomain<K>::_dof * Subdomain<K>::_dof, H);
            Lapack<K>::gehrd(&(Subdomain<K>::_dof), &i__1, &(Subdomain<K>::_dof), H, &(Subdomain<K>::_dof), H + Subdomain<K>::_dof * Subdomain<K>::_dof, work, &lwork, &info);
            K* w = new K[Wrapper<K>::is_complex ? Subdomain<K>::_dof : (2 * Subdomain<K>::_dof)];
            K* backup = new K[Subdomain<K>::_dof * Subdomain<K>::_dof];
            std::copy_n(H, Subdomain<K>::_dof * Subdomain<K>::_dof, backup);
            Lapack<K>::hseqr("E", "N", &(Subdomain<K>::_dof), &i__1, &(Subdomain<K>::_dof), backup, &(Subdomain<K>::_dof), w, w + Subdomain<K>::_dof, nullptr, &i__1, work, &lwork, &info);
            delete [] backup;
            std::vector<std::pair<unsigned short, std::complex<underlying_type<K>>>> q;
            q.reserve(Subdomain<K>::_dof);
            selectNu(HPDDM_RECYCLE_TARGET_LM, q, Subdomain<K>::_dof, w, w + Subdomain<K>::_dof);
            const std::string prefix = super::prefix();
            Option& opt = *Option::get();
            const underlying_type<K>& threshold = opt.val(prefix + "geneo_threshold", -1.0);
            int k;
            if(threshold > 0.0)
                k = std::distance(q.begin(), std::lower_bound(q.begin() + 1, q.end(), std::pair<unsigned short, std::complex<underlying_type<K>>>(0, threshold), [](const std::pair<unsigned short, std::complex<underlying_type<K>>>& lhs, const std::pair<unsigned short, std::complex<underlying_type<K>>>& rhs) { return std::norm(lhs.second) > std::norm(rhs.second); }));
            else
                k = opt.val<int>(prefix + "geneo_nu", 20);
            q.resize(k);
            int mm = Wrapper<K>::is_complex ? k : 0;
            int* select = new int[Subdomain<K>::_dof]();
            for(typename decltype(q)::const_iterator it = q.cbegin(); it < q.cend(); ++it) {
                if(Wrapper<K>::is_complex)
                    select[it->first] = 1;
                else {
                    if(std::abs(w[Subdomain<K>::_dof + it->first]) < HPDDM_EPS) {
                        select[it->first] = 1;
                        ++mm;
                    }
                    else if(mm < k + 1) {
                        select[it->first] = 1;
                        mm += 2;
                        ++it;
                    }
                    else
                        break;
                }
            }
            decltype(q)().swap(q);
            underlying_type<K>* rwork = Wrapper<K>::is_complex ? new underlying_type<K>[Subdomain<K>::_dof] : nullptr;
            if(super::_ev) {
                if(*super::_ev)
                    delete [] *super::_ev;
                delete [] super::_ev;
            }
            super::_ev = new K*[mm];
            *super::_ev = new K[mm * Subdomain<K>::_dof];
            for(unsigned short i = 1; i < mm; ++i)
                super::_ev[i] = *super::_ev + i * Subdomain<K>::_dof;
            int* ifailr = new int[mm];
            int col;
            Lapack<K>::hsein("R", "Q", "N", select, &(Subdomain<K>::_dof), H, &(Subdomain<K>::_dof), w, w + Subdomain<K>::_dof, nullptr, &i__1, *super::_ev, &(Subdomain<K>::_dof), &mm, &col, work, rwork, nullptr, ifailr, &info);
            Lapack<K>::mhr("L", "N", &(Subdomain<K>::_dof), &mm, &i__1, &(Subdomain<K>::_dof), H, &(Subdomain<K>::_dof), H + Subdomain<K>::_dof * Subdomain<K>::_dof, *super::_ev, &(Subdomain<K>::_dof), work, &lwork, &info);
            opt[prefix + "geneo_nu"] = mm;
            if(super::_co)
                super::_co->setLocal(mm);
            delete [] ifailr;
            delete [] select;
            delete [] rwork;
            delete [] w;
            delete [] work;
            delete [] H;
        }
        template<unsigned short excluded = 0>
        std::pair<MPI_Request, const K*>* buildTwo(const MPI_Comm& comm, const K* const E) {
            struct ClassWithPtr {
                typedef Schwarz<
#if HPDDM_DENSE
                    Solver, CoarseSolver,
#endif
                    S, K> super;
                const Dense<
#if HPDDM_DENSE
                    Solver, CoarseSolver,
#endif
                    S, K>* const _A;
                const K* const   _E;
                ClassWithPtr(const Dense<
#if HPDDM_DENSE
                    Solver, CoarseSolver,
#endif
                    S, K>* const A, const K* const E) : _A(A), _E(E) { }
                const MPI_Comm& getCommunicator() const { return _A->getCommunicator(); }
                const vectorNeighbor& getMap() const { return _A->getMap(); }
                int getDof() const { return _A->getDof(); }
                unsigned short getLocal() const { return _A->getLocal(); }
                const K* const* getVectors() const { return _A->getVectors(); }
                const K* getOperator() const { return _E; }
            };
            ClassWithPtr Op(this, E);
            return super::super::template buildTwo<excluded, UserCoarseOperator<ClassWithPtr, K>>(&Op, comm);
        }
        static constexpr std::unordered_map<unsigned int, K> boundaryConditions() { return std::unordered_map<unsigned int, K>(); }
        virtual void GMV(const K* const in, K* const out, const int& mu = 1) const = 0;
};

template<
#if HPDDM_DENSE
    template<class> class Solver, template<class> class CoarseSolver,
#endif
    char S, class K>
struct hpddm_method_id<Dense<
#if HPDDM_DENSE
    Solver, CoarseSolver,
#endif
    S, K>> { static constexpr char value = 4; };
} // HPDDM
#endif // _HPDDM_DENSE_
