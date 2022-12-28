/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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

#ifndef HPDDM_DENSE_HPP_
#define HPDDM_DENSE_HPP_

#include "HPDDM_schwarz.hpp"

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
        Dense(const Subdomain<K>& s) : super(s) { super::d_ = nullptr; }
        ~Dense() { super::d_ = nullptr; }
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
            super::s_.numfact(A);
            if(resetPrefix)
                opt.setPrefix("");
        }
        void setMatrix(const int& n, const bool sym, K* const& p) {
            MatrixCSR<K>* A = new MatrixCSR<K>(n, n, n * n, p, nullptr, nullptr, sym);
            setMatrix(A);
        }
        void solveEVP(const K* const A) {
            const std::string prefix = super::prefix();
            Option& opt = *Option::get();
            const underlying_type<K>& threshold = opt.val(prefix + "geneo_threshold", -1.0);
            if(super::ev_) {
                delete [] *super::ev_;
                delete [] super::ev_;
            }
#ifdef MU_ARPACK
            K* const a = new K[Subdomain<K>::dof_ * Subdomain<K>::dof_];
            std::copy_n(A, Subdomain<K>::dof_ * Subdomain<K>::dof_, a);
            MatrixCSR<K> rhs(Subdomain<K>::dof_, Subdomain<K>::dof_, Subdomain<K>::dof_ * Subdomain<K>::dof_, a, nullptr, nullptr, false, true);
            K* const b = new K[Subdomain<K>::dof_ * Subdomain<K>::dof_]();
            for(unsigned int i = 0; i < Subdomain<K>::dof_; ++i)
                b[(Subdomain<K>::dof_ + 1) * i] = Wrapper<K>::d__1;
            MatrixCSR<K> lhs(Subdomain<K>::dof_, Subdomain<K>::dof_, Subdomain<K>::dof_ * Subdomain<K>::dof_, b, nullptr, nullptr, false, true);
            EIGENSOLVER<K> evp(threshold, Subdomain<K>::dof_, opt.val<unsigned short>("geneo_nu", 20));
            evp.template solve<SUBDOMAIN>(&lhs, &rhs, super::ev_, Subdomain<K>::communicator_, nullptr);
            int mm = evp.nu_;
            const int n = Subdomain<K>::dof_;
            std::for_each(super::ev_, super::ev_ + evp.nu_, [&](K* const v) { std::replace_if(v, v + n, [](K x) { return std::abs(x) < 1.0 / (HPDDM_EPS * HPDDM_PEN); }, K()); });
#else
            K* H = new K[std::max(2, Subdomain<K>::dof_ * (Subdomain<K>::dof_ + 1))]();
            int info;
            int lwork = -1;
            {
                Lapack<K>::gehrd(&(Subdomain<K>::dof_), &i__1, &(Subdomain<K>::dof_), nullptr, &(Subdomain<K>::dof_), nullptr, H, &lwork, &info);
                Lapack<K>::hseqr("E", "N", &(Subdomain<K>::dof_), &i__1, &(Subdomain<K>::dof_), nullptr, &(Subdomain<K>::dof_), nullptr, nullptr, nullptr, &i__1, H + 1, &lwork, &info);
                lwork = std::max(Subdomain<K>::dof_ * (Subdomain<K>::dof_ + (Wrapper<K>::is_complex ? 0 : 2)), static_cast<int>(std::max(std::real(H[0]), std::real(H[1]))));
            }
            K* work = new K[lwork]();
            std::copy_n(A, Subdomain<K>::dof_ * Subdomain<K>::dof_, H);
            Lapack<K>::gehrd(&(Subdomain<K>::dof_), &i__1, &(Subdomain<K>::dof_), H, &(Subdomain<K>::dof_), H + Subdomain<K>::dof_ * Subdomain<K>::dof_, work, &lwork, &info);
            K* w = new K[Wrapper<K>::is_complex ? Subdomain<K>::dof_ : (2 * Subdomain<K>::dof_)];
            K* backup = new K[Subdomain<K>::dof_ * Subdomain<K>::dof_];
            std::copy_n(H, Subdomain<K>::dof_ * Subdomain<K>::dof_, backup);
            Lapack<K>::hseqr("E", "N", &(Subdomain<K>::dof_), &i__1, &(Subdomain<K>::dof_), backup, &(Subdomain<K>::dof_), w, w + Subdomain<K>::dof_, nullptr, &i__1, work, &lwork, &info);
            delete [] backup;
            std::vector<std::pair<unsigned short, std::complex<underlying_type<K>>>> q;
            q.reserve(Subdomain<K>::dof_);
            selectNu(HPDDM_RECYCLE_TARGET_LM, q, Subdomain<K>::dof_, w, w + Subdomain<K>::dof_);
            int k;
            if(threshold > 0.0)
                k = std::distance(q.begin(), std::lower_bound(q.begin() + 1, q.end(), std::pair<unsigned short, std::complex<underlying_type<K>>>(0, threshold), [](const std::pair<unsigned short, std::complex<underlying_type<K>>>& lhs, const std::pair<unsigned short, std::complex<underlying_type<K>>>& rhs) { return std::norm(lhs.second) > std::norm(rhs.second); }));
            else
                k = opt.val<int>(prefix + "geneo_nu", 20);
            q.resize(k);
            int mm = Wrapper<K>::is_complex ? k : 0;
            int* select = new int[Subdomain<K>::dof_]();
            for(typename decltype(q)::const_iterator it = q.cbegin(); it < q.cend(); ++it) {
                if(Wrapper<K>::is_complex)
                    select[it->first] = 1;
                else {
                    if(std::abs(w[Subdomain<K>::dof_ + it->first]) < HPDDM_EPS) {
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
            underlying_type<K>* rwork = Wrapper<K>::is_complex ? new underlying_type<K>[Subdomain<K>::dof_] : nullptr;
            super::ev_ = new K*[mm];
            *super::ev_ = new K[mm * Subdomain<K>::dof_];
            for(unsigned short i = 1; i < mm; ++i)
                super::ev_[i] = *super::ev_ + i * Subdomain<K>::dof_;
            int* ifailr = new int[mm];
            int col;
            Lapack<K>::hsein("R", "Q", "N", select, &(Subdomain<K>::dof_), H, &(Subdomain<K>::dof_), w, w + Subdomain<K>::dof_, nullptr, &i__1, *super::ev_, &(Subdomain<K>::dof_), &mm, &col, work, rwork, nullptr, ifailr, &info);
            Lapack<K>::mhr("L", "N", &(Subdomain<K>::dof_), &mm, &i__1, &(Subdomain<K>::dof_), H, &(Subdomain<K>::dof_), H + Subdomain<K>::dof_ * Subdomain<K>::dof_, *super::ev_, &(Subdomain<K>::dof_), work, &lwork, &info);
            delete [] ifailr;
            delete [] select;
            delete [] rwork;
            delete [] w;
            delete [] work;
            delete [] H;
#endif
            opt[prefix + "geneo_nu"] = mm;
            if(super::co_)
                super::co_->setLocal(mm);
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
                    S, K>* const A_;
                const K* const   E_;
                ClassWithPtr(const Dense<
#if HPDDM_DENSE
                    Solver, CoarseSolver,
#endif
                    S, K>* const A, const K* const E) : A_(A), E_(E) { }
                const MPI_Comm& getCommunicator() const { return A_->getCommunicator(); }
                const vectorNeighbor& getMap() const { return A_->getMap(); }
                constexpr int getDof() const { return A_->getDof(); }
                constexpr unsigned short getLocal() const { return A_->getLocal(); }
                const K* const* getVectors() const { return A_->getVectors(); }
                const K* getOperator() const { return E_; }
            };
            ClassWithPtr Op(this, E);
            return super::super::template buildTwo<excluded, UserCoarseOperator<ClassWithPtr, K>>(&Op, comm);
        }
        static constexpr std::unordered_map<unsigned int, K> boundaryConditions() { return std::unordered_map<unsigned int, K>(); }
        virtual int GMV(const K* const in, K* const out, const int& mu = 1) const override = 0;
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
#endif // HPDDM_DENSE_HPP_
