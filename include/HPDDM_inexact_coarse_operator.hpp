/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2016-09-13

   Copyright (C) 2016-     Centre National de la Recherche Scientifique

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

#ifndef HPDDM_INEXACT_COARSE_OPERATOR_HPP_
#define HPDDM_INEXACT_COARSE_OPERATOR_HPP_

#if !HPDDM_PETSC
# if !defined(PETSCSUB)
class Mat;
# endif
#else
# include "HPDDM_subdomain.hpp"
# if HPDDM_SLEPC
#  include <slepc.h>
# endif
#endif

namespace HPDDM {
HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K) class CoarseOperator;
template<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
    template<class> class Solver, template<class> class CoarseSolver, char S,
#endif
    class K>
class Schwarz;

HPDDM_CLASS_COARSE_OPERATOR(Solver, S, K)
class InexactCoarseOperator : public OptionsPrefix<K>, public Solver
#if !HPDDM_PETSC
                                                                    <K>
#endif
    {
    protected:
#if !HPDDM_PETSC
        typedef typename Subdomain<K>::integer_type integer_type;
        typedef void                                 return_type;
        Schwarz<SUBDOMAIN, COARSEOPERATOR, S, K>*       s_;
        const Schwarz<SUBDOMAIN, COARSEOPERATOR, S, K>* p_;
        MPI_Comm communicator_;
        K**              buff_;
        MPI_Request*       rq_;
        K*                  x_;
        K*                  o_;
        unsigned short     mu_;
#else
        typedef PetscErrorCode return_type;
        typedef PetscInt      integer_type;
        PC_HPDDM_Level*                 s_;
        PetscInt*                     idx_;
#endif
        vectorNeighbor   recv_;
        std::map<unsigned short, std::vector<int>> send_;
        K*                 da_;
        K*                 oa_;
        integer_type*      di_;
        integer_type*      oi_;
        integer_type*      dj_;
        integer_type*      oj_;
        integer_type*     ogj_;
        unsigned int*   range_;
        int               dof_;
        int               off_;
        int                bs_;
        template<char
#if !HPDDM_PETSC
                      T
#else
                      S
#endif
                       , bool factorize>
        void numfact(unsigned int nrow, integer_type* I, int* loc2glob, integer_type* J, K* C, unsigned short* neighbors) {
            da_ = C;
            dj_ = J;
#if !HPDDM_PETSC
            Option& opt = *Option::get();
#ifdef DMKL_PARDISO
            if(factorize) {
                range_ = new unsigned int[3];
                std::copy_n(loc2glob, 2, range_);
                range_[2] = 0;
            }
#else
            if(factorize && S == 'S') {
                range_ = new unsigned int[2];
                std::copy_n(loc2glob, 2, range_);
            }
#endif
            MPI_Comm_dup(DMatrix::communicator_, &communicator_);
            MPI_Comm_size(communicator_, &off_);
            if(off_ > 1) {
                unsigned int accumulate = 0;
                {
                    int* ia = nullptr;
                    K* a;
                    if(mu_ < off_) {
                        MPI_Group world, aggregate;
                        MPI_Comm_group(communicator_, &world);
                        int ranges[1][3];
                        ranges[0][0] = (DMatrix::rank_ / mu_) * mu_;
                        ranges[0][1] = std::min(off_, ranges[0][0] + mu_) - 1;
                        ranges[0][2] = 1;
                        MPI_Group_range_incl(world, 1, ranges, &aggregate);
                        MPI_Comm_free(&(DMatrix::communicator_));
                        MPI_Comm_create(communicator_, aggregate, &(DMatrix::communicator_));
                        MPI_Group_free(&aggregate);
                        MPI_Group_free(&world);
                        bool r = false;
                        std::vector<int*> range;
                        range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
                        range.emplace_back(J);
                        int* R = new int[nrow + 1];
                        R[0] = (super::numbering_ == 'F');
                        for(unsigned int i = 0; i < nrow; ++i) {
                            R[i + 1] = R[i];
                            for(unsigned int j = 0; j < I[i + 1]; ++j) {
                                const int k = accumulate + j;
                                if(ranges[0][0] <= neighbors[k] && neighbors[k] <= ranges[0][1]) {
                                    R[i + 1]++;
                                    if(r) {
                                        r = false;
                                        range.emplace_back(J + k);
                                    }
                                }
                                else if(!r) {
                                    r = true;
                                    range.emplace_back(J + k);
                                }
                            }
                            accumulate += I[i + 1];
                        }
                        if(range.size() > 1) {
                            range.emplace_back(J + accumulate);
                            std::vector<int*>::const_iterator it;
                            a = new K[(R[nrow] - (super::numbering_ == 'F')) * bs_ * bs_];
                            ia = new int[nrow + 1 + R[nrow] - (super::numbering_ == 'F')];
                            accumulate = 0;
                            ia += nrow + 1;
                            for(it = range.cbegin(); it < range.cend() - 1; it += 2) {
                                unsigned int size = *(it + 1) - *it;
                                for(unsigned int i = 0; i < size; ++i) {
                                    ia[accumulate + i] = (*it)[i] - di_[0];
                                    if(T == 1 && (*it)[i] > di_[1])
                                        ia[accumulate + i] -= di_[2];
                                }
                                std::copy_n(C + std::distance(J, *it) * bs_ * bs_, size * bs_ * bs_, a + accumulate * bs_ * bs_);
                                accumulate += size;
                            }
                            ia -= nrow + 1;
                            std::copy_n(R, nrow + 1, ia);
                        }
                        delete [] R;
                    }
                    else {
                        MPI_Comm_free(&communicator_);
                        communicator_ = DMatrix::communicator_;
                        accumulate = std::accumulate(I + 1, I + 1 + nrow, 0);
                    }
                    int* ja;
                    if(!ia) {
                        ia = new int[nrow + 1 + (mu_ < off_ ? accumulate : 0)];
                        std::partial_sum(I, I + nrow + 1, ia);
                        if(mu_ < off_) {
                            ja = ia + nrow + 1;
                            for(unsigned int i = 0; i < nrow; ++i) {
                                for(unsigned int j = ia[i]; j < ia[i + 1]; ++j) {
                                    const unsigned int idx = J[j - (super::numbering_ == 'F')];
                                    ja[j - (super::numbering_ == 'F')] = idx - di_[0];
                                    if(T == 1 && idx > di_[1])
                                        ja[j - (super::numbering_ == 'F')] -= di_[2];
                                }
                            }
                        }
                        else
                            ja = I + nrow + 1;
                        a = C;
                    }
                    const std::string prefix = OptionsPrefix<K>::prefix();
#ifdef DMKL_PARDISO
                    if(factorize && (a != C || Option::get()->set(prefix + "schwarz_method")))
                        range_[2] = 1;
#endif
                    if(mu_ < off_) {
                        loc2glob[0] -= di_[0];
                        loc2glob[1] -= di_[0];
                        delete [] di_;
                    }
                    if(factorize && !Option::get()->set(prefix + "schwarz_method")) {
#ifdef DMKL_PARDISO
                        Solver<K>::template numfact<S>(bs_, ia, loc2glob, ja, a);
                        loc2glob = nullptr;
#else
                        int* irn;
                        int* jcn;
                        K* b;
                        Wrapper<K>::template bsrcoo<S, Solver<K>::numbering_, 'F'>(nrow, bs_, a, ia, ja, b, irn, jcn, loc2glob[0] - (Solver<K>::numbering_ == 'F'));
                        if(a != b && a != C)
                            delete [] a;
                        if(DMatrix::n_)
                            Solver<K>::template numfact<S>(std::distance(irn, jcn), irn, jcn, a = b);
                        Solver<K>::range_ = { (loc2glob[0] - (Solver<K>::numbering_ == 'F')) * bs_, (loc2glob[1] + (Solver<K>::numbering_ == 'C')) * bs_ };
#endif
                    }
#ifdef DMKL_PARDISO
                    else
#endif
                        delete [] ia;
                    if(a != C)
                        delete [] a;
                }
                di_ = new int[nrow + 1];
                di_[0] = (super::numbering_ == 'F');
                std::map<int, unsigned short> off;
                std::set<int> on;
                std::map<unsigned short, unsigned int> allocation;
                bool r = false;
                std::vector<int*> range;
                range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
                range.emplace_back(J);
                for(unsigned int i = 0; i < nrow; ++i) {
                    di_[i + 1] = di_[i];
                    for(unsigned int j = 0; j < I[i + 1]; ++j) {
                        const int k = I[i] + di_[i] + j - (super::numbering_ == 'F' ? 2 : 0);
                        if(neighbors[k] == DMatrix::rank_) {
                            di_[i + 1]++;
                            on.insert(J[k]);
                            if(r) {
                                r = false;
                                range.emplace_back(J + k);
                            }
                        }
                        else {
                            off[J[k]] = neighbors[k];
                            allocation[neighbors[k]] += 1;
                            if(!r) {
                                r = true;
                                range.emplace_back(J + k);
                            }
                        }
                    }
                    I[i + 1] += I[i] - (di_[i + 1] - di_[i]);
                }
                delete [] neighbors;
                dof_ = on.size();
                if(opt.val<char>("krylov_method", HPDDM_KRYLOV_METHOD_GMRES) != HPDDM_KRYLOV_METHOD_NONE) {
                    accumulate = 0;
                    if(range.size() > 1) {
                        range.emplace_back(J + I[nrow] + di_[nrow] - (super::numbering_ == 'F' ? 2 : 0));
                        K* D = new K[(di_[nrow] - (super::numbering_ == 'F') - (range[1] - range[0])) * bs_ * bs_];
                        int* L = new int[di_[nrow] - (super::numbering_ == 'F') - (range[1] - range[0])];
                        std::vector<int*>::const_iterator it;
                        for(it = range.cbegin() + 2; it < range.cend() - 1; it += 2) {
                            unsigned int size = *(it + 1) - *it;
                            std::copy_n(*it, size, L + accumulate);
                            std::copy_n(C + std::distance(J, *it) * bs_ * bs_, size * bs_ * bs_, D + accumulate * bs_ * bs_);
                            accumulate += size;
                        }
                        accumulate = std::distance(J, range.back());
                        if(it != range.cend())
                            accumulate -= std::distance(*(it - 1), *it);
                        while(it > range.cbegin() + 3) {
                            it -= 2;
                            std::copy_backward(*(it - 1), *it, J + accumulate);
                            std::copy_backward(C + std::distance(J, *(it - 1)) * bs_ * bs_, C + std::distance(J, *it) * bs_ * bs_, C + (accumulate + 0) * bs_ * bs_);
                            accumulate -= *it - *(it - 1);
                        }
                        std::copy_n(D, (di_[nrow] - (super::numbering_ == 'F') - (range[1] - range[0])) * bs_ * bs_, C + (range[1] - range[0]) * bs_ * bs_);
                        std::copy_n(L, di_[nrow] - (super::numbering_ == 'F') - (range[1] - range[0]), J + (range[1] - range[0]));
                        delete [] L;
                        delete [] D;
                    }
                    recv_.reserve(allocation.size());
                    for(const std::pair<unsigned short, unsigned int>& p : allocation) {
                        recv_.emplace_back(p.first, std::vector<int>());
                        recv_.back().second.reserve(p.second);
                    }
                    std::unordered_map<int, int> g2l;
                    g2l.reserve(dof_ + off.size());
                    accumulate = 0;
                    for(const int& i : on)
                        g2l.emplace(i - (super::numbering_ == 'F'), accumulate++);
                    std::set<int>().swap(on);
                    unsigned short search[2] { 0, std::numeric_limits<unsigned short>::max() };
                    for(std::pair<const int, unsigned short>& i : off) {
                        if(search[1] != i.second) {
                            search[0] = std::distance(allocation.begin(), allocation.find(i.second));
                            search[1] = i.second;
                        }
                        recv_[search[0]].second.emplace_back(accumulate++);
                    }
                    if(S == 'S') {
                        char* table = new char[((T == 1 ? off_ * off_ : (off_ * (off_ - 1)) / 2) >> 3) + 1]();
                        std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator begin = (T == 1 ? recv_.cbegin() : std::upper_bound(recv_.cbegin(), recv_.cend(), std::make_pair(static_cast<unsigned short>(DMatrix::rank_), std::vector<int>()), [](const std::pair<unsigned short, std::vector<int>>& lhs, const std::pair<unsigned short, std::vector<int>>& rhs) { return lhs.first < rhs.first; }));
                        for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != recv_.cend(); ++it) {
                            const unsigned int idx = (T == 1 ? it->first * off_ : (it->first * (it->first - 1)) / 2) + DMatrix::rank_;
                            table[idx >> 3] |= 1 << (idx & 7);
                        }
                        MPI_Allreduce(MPI_IN_PLACE, table, ((T == 1 ? off_ * off_ : (off_ * (off_ - 1)) / 2) >> 3) + 1, MPI_CHAR, MPI_BOR, communicator_);
                        std::vector<unsigned short> infoRecv;
                        infoRecv.reserve(T == 1 ? off_ : DMatrix::rank_);
                        for(unsigned short i = 0; i < (T == 1 ? off_ : DMatrix::rank_); ++i) {
                            const unsigned int idx = (T == 1 ? DMatrix::rank_ * off_ : (DMatrix::rank_ * (DMatrix::rank_ - 1)) / 2) + i;
                            if(table[idx >> 3] & (1 << (idx & 7)))
                                infoRecv.emplace_back(i);
                        }
                        delete [] table;
                        const unsigned short size = infoRecv.size() + std::distance(begin, recv_.cend());
                        unsigned int* lengths = new unsigned int[size];
                        unsigned short distance = 0;
                        MPI_Request* rq = new MPI_Request[size];
                        for(const unsigned short& i : infoRecv) {
                            MPI_Irecv(lengths + distance, 1, MPI_UNSIGNED, i, 11, communicator_, rq + distance);
                            ++distance;
                        }
                        for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != recv_.cend(); ++it) {
                            lengths[distance] = it->second.size();
                            MPI_Isend(lengths + distance, 1, MPI_UNSIGNED, it->first, 11, communicator_, rq + distance);
                            ++distance;
                        }
                        MPI_Waitall(size, rq, MPI_STATUSES_IGNORE);
                        distance = 0;
                        for(const unsigned short& i : infoRecv) {
                            std::map<unsigned short, std::vector<int>>::iterator it = send_.emplace_hint(send_.end(), i, std::vector<int>(lengths[distance]));
                            MPI_Irecv(it->second.data(), it->second.size(), MPI_INT, i, 12, communicator_, rq + distance++);
                        }
                        accumulate = std::accumulate(lengths + infoRecv.size(), lengths + size, 0);
                        delete [] lengths;
                        int* sendIdx = new int[accumulate];
                        accumulate = 0;
                        for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != recv_.cend(); ++it) {
                            std::map<int, unsigned short>::const_iterator global = off.begin();
                            for(unsigned int k = 0; k < it->second.size(); ++k) {
                                std::advance(global, it->second[k] - (k == 0 ? dof_ : it->second[k - 1]));
                                sendIdx[accumulate + k] = global->first - (super::numbering_ == 'F');
                            }
                            MPI_Isend(sendIdx + accumulate, it->second.size(), MPI_INT, it->first, 12, communicator_, rq + distance++);
                            accumulate += it->second.size();
                        }
                        for(unsigned int i = 0; i < infoRecv.size(); ++i) {
                            int index;
                            MPI_Waitany(infoRecv.size(), rq, &index, MPI_STATUS_IGNORE);
                            std::for_each(send_[infoRecv[index]].begin(), send_[infoRecv[index]].end(), [&g2l](int& j) { j = g2l.at(j); });
                        }
                        MPI_Waitall(size - infoRecv.size(), rq + infoRecv.size(), MPI_STATUSES_IGNORE);
                        delete [] sendIdx;
                        delete [] rq;
                    }
                    else
                        for(std::pair<const unsigned short, std::vector<int>>& i : send_)
                            std::for_each(i.second.begin(), i.second.end(), [&g2l](int& j) { j = g2l.at(j); });
                    accumulate = 0;
                    for(std::pair<const int, unsigned short>& i : off)
                        g2l.emplace(i.first - (super::numbering_ == 'F'), accumulate++);
                    for(std::pair<unsigned short, std::vector<int>>& i : recv_)
                        std::for_each(i.second.begin(), i.second.end(), [&](int& j) { j -= dof_; });
                    ogj_ = new int[I[nrow] - (super::numbering_ == 'F' ? 1 : 0)];
                    std::copy_n(J + di_[nrow] - (super::numbering_ == 'F' ? 1 : 0), I[nrow] - (super::numbering_ == 'F' ? 1 : 0), ogj_);
                    std::for_each(J, J + I[nrow] + di_[nrow] - (super::numbering_ == 'F' ? 2 : 0), [&](int& i) { i = g2l[i - (this->numbering_ == 'F')] + (this->numbering_ == 'F'); });
                    buff_ = new K*[send_.size() + recv_.size()];
                    accumulate = std::accumulate(recv_.cbegin(), recv_.cend(), 0, [](unsigned int init, const std::pair<unsigned short, std::vector<int>>& i) { return init + i.second.size(); });
                    accumulate = std::accumulate(send_.cbegin(), send_.cend(), accumulate, [](unsigned int init, const std::pair<unsigned short, std::vector<int>>& i) { return init + i.second.size(); });
                    *buff_ = new K[accumulate * bs_];
                    accumulate = 0;
                    off_ = 0;
                    for(const std::pair<unsigned short, std::vector<int>>& i : recv_) {
                        buff_[off_++] = *buff_ + accumulate * bs_;
                        accumulate += i.second.size();
                    }
                    for(const std::pair<unsigned short, std::vector<int>>& i : send_) {
                        buff_[off_++] = *buff_ + accumulate * bs_;
                        accumulate += i.second.size();
                    }
                    rq_ = new MPI_Request[send_.size() + recv_.size()];
                    oi_ = I;
                    oa_ = C + (di_[nrow] - (super::numbering_ == 'F')) * bs_ * bs_;
                    oj_ = J + di_[nrow] - (super::numbering_ == 'F');
                    off_ = off.size();
                    if(DMatrix::rank_ != 0)
                        opt.remove("verbosity");
                }
                else {
                    delete [] di_;
                    di_ = nullptr;
                    off_ = 0;
                }
            }
            else {
                dof_ = nrow;
                off_ = 0;
                std::partial_sum(I, I + dof_ + 1, I);
                di_ = I;
                delete [] neighbors;
                if(factorize) {
#ifdef DMKL_PARDISO
                    Solver<K>::template numfact<S>(bs_, I, loc2glob, J, C);
                    loc2glob = nullptr;
#else
                    int* irn;
                    int* jcn;
                    K* b;
                    Wrapper<K>::template bsrcoo<S, Solver<K>::numbering_, 'F'>(nrow, bs_, C, I, J, b, irn, jcn, loc2glob[0] - (Solver<K>::numbering_ == 'F'));
                    if(DMatrix::n_)
                        Solver<K>::template numfact<S>(std::distance(irn, jcn), irn, jcn, b);
                    if(b != C)
                        delete [] b;
                    Solver<K>::range_ = { (loc2glob[0] - (Solver<K>::numbering_ == 'F')) * bs_, (loc2glob[1] + (Solver<K>::numbering_ == 'C')) * bs_ };
#endif
                }
            }
            OptionsPrefix<K>::setPrefix(opt.getPrefix());
            mu_ = 0;
#else
            constexpr char T = 0;
            if(S == 'S') {
                range_ = new unsigned int[2];
                std::copy_n(loc2glob, 2, range_);
            }
            {
                int* ia = nullptr;
                K* a;
                bool r = false;
                std::vector<integer_type*> range;
                range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
                range.emplace_back(J);
                int* R = new int[nrow + 1];
                R[0] = (super::numbering_ == 'F');
                for(unsigned int i = 0; i < nrow; ++i) {
                    R[i + 1] = R[i];
                    for(unsigned int j = 0; j < I[i + 1] - I[i]; ++j) {
                        const int k = I[i] + j - (super::numbering_ == 'F');
                        if(DMatrix::rank_ <= neighbors[k] && neighbors[k] <= DMatrix::rank_ + 1) {
                            R[i + 1]++;
                            if(r) {
                                r = false;
                                range.emplace_back(J + k);
                            }
                        }
                        else if(!r) {
                            r = true;
                            range.emplace_back(J + k);
                        }
                    }
                }
                if(range.size() > 1) {
                    range.emplace_back(J + I[nrow] - (super::numbering_ == 'F'));
                    std::vector<integer_type*>::const_iterator it;
                    a = new K[(R[nrow] - (super::numbering_ == 'F')) * bs_ * bs_];
                    ia = new int[nrow + 1 + R[nrow] - (super::numbering_ == 'F')];
                    unsigned int accumulate = 0;
                    ia += nrow + 1;
                    for(it = range.cbegin(); it < range.cend() - 1; it += 2) {
                        unsigned int size = *(it + 1) - *it;
                        for(unsigned int i = 0; i < size; ++i) {
                            ia[accumulate + i] = (*it)[i] - di_[0];
                            if(T == 1 && (*it)[i] > di_[1])
                                ia[accumulate + i] -= di_[2];
                        }
                        std::copy_n(C + std::distance(J, *it) * bs_ * bs_, size * bs_ * bs_, a + accumulate * bs_ * bs_);
                        accumulate += size;
                    }
                    ia -= nrow + 1;
                    std::copy_n(R, nrow + 1, ia);
                }
                delete [] R;
                int* ja;
                if(!ia) {
                    ia = new int[nrow + 1 + I[nrow] - (super::numbering_ == 'F')];
                    std::copy_n(I, nrow + 1, ia);
                    ja = ia + nrow + 1;
                    for(unsigned int i = 0; i < nrow; ++i) {
                        for(unsigned int j = ia[i]; j < ia[i + 1]; ++j) {
                            const unsigned int idx = J[j - (super::numbering_ == 'F')];
                            ja[j - (super::numbering_ == 'F')] = idx - di_[0];
                            if(T == 1 && idx > di_[1])
                                ja[j - (super::numbering_ == 'F')] -= di_[2];
                        }
                    }
                    a = C;
                }
                loc2glob[0] -= di_[0];
                loc2glob[1] -= di_[0];
                delete [] di_;
                delete [] ia;
                if(a != C)
                    delete [] a;
            }
            di_ = new integer_type[nrow + 1];
            di_[0] = (super::numbering_ == 'F');
            std::map<int, unsigned short> off;
            std::set<int> on;
            std::map<unsigned short, unsigned int> allocation;
            std::unordered_map<unsigned short, std::set<int>> exchange;
            bool r = false;
            std::vector<integer_type*> range;
            range.reserve((S == 'S' ? 1 : (T == 1 ? 8 : 4)) * nrow);
            range.emplace_back(J);
            for(unsigned int i = 0; i < nrow; ++i) {
                di_[i + 1] = di_[i];
                for(unsigned int j = 0; j < I[i + 1] - I[i] - di_[i]; ++j) {
                    const int k = I[i] + di_[i] + j - (super::numbering_ == 'F' ? 2 : 0);
                    if(neighbors[k] == DMatrix::rank_) {
                        di_[i + 1]++;
                        on.insert(J[k]);
                        if(r) {
                            r = false;
                            range.emplace_back(J + k);
                        }
                    }
                    else {
                        off[J[k]] = neighbors[k];
                        allocation[neighbors[k]] += 1;
                        if(S == 'S' && factorize && neighbors[k] > DMatrix::rank_)
                            exchange[neighbors[k]].insert(range_[0] + i);
                        if(!r) {
                            r = true;
                            range.emplace_back(J + k);
                        }
                    }
                }
                I[i + 1] -= di_[i + 1];
            }
            delete [] neighbors;
            dof_ = on.size();
            unsigned int accumulate = 0;
            if(range.size() > 1) {
                range.emplace_back(J + I[nrow] + di_[nrow] - (super::numbering_ == 'F' ? 2 : 0));
                K* D = new K[(di_[nrow] - (super::numbering_ == 'F') - (range[1] - range[0])) * bs_ * bs_];
                int* L = new int[di_[nrow] - (super::numbering_ == 'F') - (range[1] - range[0])];
                std::vector<integer_type*>::const_iterator it;
                for(it = range.cbegin() + 2; it < range.cend() - 1; it += 2) {
                    unsigned int size = *(it + 1) - *it;
                    std::copy_n(*it, size, L + accumulate);
                    std::copy_n(C + std::distance(J, *it) * bs_ * bs_, size * bs_ * bs_, D + accumulate * bs_ * bs_);
                    accumulate += size;
                }
                accumulate = std::distance(J, range.back());
                if(it != range.cend())
                    accumulate -= std::distance(*(it - 1), *it);
                while(it > range.cbegin() + 3) {
                    it -= 2;
                    std::copy_backward(*(it - 1), *it, J + accumulate);
                    std::copy_backward(C + std::distance(J, *(it - 1)) * bs_ * bs_, C + std::distance(J, *it) * bs_ * bs_, C + (accumulate + 0) * bs_ * bs_);
                    accumulate -= *it - *(it - 1);
                }
                std::copy_n(D, (di_[nrow] - (super::numbering_ == 'F') - (range[1] - range[0])) * bs_ * bs_, C + (range[1] - range[0]) * bs_ * bs_);
                std::copy_n(L, di_[nrow] - (super::numbering_ == 'F') - (range[1] - range[0]), J + (range[1] - range[0]));
                delete [] L;
                delete [] D;
            }
            recv_.reserve(allocation.size());
            for(const std::pair<const unsigned short, unsigned int>& p : allocation) {
                recv_.emplace_back(p.first, std::vector<int>());
                recv_.back().second.reserve(p.second);
            }
            std::unordered_map<int, int> g2l;
            g2l.reserve(dof_ + off.size());
            accumulate = 0;
            for(const int& i : on)
                g2l.emplace(i - (super::numbering_ == 'F'), accumulate++);
            unsigned short search[2] { 0, std::numeric_limits<unsigned short>::max() };
            for(std::pair<const int, unsigned short>& i : off) {
                if(search[1] != i.second) {
                    search[0] = std::distance(allocation.begin(), allocation.find(i.second));
                    search[1] = i.second;
                }
                recv_[search[0]].second.emplace_back(accumulate++);
            }
            std::set<int> overlap;
            if(S == 'S') {
                MPI_Comm_size(DMatrix::communicator_, &off_);
                char* table = new char[((T == 1 ? off_ * off_ : (off_ * (off_ - 1)) / 2) >> 3) + 1]();
                std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator begin = (T == 1 ? recv_.cbegin() : std::upper_bound(recv_.cbegin(), recv_.cend(), std::make_pair(static_cast<unsigned short>(DMatrix::rank_), std::vector<int>()), [](const std::pair<unsigned short, std::vector<int>>& lhs, const std::pair<unsigned short, std::vector<int>>& rhs) { return lhs.first < rhs.first; }));
                for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != recv_.cend(); ++it) {
                    const unsigned int idx = (T == 1 ? it->first * off_ : (it->first * (it->first - 1)) / 2) + DMatrix::rank_;
                    table[idx >> 3] |= 1 << (idx & 7);
                }
                MPI_Allreduce(MPI_IN_PLACE, table, ((T == 1 ? off_ * off_ : (off_ * (off_ - 1)) / 2) >> 3) + 1, MPI_CHAR, MPI_BOR, DMatrix::communicator_);
                std::vector<unsigned short> infoRecv;
                infoRecv.reserve(T == 1 ? off_ : DMatrix::rank_);
                for(unsigned short i = 0; i < (T == 1 ? off_ : DMatrix::rank_); ++i) {
                    const unsigned int idx = (T == 1 ? DMatrix::rank_ * off_ : (DMatrix::rank_ * (DMatrix::rank_ - 1)) / 2) + i;
                    if(table[idx >> 3] & (1 << (idx & 7)))
                        infoRecv.emplace_back(i);
                }
                delete [] table;
                const unsigned short size = infoRecv.size() + std::distance(begin, recv_.cend());
                unsigned int* lengths = new unsigned int[size + 1];
                unsigned short distance = 0;
                MPI_Request* rq = new MPI_Request[size];
                for(const unsigned short& i : infoRecv) {
                    MPI_Irecv(lengths + distance, 1, MPI_UNSIGNED, i, 11, DMatrix::communicator_, rq + distance);
                    ++distance;
                }
                for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != recv_.cend(); ++it, ++distance) {
                    lengths[distance] = it->second.size();
                    MPI_Isend(lengths + distance, 1, MPI_UNSIGNED, it->first, 11, DMatrix::communicator_, rq + distance);
                }
                MPI_Waitall(size, rq, MPI_STATUSES_IGNORE);
                distance = 0;
                for(const unsigned short& i : infoRecv) {
                    std::map<unsigned short, std::vector<int>>::iterator it = send_.emplace_hint(send_.end(), i, std::vector<int>(lengths[distance]));
                    MPI_Irecv(it->second.data(), it->second.size(), MPI_INT, i, 12, DMatrix::communicator_, rq + distance++);
                }
                accumulate = std::accumulate(lengths + infoRecv.size(), lengths + size, 0);
                int* sendIdx = new int[accumulate];
                accumulate = 0;
                for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != recv_.cend(); ++it) {
                    std::map<int, unsigned short>::const_iterator global = off.begin();
                    for(unsigned int k = 0; k < it->second.size(); ++k) {
                        std::advance(global, it->second[k] - (k == 0 ? dof_ : it->second[k - 1]));
                        sendIdx[accumulate + k] = global->first - (super::numbering_ == 'F');
                    }
                    MPI_Isend(sendIdx + accumulate, it->second.size(), MPI_INT, it->first, 12, DMatrix::communicator_, rq + distance++);
                    accumulate += it->second.size();
                }
                for(unsigned int i = 0; i < infoRecv.size(); ++i) {
                    int index;
                    MPI_Waitany(infoRecv.size(), rq, &index, MPI_STATUS_IGNORE);
                    std::for_each(send_[infoRecv[index]].begin(), send_[infoRecv[index]].end(), [&g2l, &on, &overlap](int& j) {
                        if(factorize && on.find(j) == on.cend())
                            overlap.insert(j);
                        j = g2l.at(j);
                    });
                }
                MPI_Waitall(size - infoRecv.size(), rq + infoRecv.size(), MPI_STATUSES_IGNORE);
                if(factorize) {
                    distance = 0;
                    for(const unsigned short& i : infoRecv) {
                        MPI_Irecv(lengths + distance + 1, 1, MPI_UNSIGNED, i, 121, DMatrix::communicator_, rq + distance);
                        ++distance;
                    }
                    for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != recv_.cend(); ++it, ++distance) {
                        lengths[distance + 1] = exchange[it->first].size();
                        MPI_Isend(lengths + distance + 1, 1, MPI_UNSIGNED, it->first, 121, DMatrix::communicator_, rq + distance);
                    }
                    MPI_Waitall(size, rq, MPI_STATUSES_IGNORE);
                    delete [] sendIdx;
                    lengths[0] = 0;
                    std::partial_sum(lengths + 1, lengths + size + 1, lengths + 1);
                    sendIdx = new int[lengths[size]];
                    distance = 0;
                    for(const unsigned short& i : infoRecv) {
                        MPI_Irecv(sendIdx + lengths[distance], lengths[distance + 1] - lengths[distance], MPI_INT, i, 122, DMatrix::communicator_, rq + distance);
                        ++distance;
                    }
                    for(std::vector<std::pair<unsigned short, std::vector<int>>>::const_iterator it = begin; it != recv_.cend(); ++it, ++distance) {
                        std::copy(exchange[it->first].begin(), exchange[it->first].end(), sendIdx + lengths[distance]);
                        MPI_Isend(sendIdx + lengths[distance], exchange[it->first].size(), MPI_INT, it->first, 122, DMatrix::communicator_, rq + distance);
                    }
                    for(unsigned int i = 0; i < infoRecv.size(); ++i) {
                        int index;
                        MPI_Waitany(infoRecv.size(), rq, &index, MPI_STATUS_IGNORE);
                        std::for_each(sendIdx + lengths[index], sendIdx + lengths[index + 1], [&overlap](int& j) { overlap.insert(j); });
                    }
                    MPI_Waitall(size - infoRecv.size(), rq + infoRecv.size(), MPI_STATUSES_IGNORE);
                }
                delete [] lengths;
                delete [] sendIdx;
                delete [] rq;
            }
            else
                for(std::pair<const unsigned short, std::vector<int>>& i : send_)
                    std::for_each(i.second.begin(), i.second.end(), [&g2l](int& j) { j = g2l.at(j); });
            accumulate = 0;
            for(std::pair<const int, unsigned short>& i : off)
                g2l.emplace(i.first - (super::numbering_ == 'F'), accumulate++);
            for(std::pair<unsigned short, std::vector<int>>& i : recv_)
                std::for_each(i.second.begin(), i.second.end(), [&](int& j) { j -= dof_; });
            ogj_ = new integer_type[I[nrow] - (super::numbering_ == 'F' ? 1 : 0)];
            std::copy_n(J + di_[nrow] - (super::numbering_ == 'F' ? 1 : 0), I[nrow] - (super::numbering_ == 'F' ? 1 : 0), ogj_);
            std::for_each(J, J + I[nrow] + di_[nrow] - (super::numbering_ == 'F' ? 2 : 0), [&](integer_type& i) { i = g2l[i - (this->numbering_ == 'F')] + (this->numbering_ == 'F'); });
            oi_ = I;
            oa_ = C + (di_[nrow] - (super::numbering_ == 'F')) * bs_ * bs_;
            oj_ = J + di_[nrow] - (super::numbering_ == 'F');
            off_ = off.size();
            if(factorize && !idx_) {
                PetscCallVoid(PetscMalloc1(on.size() + overlap.size() + off.size(), &idx_));
                for(const auto& range : { on, overlap })
                    for(const int& i : range)
                        *idx_++ = i - (super::numbering_ == 'F');
                for(std::pair<const int, unsigned short>& i : off)
                    *idx_++ = i.first - (super::numbering_ == 'F');
                idx_ -= on.size() + overlap.size() + off.size();
            }
#endif
            delete [] loc2glob;
        }
    public:
        InexactCoarseOperator() : OptionsPrefix<K>(), super(), s_(),
#if !HPDDM_PETSC
                                                                    p_(), communicator_(MPI_COMM_NULL), buff_(), rq_(), x_(), mu_()
#else
                                                                    idx_(), da_()
#endif
                                                                          , di_(), oi_(), ogj_(), range_(), off_() { }
        ~InexactCoarseOperator() {
#if !HPDDM_PETSC
            if(s_)
                delete [] s_->getScaling();
            delete s_;
            if(buff_) {
                delete [] *buff_;
                delete [] buff_;
            }
            delete [] x_;
            if(communicator_ != MPI_COMM_NULL) {
                int size;
                MPI_Comm_size(communicator_, &size);
#ifdef DMKL_PARDISO
                if(size > 1)
#endif
                    delete [] di_;
#ifdef DMKL_PARDISO
                if(range_ && range_[2] == 1)
#endif
                    delete [] da_;
                if(communicator_ != DMatrix::communicator_)
                    MPI_Comm_free(&communicator_);
            }
            delete [] rq_;
#else
            PetscCallVoid(PetscFree(idx_));
            delete [] di_;
            delete [] da_;
#endif
            delete [] ogj_;
            delete [] range_;
            delete [] oi_;
            di_ = ogj_ = oi_ = nullptr;
            range_ = nullptr;
            da_ = nullptr;
            std::map<unsigned short, std::vector<int>>().swap(send_);
            vectorNeighbor().swap(recv_);
        }
        constexpr int getDof() const { return dof_ * bs_; }
        return_type solve(K* rhs, const unsigned short& mu) {
#if !HPDDM_PETSC
            if(s_) {
                K* in, *out;
                const unsigned int n = s_->getDof();
                in = new K[mu * n];
                out = new K[mu * n]();
                for(unsigned short i = 0; i < mu; ++i) {
                    std::copy_n(rhs + i * dof_ * bs_, dof_ * bs_, in + i * n);
                    std::fill(in + i * n + dof_ * bs_, in + (i + 1) * n, K());
                }
                bool allocate = s_->setBuffer();
                s_->exchange(in, mu);
                s_->clearBuffer(allocate);
                IterativeMethod::solve(*s_, in, out, mu, communicator_);
                for(unsigned short i = 0; i < mu; ++i)
                    std::copy_n(out + i * n, dof_ * bs_, rhs + i * n);
                delete [] out;
                delete [] in;
            }
            else {
                if(mu_ != mu) {
                    delete [] x_;
                    x_ = new K[mu * dof_ * bs_]();
                    mu_ = mu;
                }
                const std::string prefix = OptionsPrefix<K>::prefix();
                if(Option::get()->val<char>(prefix + "krylov_method", HPDDM_KRYLOV_METHOD_GMRES) != HPDDM_KRYLOV_METHOD_NONE)
                    IterativeMethod::solve(*this, rhs, x_, mu, communicator_);
                else
                    Solver<K>::solve(rhs, x_, mu);
                std::copy_n(x_, mu * dof_ * bs_, rhs);
            }
#else
            PetscFunctionBeginUser;
            PetscCall(PCHPDDMSolve_Private(s_, rhs, mu));
            PetscFunctionReturn(0);
#endif
        }
        decltype(s_) getSubdomain() const {
            return s_;
        }
#if !HPDDM_PETSC
        void setParent(decltype(p_) const p) {
            p_ = p;
        }
        int GMV(const K* const in, K* const out, const int& mu = 1) const {
            exchange<'N'>(in, nullptr, mu);
            Wrapper<K>::template bsrmm<Solver<K>::numbering_>(S == 'S', &dof_, &mu, &bs_, da_, di_, dj_, in, out);
            wait<'N'>(o_ + (mu - 1) * off_ * bs_);
            Wrapper<K>::template bsrmm<Solver<K>::numbering_>("N", &dof_, &mu, &off_, &bs_, &(Wrapper<K>::d__1), false, oa_, oi_, oj_, o_, &(Wrapper<K>::d__1), out);
            if(S == 'S') {
                Wrapper<K>::template bsrmm<Solver<K>::numbering_>(&(Wrapper<K>::transc), &dof_, &mu, &off_, &bs_, &(Wrapper<K>::d__1), false, oa_, oi_, oj_, in, &(Wrapper<K>::d__0), o_);
                exchange<'T'>(nullptr, out, mu);
                wait<'T'>(out + (mu - 1) * dof_ * bs_);
            }
            return 0;
        }
        template<bool>
        int apply(const K* const in, K* const out, const unsigned short& mu = 1, K* = nullptr) const {
#ifdef DMUMPS
            if(DMatrix::n_)
#endif
                Solver<K>::solve(in, out, mu);
            return 0;
        }
        template<bool = false>
        bool start(const K* const, K* const, const unsigned short& mu = 1) const {
            if(off_) {
                unsigned short k = 1;
                const std::string prefix = OptionsPrefix<K>::prefix();
                const Option& opt = *Option::get();
                if(opt.any_of(prefix + "krylov_method", { HPDDM_KRYLOV_METHOD_GCRODR, HPDDM_KRYLOV_METHOD_BGCRODR }) && !opt.val<unsigned short>(prefix + "recycle_same_system"))
                    k = std::max(opt.val<int>(prefix + "recycle", 1), 1);
                K** ptr = const_cast<K**>(&o_);
                *ptr = new K[mu * k * off_ * bs_]();
                return true;
            }
            else
                return false;
        }
        void end(const bool free) const {
            if(free)
                delete [] o_;
        }
#endif
        static constexpr underlying_type<K>* getScaling() { return nullptr; }
        static constexpr std::unordered_map<unsigned int, K> boundaryConditions() { return std::unordered_map<unsigned int, K>(); }
        typename std::conditional<HPDDM_PETSC, Mat, MatrixCSR<K>*>::type buildMatrix(const InexactCoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>* in, const std::vector<unsigned int>& displs, const unsigned int* const ranges, const std::vector<std::vector<unsigned int>>& off, const std::vector<std::vector<std::pair<unsigned short, unsigned short>>>& reduction = std::vector<std::vector<std::pair<unsigned short, unsigned short>>>(), const std::map<std::pair<unsigned short, unsigned short>, unsigned short>& sizes = std::map<std::pair<unsigned short, unsigned short>, unsigned short>(), const std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>& extra = std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>(), const std::tuple<int*, K*, MPI_Request*>& transfer = std::tuple<int*, K*, MPI_Request*>()) const {
#if HPDDM_PETSC
            char S;
            {
                Mat A;
                PetscCallContinue(KSPGetOperators(s_->ksp, &A, nullptr));
                PetscBool symmetric;
                PetscCallContinue(PetscObjectTypeCompare((PetscObject)A, MATMPISBAIJ, &symmetric));
                S = (symmetric ? 'S' : 'G');
            }
#endif
            int rank, size;
            MPI_Comm_size(in->communicator_, &size);
            MPI_Comm_rank(in->communicator_, &rank);
            integer_type* di = new integer_type[in->dof_ + in->off_ + displs.back() + 1];
            const int bss = in->bs_ * in->bs_;
            di[0] = 0;
            for(unsigned int i = 0; i < in->dof_; ++i)
                di[i + 1] = di[i] + in->oi_[i + 1] - in->oi_[i] + in->di_[i + 1] - in->di_[i];
            std::vector<std::vector<std::pair<int, K*>>> to;
            if(S != 'S') {
                to.resize(in->off_);
                for(unsigned int i = 0; i < in->dof_ + 1; ++i)
                    for(unsigned int j = in->oi_[i]; j < in->oi_[i + 1]; ++j)
                        to[in->oj_[j - (super::numbering_ == 'F')] - (super::numbering_ == 'F')].emplace_back(i, in->oa_ + (j - (super::numbering_ == 'F')) * bss);
                for(unsigned int i = 0; i < in->off_; ++i) {
                    std::sort(to[i].begin(), to[i].end(), [](const std::pair<int, K*>& lhs, const std::pair<int, K*>& rhs) { return lhs.first < rhs.first; });
                    di[in->dof_ + i + 1] = di[in->dof_ + i] + to[i].size();
                }
            }
            else {
                if(in != this) {
                    for(std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::const_iterator it = extra.cbegin(); it != extra.cend(); ++it) {
                        if(bss > 1)
                            std::for_each(di + std::get<1>(it->second) + 1, di + in->dof_ + 1, [&](integer_type& k) { k += std::get<2>(it->second).size(); });
                        else {
                            unsigned int row = 0;
                            for(const unsigned short& p : std::get<2>(it->second))
                                row += sizes.at(std::make_pair(p, p));
                            for(unsigned short i = 0; i < std::get<0>(it->second); ++i)
                                di[std::get<1>(it->second) + i + 1] += (i + 1) * row;
                            std::for_each(di + std::get<1>(it->second) + std::get<0>(it->second) + 1, di + in->dof_ + 1, [&](integer_type& k) { k += std::get<0>(it->second) * row; });
                        }
                    }
                }
                std::fill(di + in->dof_ + 1, di + in->dof_ + in->off_ + displs.back() + 1, di[in->dof_]);
            }
            std::vector<std::pair<std::map<std::pair<unsigned short, unsigned short>, unsigned short>::const_iterator, unsigned short>> rows;
            std::unordered_map<unsigned short, unsigned int> map;
            std::vector<std::pair<int*, K*>>* recv;
            std::vector<int> nnz;
            std::vector<std::pair<unsigned int, K*>>* oa = nullptr;
            if(in == this) {
                recv = new std::vector<std::pair<int*, K*>>[S == 'S' ? 2 : 1]();
                MPI_Request* rqRecv = new MPI_Request[2 * (in->send_.size() + in->recv_.size())];
                MPI_Request* rqSend;
                std::vector<std::pair<int*, K*>> send;
                {
                    std::set<int> unique;
                    for(int i = 0; i < in->dof_; ++i) {
                        for(int j = in->oi_[i] - (super::numbering_ == 'F'); j < in->oi_[i + 1] - (super::numbering_ == 'F'); ++j)
                            unique.insert(in->ogj_[j] - (super::numbering_ == 'F'));
                    }
                    nnz.reserve(unique.size());
                    std::copy(unique.cbegin(), unique.cend(), std::back_inserter(nnz));
                }
                if(S == 'S') {
                    rqSend = rqRecv + 2 * in->send_.size();
                    recv[1].reserve(in->send_.size());
                    send.reserve(in->recv_.size());
                    std::map<unsigned short, std::vector<int>>::const_iterator it = in->send_.cbegin();
                    for(unsigned short i = 0; i < in->send_.size(); ++i) {
                        recv[1].emplace_back(std::make_pair(new int[2 * (displs[i + 1] - displs[i])], nullptr));
                        MPI_Irecv(recv[1].back().first, 2 * (displs[i + 1] - displs[i]), MPI_INT, it->first, 11, in->communicator_, rqRecv + i);
                        ++it;
                    }
                    for(unsigned short j = 0; j < in->recv_.size(); ++j) {
                        send.emplace_back(std::make_pair(new int[2 * off[j].size()], nullptr));
                        unsigned int nnz = 0;
                        for(unsigned int i = 0; i < off[j].size(); ++i) {
                            send.back().first[i] = in->oi_[off[j][i] + 1] - in->oi_[off[j][i]];
                            nnz += send.back().first[i];
                        }
                        std::copy(off[j].cbegin(), off[j].cend(), send.back().first + off[j].size());
                        MPI_Isend(send.back().first, 2 * off[j].size(), MPI_INT, in->recv_[j].first, 11, in->communicator_, rqSend + j);
                        send.back().second = new K[nnz + (nnz + (off[j].size() * (off[j].size() + 1)) / 2) * bss]();
                        int* ja = reinterpret_cast<int*>(send.back().second + nnz * bss);
                        nnz = 0;
                        for(unsigned int i = 0; i < off[j].size(); ++i) {
                            for(unsigned k = in->oi_[off[j][i]] - (super::numbering_ == 'F'); k < in->oi_[off[j][i] + 1] - (super::numbering_ == 'F'); ++k) {
                                std::copy_n(in->oa_ + k * bss, bss, send.back().second + nnz * bss);
                                ja[nnz++] = in->ogj_[k];
                            }
                        }
                        K* const diag = send.back().second + nnz * (1 + bss);
                        std::fill_n(diag, ((off[j].size() * (off[j].size() + 1)) / 2) * bss, K());
                        for(unsigned int i = 0; i < off[j].size(); ++i)
                            for(unsigned int k = i; k < off[j].size(); ++k) {
                                integer_type* const pt = std::lower_bound(in->dj_ + in->di_[off[j][i]] - (super::numbering_ == 'F'), in->dj_ + in->di_[off[j][i] + 1] - (super::numbering_ == 'F'), off[j][k] + (super::numbering_ == 'F'));
                                if(pt != in->dj_ + in->di_[off[j][i] + 1] - (super::numbering_ == 'F') && *pt == off[j][k] + (super::numbering_ == 'F'))
                                    std::copy_n(in->da_ + std::distance(in->dj_, pt) * bss, bss, diag + (i * off[j].size() + k - ((i * (i - 1)) / 2 + i)) * bss);
                            }
                        MPI_Isend(send.back().second, nnz + (nnz + (off[j].size() * (off[j].size() + 1)) / 2) * bss, Wrapper<K>::mpi_type(), in->recv_[j].first, 12, in->communicator_, rqSend + in->recv_.size() + j);
                    }
                    for(unsigned short i = 0; i < in->send_.size(); ++i) {
                        it = in->send_.cbegin();
                        int index;
                        MPI_Waitany(in->send_.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                        std::advance(it, index);
                        unsigned int nnz = std::accumulate(recv[1][index].first, recv[1][index].first + displs[index + 1] - displs[index], 0);
                        recv[1][index].second = new K[nnz + (nnz + ((displs[index + 1] - displs[index]) * (displs[index + 1] - displs[index] + 1)) / 2) * bss];
                        MPI_Irecv(recv[1][index].second, nnz + (nnz + ((displs[index + 1] - displs[index]) * (displs[index + 1] - displs[index] + 1)) / 2) * bss, Wrapper<K>::mpi_type(), it->first, 12, in->communicator_, rqRecv + in->send_.size() + index);
                    }
                    oa = new std::vector<std::pair<unsigned int, K*>>[in->dof_ + in->off_ + displs.back()]();
                    for(unsigned short i = 0; i < in->send_.size(); ++i) {
                        it = in->send_.cbegin();
                        int index;
                        MPI_Waitany(in->send_.size(), rqRecv + in->send_.size(), &index, MPI_STATUS_IGNORE);
                        std::advance(it, index);
                        unsigned int accumulate = std::accumulate(recv[1][index].first, recv[1][index].first + displs[index + 1] - displs[index], 0);
                        int* ja = reinterpret_cast<int*>(recv[1][index].second + accumulate * bss);
                        accumulate = 0;
                        for(unsigned int j = 0; j < displs[index + 1] - displs[index]; ++j) {
                            for(unsigned int k = 0; k < recv[1][index].first[j]; ++k) {
                                if(ja[accumulate] >= range_[0] && ja[accumulate] <= range_[1])
                                    oa[ja[accumulate] - range_[0]].emplace_back(in->dof_ + displs[index] + j, recv[1][index].second + accumulate * bss);
                                else {
                                    bool kept = false;
                                    for(unsigned int i = index + 1; i < in->send_.size() && !kept; ++i) {
                                        if(ja[accumulate] >= ranges[2 * i] && ja[accumulate] <= ranges[2 * i + 1]) {
                                            const int* pt = std::lower_bound(recv[1][i].first + displs[i + 1] - displs[i], recv[1][i].first + 2 * (displs[i + 1] - displs[i]), ja[accumulate] - ranges[2 * i]);
                                            if(pt != recv[1][i].first + 2 * (displs[i + 1] - displs[i]) && *pt == ja[accumulate] - ranges[2 * i])
                                                oa[in->dof_ + displs[index] + j].emplace_back(in->dof_ + displs[i] + std::distance(recv[1][i].first + displs[i + 1] - displs[i], std::lower_bound(recv[1][i].first + displs[i + 1] - displs[i], recv[1][i].first + 2 * (displs[i + 1] - displs[i]), ja[accumulate] - ranges[2 * i])), recv[1][index].second + accumulate * bss);
                                            kept = true;
                                        }
                                    }
                                    for(unsigned int i = in->send_.size(); i < in->send_.size() + in->recv_.size() && !kept; ++i) {
                                        if(ja[accumulate] >= ranges[2 * i] && ja[accumulate] <= ranges[2 * i + 1]) {
                                            std::vector<int>::const_iterator pt = std::lower_bound(nnz.cbegin(), nnz.cend(), ja[accumulate] - (super::numbering_ == 'F'));
                                            if(pt != nnz.cend() && *pt == ja[accumulate] - (super::numbering_ == 'F'))
                                                oa[in->dof_ + displs[index] + j].emplace_back(in->dof_ + displs.back() + std::distance(nnz.cbegin(), pt), recv[1][index].second + accumulate * bss);
                                            kept = true;
                                        }
                                    }
                                }
                                ++accumulate;
                            }
                        }
                        for(unsigned int j = 0; j < displs[index + 1] - displs[index]; ++j) {
                            for(unsigned int k = j; k < displs[index + 1] - displs[index]; ++k)
                                oa[in->dof_ + displs[index] + j].emplace_back(in->dof_ + displs[index] + k, recv[1][index].second + accumulate + (accumulate + j * (displs[index + 1] - displs[index]) - (j * (j - 1)) / 2 + k - j) * bss);
                        }
                    }
                    unsigned int accumulate = 0;
                    for(unsigned int i = 0; i < in->dof_ + in->off_ + displs.back(); ++i) {
                        accumulate += oa[i].size();
                        di[i + 1] += accumulate;
                        std::sort(oa[i].begin(), oa[i].end());
                    }
                    for(unsigned int i = 0; i < 2 * in->recv_.size(); ++i) {
                        int index;
                        MPI_Waitany(2 * in->recv_.size(), rqSend, &index, MPI_STATUS_IGNORE);
                        if(index < in->recv_.size())
                            delete [] send[index].first;
                        else
                            delete [] send[index - in->recv_.size()].second;
                    }
                    send.clear();
                }
                rqSend = rqRecv + 2 * in->recv_.size();
                recv[0].reserve(in->recv_.size());
                send.reserve(in->send_.size());
                for(unsigned short j = 0; j < in->recv_.size(); ++j) {
                    recv[0].emplace_back(std::make_pair(new int[2 * in->recv_[j].second.size()], nullptr));
                    MPI_Irecv(recv[0].back().first, 2 * in->recv_[j].second.size(), MPI_INT, in->recv_[j].first, 322, in->communicator_, rqRecv + j);
                }
                for(const std::pair<const unsigned short, std::vector<int>>& p : in->send_) {
                    send.emplace_back(std::make_pair(new int[2 * p.second.size()](), nullptr));
                    std::vector<int> col;
                    std::vector<K> val;
                    for(unsigned int i = 0; i < p.second.size(); ++i) {
                        for(unsigned int k = (S != 'S' ? 0 : i); k < p.second.size(); ++k) {
                            integer_type* const pt = std::lower_bound(in->dj_ + in->di_[p.second[i]] - (super::numbering_ == 'F'), in->dj_ + in->di_[p.second[i] + 1] - (super::numbering_ == 'F'), p.second[k] + (super::numbering_ == 'F'));
                            if(pt != in->dj_ + in->di_[p.second[i] + 1] - (super::numbering_ == 'F') && *pt == p.second[k] + (super::numbering_ == 'F')) {
                                send.back().first[2 * i]++;
                                col.emplace_back(k);
                                val.insert(val.end(), in->da_ + std::distance(in->dj_, pt) * bss, in->da_ + (std::distance(in->dj_, pt) + 1) * bss);
                            }
                        }
                        send.back().first[2 * i + 1] = in->oi_[p.second[i] + 1] - in->oi_[p.second[i]];
                        for(unsigned int k = in->oi_[p.second[i]] - (super::numbering_ == 'F'); k < in->oi_[p.second[i] + 1] - (super::numbering_ == 'F'); ++k) {
                            col.emplace_back(in->ogj_[k] - (super::numbering_ == 'F'));
                            val.insert(val.end(), in->oa_ + k * bss, in->oa_ + (k + 1) * bss);
                        }
                    }
                    MPI_Isend(send.back().first, 2 * p.second.size(), MPI_INT, p.first, 322, in->communicator_, rqSend);
                    unsigned int accumulate = 0;
                    for(unsigned int i = 0; i < p.second.size(); ++i)
                        accumulate += send.back().first[2 * i] + send.back().first[2 * i + 1];
                    send.back().second = new K[accumulate * (1 + bss)];
                    accumulate = 0;
                    for(unsigned int i = 0; i < p.second.size(); ++i) {
                        std::copy_n(col.cbegin() + accumulate, send.back().first[2 * i], send.back().second + accumulate * (1 + bss));
                        std::copy_n(val.cbegin() + accumulate * bss, send.back().first[2 * i] * bss, send.back().second + accumulate * (1 + bss) + send.back().first[2 * i]);
                        std::copy_n(col.cbegin() + accumulate + send.back().first[2 * i], send.back().first[2 * i + 1], send.back().second + (accumulate + send.back().first[2 * i]) * (1 + bss));
                        std::copy_n(val.cbegin() + (accumulate + send.back().first[2 * i]) * bss, send.back().first[2 * i + 1] * bss, send.back().second + (accumulate + send.back().first[2 * i]) * (1 + bss) + send.back().first[2 * i + 1]);
                        accumulate += send.back().first[2 * i] + send.back().first[2 * i + 1];
                    }
                    MPI_Isend(send.back().second, accumulate * (1 + bss), Wrapper<K>::mpi_type(), p.first, 323, in->communicator_, rqSend + in->send_.size());
                    ++rqSend;
                }
                rqSend -= in->send_.size();
                for(unsigned short j = 0; j < in->recv_.size(); ++j) {
                    int index;
                    MPI_Waitany(in->recv_.size(), rqRecv, &index, MPI_STATUS_IGNORE);
                    unsigned int nnz = std::accumulate(recv[0][index].first, recv[0][index].first + 2 * in->recv_[index].second.size(), 0);
                    recv[0][index].second = new K[nnz * (1 + bss)];
                    MPI_Irecv(recv[0][index].second, nnz * (1 + bss), Wrapper<K>::mpi_type(), in->recv_[index].first, 323, in->communicator_, rqRecv + in->recv_.size() + index);
                }
                for(unsigned short j = 0; j < in->recv_.size(); ++j) {
                    int index;
                    MPI_Waitany(in->recv_.size(), rqRecv + in->recv_.size(), &index, MPI_STATUS_IGNORE);
                    unsigned int accumulate = 0;
                    for(unsigned int j = 0; j < in->recv_[index].second.size(); ++j) {
                        unsigned int onnz = 0;
                        std::vector<int>::const_iterator it = nnz.cbegin();
                        for(unsigned int k = 0; k < recv[0][index].first[2 * j + 1]; ++k) {
                            it = std::lower_bound(it, nnz.cend(), HPDDM::lround(HPDDM::abs(recv[0][index].second[accumulate + recv[0][index].first[2 * j] * (1 + bss) + k])));
                            if(it != nnz.cend() && *it == HPDDM::lround(HPDDM::abs(recv[0][index].second[accumulate + recv[0][index].first[2 * j] * (1 + bss) + k])))
                                ++onnz;
                        }
                        for(unsigned int k = in->recv_[index].second[j]; k < in->off_; ++k)
                            di[in->dof_ + displs.back() + k + 1] += recv[0][index].first[2 * j] + onnz;
                        accumulate += (recv[0][index].first[2 * j] + recv[0][index].first[2 * j + 1]) * (1 + bss);
                    }
                }
                for(unsigned int i = 0; i < 2 * in->send_.size(); ++i) {
                    int index;
                    MPI_Waitany(2 * in->send_.size(), rqSend, &index, MPI_STATUS_IGNORE);
                    if(index < in->send_.size())
                        delete [] send[index].first;
                    else
                        delete [] send[index - in->send_.size()].second;
                }
                delete [] rqRecv;
            }
            else if(!sizes.empty()) {
                std::map<unsigned short, unsigned short> nnz;
                rows.reserve(sizes.size());
                rows.emplace_back(std::make_pair(sizes.cbegin(), in->bs_ > 1 ? in->bs_ : sizes.at(std::make_pair(sizes.cbegin()->first.first, sizes.cbegin()->first.first))));
                for(std::map<std::pair<unsigned short, unsigned short>, unsigned short>::const_iterator it = sizes.cbegin(); it != sizes.cend(); ++it) {
                    if(it->first.first != rows.back().first->first.first)
                        rows.emplace_back(std::make_pair(it, in->bs_ > 1 ? in->bs_ : sizes.at(std::make_pair(it->first.first, it->first.first))));
                }
                rows.emplace_back(std::make_pair(sizes.cend(), 0));
                unsigned int accumulate = 0;
                for(unsigned short k = 0; k < rows.size() - 1; ++k) {
                    unsigned int row = 0;
                    if(in->bs_ == 1) {
                        for(std::map<std::pair<unsigned short, unsigned short>, unsigned short>::const_iterator it = rows[k].first; it != rows[k + 1].first; ++it)
                            row += sizes.at(std::make_pair(it->first.second, it->first.second));
                        map[rows[k].first->first.first] = accumulate;
                        for(unsigned int j = 0; j < rows[k].second; ++j)
                            di[in->dof_ + accumulate + j + 1] += row * (j + 1) - (S == 'S' ? (j * (j + 1)) / 2 : 0);
                        accumulate += rows[k].second;
                        for(unsigned int j = accumulate; j < in->off_ + displs.back(); ++j)
                            di[in->dof_ + j + 1] += row * rows[k].second - (S == 'S' ? (rows[k].second * (rows[k].second - 1)) / 2 : 0);
                    }
                    else {
                        row += std::distance(rows[k].first, rows[k + 1].first);
                        map[rows[k].first->first.first] = accumulate;
                        di[in->dof_ + accumulate + 1] += row;
                        accumulate += 1;
                        for(unsigned int j = accumulate; j < in->off_ + displs.back(); ++j)
                            di[in->dof_ + j + 1] += row;
                    }
                }
            }
            integer_type* dj = new integer_type[di[in->dof_ + in->off_ + displs.back()]];
            K* da = new K[di[in->dof_ + in->off_ + displs.back()] * bss]();
            for(unsigned int i = 0; i < in->dof_; ++i) {
                for(unsigned int j = in->di_[i]; j < in->di_[i + 1]; ++j)
                    dj[di[i] + j - in->di_[i]] = in->dj_[j - (super::numbering_ == 'F')] - (super::numbering_ == 'F');
                for(unsigned int j = in->di_[i]; j < in->di_[i + 1]; ++j) {
#if HPDDM_PETSC
                    if(in == this && S == 'S' && i == in->dj_[j])
                        Wrapper<K>::template omatcopy<'T'>(in->bs_, in->bs_, in->da_ + (j - (super::numbering_ == 'F')) * bss, in->bs_, da + (di[i] + j - in->di_[i]) * bss, in->bs_);
                    else
#endif
                        std::copy_n(in->da_ + (j - (super::numbering_ == 'F')) * bss, bss, da + (di[i] + j - in->di_[i]) * bss);
                }
                unsigned int shift[2] = { 0, 0 };
                if(S == 'S') {
                    if(in == this) {
                        for(unsigned int j = 0; j < oa[i].size(); ++j) {
                            dj[di[i] + in->di_[i + 1] - in->di_[i] + j] = oa[i][j].first;
                            Wrapper<K>::template omatcopy<'T'>(in->bs_, in->bs_, oa[i][j].second, in->bs_, da + (di[i] + in->di_[i + 1] - in->di_[i] + j) * bss, in->bs_);
                        }
                        shift[0] = oa[i].size();
                        shift[1] = displs.back();
                    }
                    else if(!extra.empty()) {
                        std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::const_iterator it = extra.cbegin();
                        while(it != extra.cend()) {
                            if(std::get<1>(it->second) <= i && i < (std::get<1>(it->second) + std::get<0>(it->second)))
                                break;
                            ++it;
                        }
                        if(it != extra.cend()) {
                            for(const unsigned short& p : std::get<2>(it->second)) {
                                for(unsigned int j = 0; j < (in->bs_ > 1 ? 1 : sizes.at(std::make_pair(p, p))); ++j)
                                    dj[di[i] + shift[0] + j + in->di_[i + 1] - in->di_[i]] = in->dof_ + map[p] + j;
                                shift[0] += (in->bs_ > 1 ? 1 : sizes.at(std::make_pair(p, p)));
                            }
                        }
                        shift[1] = displs.back();
                    }
                }
                for(unsigned int j = in->oi_[i]; j < in->oi_[i + 1]; ++j)
                    dj[di[i] + shift[0] + j - in->oi_[i] + in->di_[i + 1] - in->di_[i]] = in->dof_ + shift[1] + in->oj_[j - (super::numbering_ == 'F')] - (super::numbering_ == 'F');
                for(unsigned int j = in->oi_[i]; j < in->oi_[i + 1]; ++j)
                    std::copy_n(in->oa_ + (j - (super::numbering_ == 'F')) * bss, bss, da + (di[i] + shift[0] + j - in->oi_[i] + in->di_[i + 1] - in->di_[i]) * bss);
            }
            if(S != 'S') {
                for(unsigned int i = 0; i < in->off_; ++i)
                    for(unsigned int j = 0; j < to[i].size(); ++j) {
                        dj[di[in->dof_ + i] + j] = to[i][j].first;
                        Wrapper<K>::template omatcopy<'T'>(in->bs_, in->bs_, to[i][j].second, in->bs_, da + (di[in->dof_ + i] + j) * bss, in->bs_);
                    }
            }
            if(in == this) {
                for(unsigned short i = 0; i < in->recv_.size(); ++i) {
                    unsigned int accumulate = 0;
                    for(unsigned int j = 0; j < in->recv_[i].second.size(); ++j) {
                        std::vector<std::pair<int, K*>> row;
                        row.reserve(di[in->dof_ + displs.back() + in->recv_[i].second[j] + 1] - di[in->dof_ + displs.back() + in->recv_[i].second[j]] - (S != 'S' ? to[in->recv_[i].second[j]].size() : 0));
                        for(unsigned int k = 0; k < recv[0][i].first[2 * j]; ++k) {
                            row.emplace_back(in->recv_[i].second[HPDDM::lround(HPDDM::abs(recv[0][i].second[accumulate * (1 + bss) + k]))], recv[0][i].second + accumulate * (1 + bss) + recv[0][i].first[2 * j] + k * bss);
                        }
                        for(unsigned int k = 0; k < recv[0][i].first[2 * j + 1]; ++k) {
                            std::vector<int>::const_iterator pt = std::lower_bound(nnz.cbegin(), nnz.cend(), HPDDM::lround(HPDDM::abs(recv[0][i].second[(accumulate + recv[0][i].first[2 * j]) * (1 + bss) + k])));
                            if(pt != nnz.cend() && *pt == HPDDM::lround(HPDDM::abs(recv[0][i].second[(accumulate + recv[0][i].first[2 * j]) * (1 + bss) + k]))) {
                                row.emplace_back(std::distance(nnz.cbegin(), pt), recv[0][i].second + (accumulate + recv[0][i].first[2 * j]) * (1 + bss) + recv[0][i].first[2 * j + 1] + k * bss);
                            }
                        }
                        std::sort(row.begin(), row.end());
                        for(unsigned int k = 0; k < row.size(); ++k) {
                            dj[di[in->dof_ + displs.back() + in->recv_[i].second[j]] + (S != 'S' ? to[in->recv_[i].second[j]].size() : 0) + k] = in->dof_ + displs.back() + row[k].first;
                            std::copy_n(row[k].second, bss, da + (di[in->dof_ + displs.back() + in->recv_[i].second[j]] + (S != 'S' ? to[in->recv_[i].second[j]].size() : 0) + k) * bss);
                        }
                        accumulate += recv[0][i].first[2 * j] + recv[0][i].first[2 * j + 1];
                    }
                    delete [] recv[0][i].first;
                    delete [] recv[0][i].second;
                }
                if(S == 'S') {
                    for(unsigned int i = in->dof_; i < in->dof_ + in->off_ + displs.back(); ++i) {
                        for(unsigned int j = 0; j < oa[i].size(); ++j) {
                            dj[di[i] + j] = oa[i][j].first;
                            std::copy_n(oa[i][j].second, bss, da + (di[i] + j) * bss);
                        }
                    }
                    delete [] oa;
                    for(unsigned short i = 0; i < in->send_.size(); ++i) {
                        delete [] recv[1][i].first;
                        delete [] recv[1][i].second;
                    }
                }
                delete [] recv;
            }
            else {
                unsigned int row = 0;
                const std::function<size_t(unsigned short, unsigned short)> key = [](unsigned short i, unsigned short j) { return static_cast<size_t>(i) << 32 | static_cast<size_t>(j); };
                std::unordered_map<size_t, unsigned int> loc;
                for(unsigned short k = 0; k < rows.size() - 1; ++k) {
                    unsigned int col = 0;
                    for(std::map<std::pair<unsigned short, unsigned short>, unsigned short>::const_iterator it = rows[k].first; it != rows[k + 1].first; ++it) {
                        loc[key(rows[k].first->first.first, it->first.second)] = di[in->dof_ + map[rows[k].first->first.first]] + (S != 'S' ? to[row].size() : 0) + col;
                        if(in->bs_ == 1) {
                            for(unsigned short j = 0; j < sizes.at(std::make_pair(it->first.second, it->first.second)); ++j) {
                                dj[di[in->dof_ + map[rows[k].first->first.first]] + (S != 'S' ? to[row].size() : 0) + col] = in->dof_ + map[it->first.second] + j;
                                ++col;
                            }
                        }
                        else {
                            dj[di[in->dof_ + map[rows[k].first->first.first]] + (S != 'S' ? to[row].size() : 0) + col] = in->dof_ + map[it->first.second];
                            ++col;
                        }
                    }
                    if(in->bs_ == 1) {
                        for(unsigned short j = 1; j < rows[k].second; ++j)
                            std::copy_n(dj + di[in->dof_ + map[rows[k].first->first.first]] + (S != 'S' ? to[row].size() : j), col - (S == 'S' ? j : 0), dj + di[in->dof_ + map[rows[k].first->first.first] + j] + (S != 'S' ? to[row + j].size() : 0));
                        row += rows[k].second;
                    }
                    else
                        row += 1;
                }
                for(unsigned int j = 0; j < reduction.size(); ++j) {
                    int index;
                    MPI_Status st;
                    MPI_Waitany(reduction.size(), std::get<2>(transfer), &index, &st);
                    if(st.MPI_TAG != MPI_ANY_TAG && st.MPI_SOURCE != MPI_ANY_SOURCE) {
                        unsigned int accumulate = std::get<0>(transfer)[index];
                        for(unsigned short j = 0; j < reduction[index].size(); ++j) {
                            const unsigned short r1 = reduction[index][j].first;
                            const unsigned short r2 = reduction[index][j].second;
                            const unsigned int row = std::distance(di, std::lower_bound(di + in->dof_, di + in->dof_ + in->off_ + displs.back(), loc[key(r1, r2)])) - (S == 'S' ? 0 : 1);
                            if(in->bs_ == 1) {
                                const unsigned short mu1 = sizes.at(std::make_pair(r1, r1));
                                const unsigned short mu2 = sizes.at(std::make_pair(r2, r2));
                                for(unsigned short nu1 = 0; nu1 < mu1; ++nu1)
                                    for(unsigned short nu2 = (S == 'S' && r1 == r2 ? nu1 : 0); nu2 < mu2; ++nu2) {
                                        if(S != 'S' || r1 != r2)
                                            da[loc[key(r1, r2)] + nu2 + nu1 * (di[row + 1] - di[row]) - (S == 'S' ? (nu1 * (nu1 - 1)) / 2 : 0)] += std::get<1>(transfer)[nu2 * mu1 + nu1 + accumulate];
                                        else
                                            da[loc[key(r1, r2)] + nu2 + nu1 * (di[row + 1] - di[row]) - (nu1 * (nu1 + 1)) / 2] += std::get<1>(transfer)[nu2 * mu1 + nu1 + accumulate];
                                    }
                                accumulate += mu1 * mu2;
                            }
                            else {
                                if(super::numbering_ == 'F')
                                    Blas<K>::axpy(&bss, &(Wrapper<K>::d__1), std::get<1>(transfer) + accumulate, &i__1, da + loc[key(r1, r2)] * bss, &i__1);
                                else
                                    for(unsigned short nu1 = 0; nu1 < in->bs_; ++nu1)
                                        for(unsigned short nu2 = 0; nu2 < in->bs_; ++nu2)
                                            da[loc[key(r1, r2)] * bss + nu2 * in->bs_ + nu1] += std::get<1>(transfer)[nu2 + nu1 * in->bs_ + accumulate];
                                accumulate += bss;
                            }
                        }
                        if(S == 'S') {
                            std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::const_iterator it = extra.find(index);
                            if(it != extra.cend()) {
                                unsigned int shift = 0;
                                for(const unsigned short& p : std::get<2>(it->second)) {
                                    const unsigned int row = std::get<1>(it->second);
                                    if(in->bs_ == 1) {
                                        const unsigned short mu1 = std::get<0>(it->second);
                                        const unsigned short mu2 = sizes.at(std::make_pair(p, p));
                                        for(unsigned short nu1 = 0; nu1 < mu1; ++nu1)
                                            for(unsigned short nu2 = 0; nu2 < mu2; ++nu2)
                                                da[di[row] + in->di_[row + 1] - in->di_[row] + shift + nu2 + nu1 * (di[row + 1] - di[row]) - (nu1 * (nu1 + 1)) / 2] += std::get<1>(transfer)[nu2 + nu1 * mu2 + accumulate];
                                        accumulate += mu1 * mu2;
                                        shift += mu2;
                                    }
                                    else {
                                        if(super::numbering_ == 'C')
                                            Blas<K>::axpy(&bss, &(Wrapper<K>::d__1), std::get<1>(transfer) + accumulate, &i__1, da + (di[row] + in->di_[row + 1] - in->di_[row]) * bss + shift, &i__1);
                                        else
                                            for(unsigned short nu1 = 0; nu1 < in->bs_; ++nu1)
                                                for(unsigned short nu2 = 0; nu2 < in->bs_; ++nu2)
                                                    da[(di[row] + in->di_[row + 1] - in->di_[row]) * bss + shift + nu2 * in->bs_ + nu1] += std::get<1>(transfer)[nu2 + nu1 * in->bs_ + accumulate];
                                        accumulate += bss;
                                        shift += bss;
                                    }
                                }
                            }
                        }
                    }
                }
                delete [] std::get<0>(transfer);
                delete [] std::get<1>(transfer);
                delete [] std::get<2>(transfer);
            }
#if !HPDDM_PETSC
            MatrixCSR<K>* ret = new MatrixCSR<K>(in->dof_ + in->off_ + displs.back(), in->dof_ + in->off_ + displs.back(), di[in->dof_ + in->off_ + displs.back()], da, di, dj, S == 'S', true);
            if(bss > 1) {
                int* di = new int[ret->n_ * in->bs_ + 1];
                for(int i = 0; i < ret->n_; ++i)
                    for(int k = 0; k < in->bs_; ++k)
                        di[i * in->bs_ + k] = ret->ia_[i] * bss + (ret->ia_[i + 1] - ret->ia_[i]) * in->bs_ * k - (S == 'S' ? i * (in->bs_ * (in->bs_ - 1)) / 2 + (k * (k - 1)) / 2 : 0);
                ret->nnz_ *= bss;
                if(S == 'S')
                    ret->nnz_ -= ret->n_ * (in->bs_ * (in->bs_ - 1)) / 2;
                di[ret->n_ * in->bs_] = ret->nnz_;
                int* dj = new int[ret->nnz_];
                for(int i = 0; i < ret->n_; ++i) {
                    for(int j = ret->ia_[i]; j < ret->ia_[i + 1]; ++j)
                        for(int k = 0; k < in->bs_; ++k)
                            dj[di[i * in->bs_] + (j - ret->ia_[i]) * in->bs_ + k] = ret->ja_[j] * in->bs_ + k;
                    for(int k = 1; k < in->bs_; ++k)
                        std::copy_n(dj + di[i * in->bs_] + (S == 'S' ? k : 0), (ret->ia_[i + 1] - ret->ia_[i]) * in->bs_ - (S == 'S' ? k : 0), dj + di[i * in->bs_ + k]);
                }
                K* da = new K[ret->nnz_];
                if(S != 'S') {
                    if(super::numbering_ == 'F') {
                        for(int i = 0; i < ret->n_; ++i)
                            Wrapper<K>::template omatcopy<'T'>((ret->ia_[i + 1] - ret->ia_[i]) * in->bs_, in->bs_, ret->a_ + ret->ia_[i] * bss, in->bs_, da + ret->ia_[i] * bss, (ret->ia_[i + 1] - ret->ia_[i]) * in->bs_);
                    }
                    else {
                        for(int i = 0; i < ret->n_; ++i)
                            for(int j = ret->ia_[i]; j < ret->ia_[i + 1]; ++j)
                                Wrapper<K>::template omatcopy<'N'>(in->bs_, in->bs_, ret->a_ + j * bss, in->bs_, da + ret->ia_[i] * bss + (j - ret->ia_[i]) * in->bs_, (ret->ia_[i + 1] - ret->ia_[i]) * in->bs_);
                    }
                }
                else {
                    for(int i = 0; i < ret->n_; ++i)
                        for(int j = ret->ia_[i]; j < ret->ia_[i + 1]; ++j) {
                            for(int nu = 0; nu < in->bs_; ++nu) {
                                for(int mu = (j == ret->ia_[i] ? nu : 0); mu < in->bs_; ++mu) {
                                    da[di[i * in->bs_ + nu] + (j - ret->ia_[i]) * in->bs_ + mu - nu] = ret->a_[j * bss + (super::numbering_ == 'F' ? mu * in->bs_ + nu : mu + nu * in->bs_)];
                                }
                            }
                        }
                }
                ret->n_ *= in->bs_;
                ret->m_ *= in->bs_;
                delete [] ret->ia_;
                ret->ia_ = di;
                delete [] ret->ja_;
                ret->ja_ = dj;
                delete [] ret->a_;
                ret->a_ = da;
            }
            if(S == 'S') {
                MatrixCSR<K>* t = new MatrixCSR<K>(ret->n_, ret->n_, ret->ia_[ret->n_], true);
                Wrapper<K>::template csrcsc<'C', HPDDM_NUMBERING>(&ret->n_, ret->a_, ret->ja_, ret->ia_, t->a_, t->ja_, t->ia_);
                delete ret;
                ret = t;
            }
            else if(HPDDM_NUMBERING == 'F') {
                std::for_each(ret->ja_, ret->ja_ + ret->ia_[ret->n_], [](int& i) { ++i; });
                std::for_each(ret->ia_, ret->ia_ + ret->n_, [](int& i) { ++i; });
            }
#else
            Mat ret;
            PetscCallContinue(MatCreate(PETSC_COMM_SELF, &ret));
            {
                const char* prefix;
                PetscCallContinue(KSPGetOptionsPrefix(s_->ksp, &prefix));
                PetscCallContinue(MatSetOptionsPrefix(ret, prefix));
            }
            PetscCallContinue(MatSetFromOptions(ret));
            PetscCallContinue(MatSetBlockSize(ret, in->bs_));
            PetscCallContinue(MatSetSizes(ret, (in->dof_ + in->off_ + displs.back()) * in->bs_, (in->dof_ + in->off_ + displs.back()) * in->bs_, (in->dof_ + in->off_ + displs.back()) * in->bs_, (in->dof_ + in->off_ + displs.back()) * in->bs_));
            if(S == 'S') {
                PetscCallContinue(MatSetType(ret, MATSEQSBAIJ));
                PetscCallContinue(MatSeqSBAIJSetPreallocationCSR(ret, in->bs_, di, dj, da));
            }
            else {
                if(in->bs_ > 1) {
                    PetscCallContinue(MatSetType(ret, MATSEQBAIJ));
                    PetscCallContinue(MatSeqBAIJSetPreallocationCSR(ret, in->bs_, di, dj, da));
                }
                else {
                    PetscCallContinue(MatSetType(ret, MATSEQAIJ));
                    PetscCallContinue(MatSeqAIJSetPreallocationCSR(ret, di, dj, da));
                }
            }
            delete [] da;
            delete [] dj;
            delete [] di;
#endif
            return ret;
        }
        typedef Solver
#if !HPDDM_PETSC
                      <K>
#endif
                          super;

        return_type buildThree(CoarseOperator<HPDDM_TYPES_COARSE_OPERATOR(Solver, S, K)>* const& A, const std::vector<std::vector<std::pair<unsigned short, unsigned short>>>& reduction, const std::map<std::pair<unsigned short, unsigned short>, unsigned short>& sizes, const std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>& extra
#if HPDDM_PETSC
                , Mat* D, Mat* N, PC_HPDDM_Level* const level
#endif
                ) {
#if HPDDM_PETSC
            PetscFunctionBeginUser;
#endif
            if(DMatrix::communicator_ != MPI_COMM_NULL) {
#if HPDDM_PETSC
                char S;
                {
                    Mat A;
                    PetscCall(KSPGetOperators(level->ksp, &A, nullptr));
                    PetscBool symmetric;
                    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPISBAIJ, &symmetric));
                    S = (symmetric ? 'S' : 'G');
                }
#endif
                std::tuple<int*, K*, MPI_Request*> transfer;
                std::get<0>(transfer) = new int[reduction.size() + 1]();
                for(unsigned short i = 0; i < reduction.size(); ++i) {
                    std::get<0>(transfer)[i + 1] = std::get<0>(transfer)[i];
                    for(unsigned short j = 0; j < reduction[i].size(); ++j)
                        std::get<0>(transfer)[i + 1] += (bs_ > 1 ? bs_ * bs_ : sizes.at(std::make_pair(reduction[i][j].first, reduction[i][j].first)) * sizes.at(std::make_pair(reduction[i][j].first, reduction[i][j].second)));
                    if(S == 'S') {
                        std::unordered_map<unsigned short, std::tuple<unsigned short, unsigned int, std::vector<unsigned short>>>::const_iterator it = extra.find(i);
                        if(it != extra.cend()) {
                            if(bs_ > 1)
                                std::get<0>(transfer)[i + 1] += bs_ * bs_ * std::get<2>(it->second).size();
                            else {
                                for(std::vector<unsigned short>::const_iterator p = std::get<2>(it->second).cbegin(); p != std::get<2>(it->second).cend(); ++p)
                                    std::get<0>(transfer)[i + 1] += std::get<0>(it->second) * sizes.at(std::make_pair(*p, *p));
                            }
                        }
                    }
                }
                std::get<1>(transfer) = new K[std::get<0>(transfer)[reduction.size()]];
                std::get<2>(transfer) = new MPI_Request[reduction.size()];
                for(unsigned short i = 0; i < reduction.size(); ++i) {
                    if(std::get<0>(transfer)[i + 1] - std::get<0>(transfer)[i])
                        MPI_Irecv(std::get<1>(transfer) + std::get<0>(transfer)[i], std::get<0>(transfer)[i + 1] - std::get<0>(transfer)[i], Wrapper<K>::mpi_type(), i, 300, A->getCommunicator(), std::get<2>(transfer) + i);
                    else
                        std::get<2>(transfer)[i] = MPI_REQUEST_NULL;
                }
                std::vector<std::vector<unsigned int>> off;
                std::vector<unsigned int> displs;
                unsigned int* ranges = nullptr;
                if(S == 'S') {
                    unsigned int* s = new unsigned int[3 * (recv_.size() + send_.size())];
                    ranges = new unsigned int[2 * (recv_.size() + send_.size())];
                    MPI_Request* rq = new MPI_Request[2 * (recv_.size() + send_.size())];
                    displs.resize(send_.size() + 1);
                    off.resize(recv_.size());
                    s += 3 * recv_.size();
                    for(const std::pair<const unsigned short, std::vector<int>>& p : send_) {
#if !HPDDM_PETSC
                        MPI_Irecv(s, 3, MPI_UNSIGNED, p.first, 13, communicator_, rq++);
#else
                        MPI_Irecv(s, 3, MPI_UNSIGNED, p.first, 13, DMatrix::communicator_, rq++);
#endif
                        s += 3;
                    }
                    s -= 3 * (recv_.size() + send_.size());
                    for(unsigned short i = 0; i < recv_.size(); ++i)
#if !HPDDM_PETSC
                        MPI_Irecv(ranges + 2 * (send_.size() + i), 2, MPI_UNSIGNED, recv_[i].first, 13, communicator_, rq++);
#else
                        MPI_Irecv(ranges + 2 * (send_.size() + i), 2, MPI_UNSIGNED, recv_[i].first, 13, DMatrix::communicator_, rq++);
#endif
                    for(const std::pair<const unsigned short, std::vector<int>>& p : send_)
#if !HPDDM_PETSC
                        MPI_Isend(range_, 2, MPI_UNSIGNED, p.first, 13, communicator_, rq++);
#else
                        MPI_Isend(range_, 2, MPI_UNSIGNED, p.first, 13, DMatrix::communicator_, rq++);
#endif
                    for(unsigned short i = 0; i < recv_.size(); ++i) {
                        off[i].reserve(dof_);
                        for(unsigned int j = 0; j < dof_; ++j) {
                            if(oi_[j + 1] - oi_[j]) {
                                integer_type* start = std::lower_bound(oj_ + oi_[j] - (super::numbering_ == 'F'), oj_ + oi_[j + 1] - (super::numbering_ == 'F'), recv_[i].second[0] + (super::numbering_ == 'F'));
                                integer_type* end = std::lower_bound(start, oj_ + oi_[j + 1] - (super::numbering_ == 'F'), recv_[i].second.back() + (super::numbering_ == 'F'));
                                if(start == oj_ + oi_[j + 1] - (super::numbering_ == 'F'))
                                    --start;
                                if(end == oj_ + oi_[j + 1] - (super::numbering_ == 'F'))
                                    --end;
                                if((*start - (super::numbering_ == 'F')) <= recv_[i].second.back() && recv_[i].second[0] <= (*end - (super::numbering_ == 'F')))
                                    off[i].emplace_back(j);
                            }
                        }
                        s[3 * i] = off[i].size();
                        s[3 * i + 1] = range_[0];
                        s[3 * i + 2] = range_[1];
#if !HPDDM_PETSC
                        MPI_Isend(s + 3 * i, 3, MPI_UNSIGNED, recv_[i].first, 13, communicator_, rq++);
#else
                        MPI_Isend(s + 3 * i, 3, MPI_UNSIGNED, recv_[i].first, 13, DMatrix::communicator_, rq++);
#endif
                    }
                    rq -= 2 * (recv_.size() + send_.size());
                    MPI_Waitall(send_.size(), rq, MPI_STATUSES_IGNORE);
                    displs[0] = 0;
                    for(unsigned short i = 0; i < send_.size(); ++i) {
                        displs[i + 1] = displs[i] + s[3 * (recv_.size() + i)];
                        ranges[2 * i] = s[3 * (recv_.size() + i) + 1];
                        ranges[2 * i + 1] = s[3 * (recv_.size() + i) + 2];
                    }
                    MPI_Waitall(send_.size() + 2 * recv_.size(), rq + send_.size(), MPI_STATUSES_IGNORE);
                    delete [] rq;
                    delete [] s;
                }
                else {
                    displs.resize(1);
                    displs[0] = 0;
                }
                auto overlapDirichlet = buildMatrix(this, displs, ranges, off);
                delete [] ranges;
                ranges = nullptr;
#if !HPDDM_PETSC
                s_ = new Schwarz<SUBDOMAIN, COARSEOPERATOR, S, K>;
                Option& opt = *Option::get();
                s_->setPrefix(opt.getPrefix());
#else
                Schwarz<K>* s = new Schwarz<K>;
                s_->P = s;
#endif
                std::vector<unsigned short> o;
                o.reserve(recv_.size() + (S == 'S' ? send_.size() : 0));
                std::vector<std::vector<int>> r(recv_.size() + (S == 'S' ? send_.size() : 0));
                unsigned short j = 0;
                if(S == 'S') {
                    for(const std::pair<const unsigned short, std::vector<int>>& p : send_) {
                        o.emplace_back(p.first);
                        r[j].resize((p.second.size() + displs[j + 1] - displs[j]) * bs_);
                        std::vector<int>::iterator it = r[j].begin();
                        for(unsigned int i = 0; i < displs[j + 1] - displs[j]; ++i)
                            for(unsigned int k = 0; k < bs_; ++k)
                                *it++ = (dof_ + displs[j] + i) * bs_ + k;
                        for(unsigned int i = 0; i < p.second.size(); ++i)
                            for(unsigned int k = 0; k < bs_; ++k)
                                *it++ = p.second[i] * bs_ + k;
                        ++j;
                    }
                }
                for(const pairNeighbor& p : recv_) {
                    o.emplace_back(p.first);
                    r[j].resize((p.second.size() + (S != 'S' ? send_[p.first].size() : off[j - send_.size()].size())) * bs_);
                    if(S != 'S' && p.first < DMatrix::rank_) {
                        for(unsigned int i = 0; i < p.second.size(); ++i)
                            for(unsigned int k = 0; k < bs_; ++k)
                                r[j][i * bs_ + k] = (dof_ + p.second[i]) * bs_ + k;
                        for(unsigned int i = 0; i < send_[p.first].size(); ++i)
                            for(unsigned int k = 0; k < bs_; ++k)
                                r[j][(p.second.size() + i) * bs_ + k] = send_[p.first][i] * bs_ + k;
                    }
                    else {
                        std::vector<int>::iterator it = r[j].begin();
                        if(S != 'S')
                            for(unsigned int i = 0; i < send_[p.first].size(); ++i)
                                for(unsigned int k = 0; k < bs_; ++k)
                                    *it++ = send_[p.first][i] * bs_ + k;
                        else {
                            for(unsigned int i = 0; i < off[j - send_.size()].size(); ++i) {
                                for(unsigned int k = 0; k < bs_; ++k)
                                    *it++ = off[j - send_.size()][i] * bs_ + k;
                            }
                        }
                        for(unsigned int i = 0; i < p.second.size(); ++i)
                            for(unsigned int k = 0; k < bs_; ++k)
                                *it++ = (dof_ + displs.back() + p.second[i]) * bs_ + k;
                    }
                    ++j;
                }
#if !HPDDM_PETSC
                s_->Subdomain<K>::initialize(overlapDirichlet, o, r, &communicator_);
                int m = overlapDirichlet->n_;
                underlying_type<K>* d = new underlying_type<K>[m]();
                s_->initialize(d);
#else
                const char* prefix;
                PetscInt m;
                IS is;
                PetscCall(KSPGetOptionsPrefix(s_->ksp, &prefix));
                PetscCall(MatGetLocalSize(overlapDirichlet, &m, nullptr));
                static_cast<Subdomain<K>*>(s)->initialize(nullptr, o, r, &(DMatrix::communicator_));
                s->setDof(m);
                PetscCall(VecCreateMPI(PETSC_COMM_SELF, m, PETSC_DETERMINE, &s_->D));
                {
                    Mat P;
                    Vec v;
                    PetscCall(KSPGetOperators(s_->ksp, &P, nullptr));
                    {
                        PetscInt n, N;
                        PetscCall(MatGetLocalSize(P, &n, nullptr));
                        PetscCall(MatGetSize(P, &N, nullptr));
                        PetscCall(VecCreateMPI(DMatrix::communicator_, n, N, &v));
                    }
                    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs_, m / bs_, idx_, PETSC_OWN_POINTER, &is));
                    PetscCall(VecScatterCreate(v, is, s_->D, nullptr, &s_->scatter));
                    PetscCall(VecDestroy(&v));
                }
                idx_ = nullptr;
                PetscCall(ISDestroy(&is));
                PetscReal* d;
                if(!std::is_same<PetscScalar, PetscReal>::value)
                    d = new PetscReal[m]();
                else {
                    PetscCall(VecSet(s_->D, 0.0));
                    PetscCall(VecGetArray(s_->D, reinterpret_cast<PetscScalar**>(&d)));
                }
#endif
                std::fill_n(d, dof_ * bs_, Wrapper<underlying_type<K>>::d__1);
#if HPDDM_PETSC
                s->initialize(d);
                if(!std::is_same<PetscScalar, PetscReal>::value) {
                    PetscScalar* c;
                    PetscCall(VecGetArray(s_->D, &c));
                    std::copy_n(d, m, c);
                }
                PetscCall(VecRestoreArray(s_->D, nullptr));
#endif
                if(A) {
                    auto overlapNeumann = buildMatrix(A, displs, ranges, off, reduction, sizes, extra, transfer);
#ifdef DMKL_PARDISO
                    delete [] A->da_;
#endif
                    {
#if !HPDDM_PETSC
                        std::vector<int> ia, ja;
                        ia.reserve(overlapDirichlet->n_ + 1);
                        ia.emplace_back(HPDDM_NUMBERING == 'F');
                        std::vector<K> a;
                        ja.reserve(overlapDirichlet->nnz_);
                        a.reserve(overlapDirichlet->nnz_);
                        for(unsigned int i = 0; i < overlapDirichlet->n_; ++i) {
                            if(std::abs(d[i]) > HPDDM_EPS)
                                for(unsigned int j = overlapDirichlet->ia_[i] - (HPDDM_NUMBERING == 'F'); j < overlapDirichlet->ia_[i + 1] - (HPDDM_NUMBERING == 'F'); ++j) {
                                    if(std::abs(d[overlapDirichlet->ja_[j]]) > HPDDM_EPS) {
                                        ja.emplace_back(overlapDirichlet->ja_[j]);
                                        a.emplace_back(overlapDirichlet->a_[j] * d[i] * d[ja.back() - (HPDDM_NUMBERING == 'F')]);
                                    }
                                }
                            ia.emplace_back(ja.size() + (HPDDM_NUMBERING == 'F'));
                        }
                        MatrixCSR<K> weighted(overlapDirichlet->n_, overlapDirichlet->m_, a.size(), a.data(), ia.data(), ja.data(), overlapDirichlet->sym_);
                        s_->template solveGEVP<EIGENSOLVER>(overlapNeumann, &weighted);
#elif HPDDM_SLEPC
                        delete [] di_;
                        delete [] da_;
                        delete [] ogj_;
                        delete [] range_;
                        delete [] oi_;
                        di_ = ogj_ = oi_ = nullptr;
                        range_ = nullptr;
                        da_ = nullptr;
                        std::map<unsigned short, std::vector<int>>().swap(send_);
                        vectorNeighbor().swap(recv_);
                        Mat weighted;
                        PetscCall(MatConvert(overlapDirichlet, MATSAME, MAT_INITIAL_MATRIX, &weighted));
                        PetscCall(MatDiagonalScale(weighted, s_->D, s_->D));
                        EPS eps;
                        ST st;
                        PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
                        PetscCall(EPSSetOptionsPrefix(eps, prefix));
                        PetscCall(EPSSetOperators(eps, overlapNeumann, weighted));
                        PetscCall(EPSSetProblemType(eps, EPS_GHEP));
                        PetscCall(EPSSetTarget(eps, 0.0));
                        PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
                        PetscCall(EPSGetST(eps, &st));
                        PetscCall(STSetType(st, STSINVERT));
                        PetscCall(STSetMatStructure(st, SAME_NONZERO_PATTERN));
                        PetscCall(EPSSetFromOptions(eps));
                        PetscInt nev, nconv;
                        PetscCall(EPSGetDimensions(eps, &nev, nullptr, nullptr));
                        PetscCall(EPSSolve(eps));
                        PetscCall(EPSGetConverged(eps, &nconv));
                        level->nu = std::min(nconv, nev);
                        if(level->threshold >= 0.0) {
                            PetscInt i = 0;
                            while(i < level->nu) {
                                PetscScalar eigr;
                                PetscCall(EPSGetEigenvalue(eps, i, &eigr, nullptr));
                                if(HPDDM::abs(eigr) > level->threshold)
                                    break;
                                ++i;
                            }
                            level->nu = i;
                        }
                        if(level->nu) {
                            K** ev = new K*[level->nu];
                            *ev = new K[m * level->nu];
                            for(unsigned short i = 1; i < level->nu; ++i)
                                ev[i] = *ev + i * m;
                            Vec vr;
                            PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m, ev[0], &vr));
                            for(int i = 0; i < level->nu; ++i) {
                                PetscCall(VecPlaceArray(vr, ev[i]));
                                PetscCall(EPSGetEigenvector(eps, i, vr, nullptr));
                                PetscCall(VecResetArray(vr));
                            }
                            PetscCall(VecDestroy(&vr));
                            s->setVectors(ev);
                        }
                        PetscCall(EPSDestroy(&eps));
                        PetscCall(MatDestroy(&weighted));
#endif
                    }
#if !HPDDM_PETSC
                    delete overlapNeumann;
                    s_->template buildTwo<0>(communicator_, nullptr);
#else
                    *D = overlapDirichlet;
                    *N = overlapNeumann;
#endif
                }
                else {
                    delete [] std::get<0>(transfer);
                    delete [] std::get<1>(transfer);
                    delete [] std::get<2>(transfer);
#if HPDDM_PETSC
                    *N = nullptr;
#endif
                }
#if !HPDDM_PETSC
                s_->callNumfact();
#endif
            }
#if HPDDM_PETSC
            else
                *D = *N = nullptr;
            PetscFunctionReturn(0);
#endif
        }
    private:
#if !HPDDM_PETSC
        template<char T>
        void exchange(const K* const in, K* const out, const unsigned short& mu = 1) const {
            for(unsigned short nu = 0; nu < mu; ++nu) {
                unsigned short i = (T == 'N' ? 0 : recv_.size());
                if(T == 'N')
                    while(i < recv_.size()) {
                        MPI_Irecv(buff_[i], recv_[i].second.size() * bs_, Wrapper<K>::mpi_type(), recv_[i].first, 10, communicator_, rq_ + i);
                        ++i;
                    }
                else
                    for(const std::pair<unsigned short, std::vector<int>>& p : send_) {
                        MPI_Irecv(buff_[i], p.second.size() * bs_, Wrapper<K>::mpi_type(), p.first, 20, communicator_, rq_ + i);
                        ++i;
                    }
                if(T == 'N')
                    for(const std::pair<unsigned short, std::vector<int>>& p : send_) {
                        for(unsigned int j = 0; j < p.second.size(); ++j)
                            std::copy_n(in + (nu * dof_ + p.second[j]) * bs_, bs_, buff_[i] + j * bs_);
                        MPI_Isend(buff_[i], p.second.size() * bs_, Wrapper<K>::mpi_type(), p.first, 10, communicator_, rq_ + i);
                        ++i;
                    }
                else {
                    i = 0;
                    while(i < recv_.size()) {
                        for(unsigned int j = 0; j < recv_[i].second.size(); ++j)
                            std::copy_n(o_ + (nu * off_ + recv_[i].second[j]) * bs_, bs_, buff_[i] + j * bs_);
                        MPI_Isend(buff_[i], recv_[i].second.size() * bs_, Wrapper<K>::mpi_type(), recv_[i].first, 20, communicator_, rq_ + i);
                        ++i;
                    }
                }
                if(nu != mu - 1)
                    wait<T>(T == 'N' ? o_ + nu * off_ * bs_ : out + nu * dof_ * bs_);
            }
        }
        template<char T>
        void wait(K* const in) const {
            if(T == 'N') {
                for(unsigned short i = 0; i < recv_.size(); ++i) {
                    int index;
                    MPI_Waitany(recv_.size(), rq_, &index, MPI_STATUS_IGNORE);
                    for(unsigned int j = 0; j < recv_[index].second.size(); ++j)
                        std::copy_n(buff_[index] + j * bs_, bs_, in + recv_[index].second[j] * bs_);
                }
                MPI_Waitall(send_.size(), rq_ + recv_.size(), MPI_STATUSES_IGNORE);
            }
            else {
                for(unsigned short i = 0; i < send_.size(); ++i) {
                    int index;
                    MPI_Status st;
                    MPI_Waitany(send_.size(), rq_ + recv_.size(), &index, &st);
                    const std::vector<int>& v = send_.at(st.MPI_SOURCE);
                    for(unsigned int j = 0; j < v.size(); ++j)
                        Blas<K>::axpy(&bs_, &(Wrapper<K>::d__1), buff_[recv_.size() + index] + j * bs_, &i__1, in + v[j] * bs_, &i__1);
                }
                MPI_Waitall(recv_.size(), rq_, MPI_STATUSES_IGNORE);
            }
        }
#endif
};
} // HPDDM
#endif // HPDDM_INEXACT_COARSE_OPERATOR_HPP_
