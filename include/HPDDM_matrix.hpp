/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2013-03-12

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

#ifndef HPDDM_MATRIX_HPP_
#define HPDDM_MATRIX_HPP_

#include <iterator>
#include <array>

namespace HPDDM {
template<class K>
class MatrixBase {
    public:
        /* Variable: ia
         *  Array of row pointers. */
        int*   ia_;
        /* Variable: ja
         *  Array of column indices. */
        int*   ja_;
        /* Variable: n
         *  Number of rows. */
        int     n_;
        /* Variable: m
         *  Number of columns. */
        int     m_;
        /* Variable: nnz
         *  Number of nonzero entries. */
        int   nnz_;
        /* Variable: sym
         *  Symmetry of the matrix. */
        bool  sym_;
    protected:
        /* Variable: free
         *  Sentinel value for knowing if the pointers <MatrixBase::ia>, <MatrixBase::ja> have to be freed. */
        bool free_;
    public:
        MatrixBase() : ia_(), ja_(), n_(0), m_(0), nnz_(0), sym_(true), free_(true) { }
        MatrixBase(const int& n, const int& m, const bool& sym) : ia_(new int[n + 1]), ja_(), n_(n), m_(m), nnz_(0),  sym_(sym), free_(true) { }
        MatrixBase(const int& n, const int& m, const int& nnz, const bool& sym) : ia_(new int[n + 1]), ja_(new int[nnz]), n_(n), m_(m), nnz_(nnz), sym_(sym), free_(true) { }
        MatrixBase(const int& n, const int& m, const int& nnz, int* const& ia, int* const& ja, const bool& sym, const bool& takeOwnership = false) : ia_(ia), ja_(ja), n_(n), m_(m), nnz_(nnz), sym_(sym), free_(takeOwnership) { }
        MatrixBase(const MatrixBase&) = delete;
        ~MatrixBase() {
            destroy();
        }
        /* Function: destroy
         *  Destroys the pointer <MatrixBase::ia>, and <MatrixBase::ja> using a custom deallocator if <MatrixBase::free> is true. */
        void destroy(void (*dtor)(void*) = ::operator delete[]) {
            if(free_) {
                dtor(MatrixBase<K>::ia_);
                dtor(MatrixBase<K>::ja_);
                MatrixBase<K>::ia_ = MatrixBase<K>::ja_ = nullptr;
            }
        }
        template<char N>
        bool structurallySymmetric() const {
            for(unsigned int i = 0; i < n_; ++i) {
                bool diagonalCoefficient = false;
                for(unsigned int j = ia_[i] - (N == 'F'); j < ia_[i + 1] - (N == 'F'); ++j) {
                    if(ja_[j] != (i + (N == 'F'))) {
                        if(!std::binary_search(ja_ + ia_[ja_[j] - (N == 'F')] - (N == 'F'), ja_ + ia_[ja_[j] - (N == 'F') + 1] - (N == 'F'), i + (N == 'F')))
                            return false;
                    }
                    else
                        diagonalCoefficient = true;
                }
                if(!diagonalCoefficient)
                    return false;
            }
            return true;
        }
        std::size_t hashIndices() const {
            std::size_t seed = 0;
            hash_range(seed, ia_, ia_ + n_);
            hash_range(seed, ja_, ja_ + nnz_);
            return seed;
        }
        template<bool I, class T, typename std::enable_if<!Wrapper<T>::is_complex>::type* = nullptr>
        static bool scan(const char* str, int* row, int* col, T* val) {
            double x;
            int ret = (I ? sscanf(str, "%i %i %le", row, col, &x) : sscanf(str, "%le %i %i", &x, row, col));
            *val = x;
            return ret != 3;
        }
        template<bool I, class T, typename std::enable_if<Wrapper<T>::is_complex>::type* = nullptr>
        static bool scan(const char* str, int* row, int* col, T* val) {
            double re, im;
            int ret = (I ? sscanf(str, "%i %i (%le,%le)", row, col, &re, &im) : sscanf(str, "(%le,%le) %i %i", &re, &im, row, col));
            *val = T(re, im);
            return ret != 4;
        }
    protected:
        /* Function: dump
         *
         *  Outputs the matrix to an output stream.
         *
         * Template Parameter:
         *    N              - 0- or 1-based indexing. */
        template<char N>
        std::ostream& dump(std::ostream& f, const K* const a) const {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            f << "# First line: n m (is symmetric) nnz indexing\n";
            f << "# For each nonzero coefficient: i j a_ij such that (i, j) \\in  {1, ..., n} x {1, ..., m}\n";
            f << MatrixBase<K>::n_ << " " << MatrixBase<K>::m_ << " " << MatrixBase<K>::sym_ << "  " << MatrixBase<K>::nnz_ << " " << N << "\n";
            std::ios_base::fmtflags ff(f.flags());
            f << std::scientific;
            unsigned int k = MatrixBase<K>::ia_[0] - (N == 'F');
            for(unsigned int i = 0; i < MatrixBase<K>::n_; ++i)
                for(unsigned int ke = MatrixBase<K>::ia_[i + 1] - (N == 'F'); k < ke; ++k)
                    f << std::setw(9) << i + 1 << " " << std::setw(9) << MatrixBase<K>::ja_[k] + (N == 'C') << " " << pts(a, k) << "\n";
            f.flags(ff);
            return f;
        }
};
template<class K>
class MatrixCSR;

template<>
class MatrixCSR<void> : public MatrixBase<void> {
    public:
        using MatrixBase<void>::MatrixBase;
        template<char N>
        std::ostream& dump(std::ostream& f) const {
            return MatrixBase<void>::template dump<N>(f, static_cast<void*>(nullptr));
        }
};
/* Class: MatrixCSR
 *
 *  A class for storing sparse matrices in Compressed Sparse Row format.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class MatrixCSR : public MatrixBase<K> {
    public:
        /* Variable: a
         *  Array of data. */
        K*      a_;
        MatrixCSR() : MatrixBase<K>(), a_() { }
        MatrixCSR(const int& n, const int& m, const bool& sym) : MatrixBase<K>(n, m, sym), a_() { }
        MatrixCSR(const int& n, const int& m, const int& nnz, const bool& sym) : MatrixBase<K>(n, m, nnz, sym), a_(new K[nnz]) { }
        MatrixCSR(const int& n, const int& m, const int& nnz, K* const& a, int* const& ia, int* const& ja, const bool& sym, const bool& takeOwnership = false) : MatrixBase<K>(n, m, nnz, ia, ja, sym, takeOwnership), a_(a) { }
        MatrixCSR(const MatrixCSR& B) : MatrixBase<K>(B.n_, B.m_, B.nnz_, B.sym_), a_(new K[B.nnz_]) {
            std::copy_n(B.ia_, MatrixBase<K>::n_ + 1, MatrixBase<K>::ia_);
            std::copy_n(B.ja_, MatrixBase<K>::nnz_, MatrixBase<K>::ja_);
            std::copy_n(B.a_, MatrixBase<K>::nnz_, a_);
        }
#if !HPDDM_PETSC
        explicit MatrixCSR(std::ifstream& file) {
            if(!file.good()) {
                a_ = nullptr;
                MatrixBase<K>::ia_ = MatrixBase<K>::ja_ = nullptr;
                MatrixBase<K>::n_ = MatrixBase<K>::m_ = MatrixBase<K>::nnz_ = 0;
            }
            else {
                std::string line;
                MatrixBase<K>::n_ = MatrixBase<K>::m_ = MatrixBase<K>::nnz_ = 0;
                while(MatrixBase<K>::nnz_ == 0 && std::getline(file, line)) {
                    if(line[0] != '#' && line[0] != '%') {
                        std::stringstream ss(line);
                        std::istream_iterator<std::string> begin(ss), end;
                        std::vector<std::string> vstrings(begin, end);
                        if(vstrings.size() == 1) {
                            if(MatrixBase<K>::n_ == 0) {
                                MatrixBase<K>::n_ = MatrixBase<K>::m_ = sto<int>(vstrings[0]);
                                MatrixBase<K>::sym_ = false;
                            }
                            else
                                MatrixBase<K>::nnz_ = sto<int>(vstrings[0]);
                        }
                        else if(vstrings.size() == 3) {
                            MatrixBase<K>::n_ = sto<int>(vstrings[0]);
                            MatrixBase<K>::m_ = sto<int>(vstrings[1]);
                            MatrixBase<K>::nnz_ = sto<int>(vstrings[2]);
                            MatrixBase<K>::sym_ = false;
                        }
                        else if(vstrings.size() > 3) {
                            MatrixBase<K>::n_ = sto<int>(vstrings[0]);
                            MatrixBase<K>::m_ = sto<int>(vstrings[1]);
                            MatrixBase<K>::sym_ = sto<int>(vstrings[2]);
                            MatrixBase<K>::nnz_ = sto<int>(vstrings[3]);
                        }
                        else {
                            a_ = nullptr;
                            MatrixBase<K>::ia_ = MatrixBase<K>::ja_ = nullptr;
                            MatrixBase<K>::n_ = MatrixBase<K>::m_ = MatrixBase<K>::nnz_ = 0;
                        }
                    }
                }
                if(MatrixBase<K>::n_ && MatrixBase<K>::m_) {
                    MatrixBase<K>::ia_ = new int[MatrixBase<K>::n_ + 1];
                    MatrixBase<K>::ja_ = new int[MatrixBase<K>::nnz_];
                    a_ = new K[MatrixBase<K>::nnz_];
                    MatrixBase<K>::ia_[0] = (HPDDM_NUMBERING == 'F');
                    std::fill_n(MatrixBase<K>::ia_ + 1, MatrixBase<K>::n_, 0);
                    MatrixBase<K>::nnz_ = 0;
                    bool order = true;
                    while(std::getline(file, line)) {
                        if(!line.empty() && line[0] != '#' && line[0] != '%') {
                            if(MatrixBase<K>::nnz_ == 0) {
                                std::istringstream iss(line);
                                std::string word;
                                iss >> word;
                                order = Option::Arg::integer(std::string(), word, false);
                            }
                            int row;
                            if((order && MatrixBase<K>::template scan<true>(line.c_str(), &row, MatrixBase<K>::ja_ + MatrixBase<K>::nnz_, a_ + MatrixBase<K>::nnz_)) || (!order && MatrixBase<K>::template scan<false>(line.c_str(), &row, MatrixBase<K>::ja_ + MatrixBase<K>::nnz_, a_ + MatrixBase<K>::nnz_))) {
                                delete [] a_;
                                a_ = nullptr;
                                delete [] MatrixBase<K>::ja_;
                                delete [] MatrixBase<K>::ia_;
                                MatrixBase<K>::ia_ = MatrixBase<K>::ja_ = nullptr;
                                MatrixBase<K>::n_ = MatrixBase<K>::m_ = MatrixBase<K>::nnz_ = 0;
                                break;
                            }
                            if(HPDDM_NUMBERING == 'C')
                                MatrixBase<K>::ja_[MatrixBase<K>::nnz_]--;
                            ++MatrixBase<K>::nnz_;
                            MatrixBase<K>::ia_[row]++;
                        }
                    }
                    if(MatrixBase<K>::ia_)
                        std::partial_sum(MatrixBase<K>::ia_, MatrixBase<K>::ia_ + MatrixBase<K>::n_ + 1, MatrixBase<K>::ia_);
                }
            }
            MatrixBase<K>::free_ = true;
        }
#endif
        MatrixCSR(const MatrixCSR<K>* const& a, const MatrixCSR<void>* const& restriction, const unsigned int* const perm) {
            MatrixBase<K>::sym_ = a->MatrixBase<K>::sym_;
            MatrixBase<K>::free_ = true;
            std::vector<std::pair<int, K>> tmp;
            tmp.reserve(a->MatrixBase<K>::nnz_);
            MatrixBase<K>::n_ = restriction->n_;
            MatrixBase<K>::m_ = MatrixBase<K>::n_;
            if(a->MatrixBase<K>::ia_) {
                MatrixBase<K>::ia_ = new int[restriction->n_ + 1]();
                for(int i = 0; i < MatrixBase<K>::n_; ++i) {
                    for(int j = a->MatrixBase<K>::ia_[restriction->ja_[i]]; j < a->MatrixBase<K>::ia_[restriction->ja_[i] + 1]; ++j) {
                        unsigned int col = perm[a->MatrixBase<K>::ja_[j]];
                        if(col > 0)
                            tmp.emplace_back(std::make_pair(col - 1, a->a_[j]));
                    }
                    std::sort(tmp.begin() + MatrixBase<K>::ia_[i], tmp.end(), [](const std::pair<int, K>& lhs, const std::pair<int, K>& rhs) { return lhs.first < rhs.first; });
                    MatrixBase<K>::ia_[i + 1] = tmp.size();
                }
                MatrixBase<K>::nnz_ = tmp.size();
                MatrixBase<K>::ja_ = new int[tmp.size()];
                a_ = new K[tmp.size()];
                for(unsigned int i = 0; i < tmp.size(); ++i) {
                    MatrixBase<K>::ja_[i] = tmp[i].first;
                    a_[i] = tmp[i].second;
                }
            }
            else {
                MatrixBase<K>::ia_ = nullptr;
                MatrixBase<K>::nnz_ = 0;
                MatrixBase<K>::ja_ = nullptr;
                a_ = nullptr;
            }
        }
        ~MatrixCSR() {
            destroy();
        }
        /* Function: destroy
         *  Destroys the pointer <MatrixCSR::a> using a custom deallocator if <MatrixCSR::free> is true. */
        void destroy(void (*dtor)(void*) = ::operator delete[]) {
            if(MatrixBase<K>::free_) {
                dtor(a_);
                a_ = nullptr;
                MatrixBase<K>::destroy(dtor);
            }
        }
        /* Function: sameSparsity
         *
         *  Checks whether the input matrix can be modified to have the same sparsity pattern as the calling object.
         *
         * Parameter:
         *    A              - Input matrix. */
        bool sameSparsity(MatrixCSR<K>* const& A) const {
            if(A->MatrixBase<K>::sym_ == MatrixBase<K>::sym_ && A->MatrixBase<K>::nnz_ >= MatrixBase<K>::nnz_) {
                if(A->MatrixBase<K>::ia_ == MatrixBase<K>::ia_ && A->MatrixBase<K>::ja_ == MatrixBase<K>::ja_)
                    return true;
                else {
                    bool same = true;
                    K* a = new K[MatrixBase<K>::nnz_];
                    for(int i = 0; i < MatrixBase<K>::n_ && same; ++i) {
                        for(int j = A->MatrixBase<K>::ia_[i], k = MatrixBase<K>::ia_[i]; j < A->MatrixBase<K>::ia_[i + 1]; ++j) {
                            while(k < MatrixBase<K>::nnz_ && k < MatrixBase<K>::ia_[i + 1] && MatrixBase<K>::ja_[k] < A->MatrixBase<K>::ja_[j])
                                a[k++] = K();
                            if(k == MatrixBase<K>::nnz_ || MatrixBase<K>::ja_[k] != A->MatrixBase<K>::ja_[j]) {
                                if(j == A->MatrixBase<K>::nnz_ || std::abs(A->a_[j]) > HPDDM_EPS || !A->MatrixBase<K>::free_)
                                    same = false;
                            }
                            else
                                a[k++] = A->a_[j];
                        }
                    }
                    if(same && A->MatrixBase<K>::free_) {
                        A->MatrixBase<K>::nnz_ = MatrixBase<K>::nnz_;
                        delete [] A->MatrixBase<K>::ja_;
                        delete [] A->MatrixBase<K>::ia_;
                        delete [] A->a_;
                        A->a_ = a;
                        A->MatrixBase<K>::ia_ = MatrixBase<K>::ia_;
                        A->MatrixBase<K>::ja_ = MatrixBase<K>::ja_;
                        A->MatrixBase<K>::free_ = true;
                    }
                    else
                        delete [] a;
                    return same;
                }
            }
            return false;
        }
        template<char N, char M>
        const MatrixCSR<K>* symmetrizedStructure() const {
            std::vector<std::array<int, 3>> missingCoefficients;
            for(int i = 0; i < MatrixBase<K>::n_; ++i) {
                if(MatrixBase<K>::ia_[i + 1] == MatrixBase<K>::ia_[i])
                    missingCoefficients.emplace_back(std::array<int, 3>({{ MatrixBase<K>::ia_[i] - (N == 'F'), i , i }}));
                else {
                    int* diagonal;
                    if(!MatrixBase<K>::sym_) {
                        diagonal = std::lower_bound(MatrixBase<K>::ja_ + MatrixBase<K>::ia_[i] - (N == 'F'), MatrixBase<K>::ja_ + MatrixBase<K>::ia_[i + 1] - (N == 'F'), i + (N == 'F'));
                        for(int j = MatrixBase<K>::ia_[i] - (N == 'F'); j < MatrixBase<K>::ia_[i + 1] - (N == 'F'); ++j) {
                            if(j != std::distance(MatrixBase<K>::ja_, diagonal)) {
                                int* it = std::lower_bound(MatrixBase<K>::ja_ + MatrixBase<K>::ia_[MatrixBase<K>::ja_[j] - (N == 'F')] - (N == 'F'), MatrixBase<K>::ja_ + MatrixBase<K>::ia_[MatrixBase<K>::ja_[j] - (N == 'F') + 1] - (N == 'F'), i + (N == 'F'));
                                if(it == MatrixBase<K>::ja_ + MatrixBase<K>::ia_[MatrixBase<K>::ja_[j] - (N == 'F') + 1] - (N == 'F') || *it != i + (N == 'F'))
                                    missingCoefficients.emplace_back(std::array<int, 3>({{ static_cast<int>(std::distance(MatrixBase<K>::ja_, it)), MatrixBase<K>::ja_[j] - (N == 'F'), i }}));
                            }
                        }
                    }
                    else
                        diagonal = MatrixBase<K>::ja_ + MatrixBase<K>::ia_[i + 1] - (N == 'F') - 1;
                    if((!MatrixBase<K>::sym_ && diagonal == MatrixBase<K>::ja_ + MatrixBase<K>::ia_[i + 1] - (N == 'F')) || *diagonal != i + (N == 'F'))
                        missingCoefficients.emplace_back(std::array<int, 3>({{ static_cast<int>(std::distance(MatrixBase<K>::ja_, diagonal)), i, i }}));
                }
            }
            if(missingCoefficients.empty()) {
                if(N == 'C' && M == 'F') {
                    std::for_each(MatrixBase<K>::ia_, MatrixBase<K>::ia_ + MatrixBase<K>::n_ + 1, [](int& i) { ++i; });
                    std::for_each(MatrixBase<K>::ja_, MatrixBase<K>::ja_ + MatrixBase<K>::nnz_, [](int& i) { ++i; });
                }
                else if(N == 'F' && M == 'C') {
                    std::for_each(MatrixBase<K>::ia_, MatrixBase<K>::ia_ + MatrixBase<K>::n_ + 1, [](int& i) { --i; });
                    std::for_each(MatrixBase<K>::ja_, MatrixBase<K>::ja_ + MatrixBase<K>::nnz_, [](int& i) { --i; });
                }
                return this;
            }
            else {
                std::sort(missingCoefficients.begin(), missingCoefficients.end());
                MatrixCSR<K>* ret = new MatrixCSR<K>(MatrixBase<K>::n_, MatrixBase<K>::m_, MatrixBase<K>::nnz_ + missingCoefficients.size(), MatrixBase<K>::sym_);
                if(N == 'C' && M == 'F')
                    std::transform(MatrixBase<K>::ia_, MatrixBase<K>::ia_ + MatrixBase<K>::n_ + 1, ret->MatrixBase<K>::ia_, [](int i) { return i + 1; });
                else if(N == 'F' && M == 'C')
                    std::transform(MatrixBase<K>::ia_, MatrixBase<K>::ia_ + MatrixBase<K>::n_ + 1, ret->MatrixBase<K>::ia_, [](int i) { return i - 1; });
                else
                    std::copy_n(MatrixBase<K>::ia_, MatrixBase<K>::n_ + 1, ret->MatrixBase<K>::ia_);
                missingCoefficients.emplace_back(std::array<int, 3>({{ MatrixBase<K>::nnz_, 0 , 0 }}));
                unsigned int prev = 0;
                for(unsigned int i = 0; i < missingCoefficients.size(); ++i) {
                    if(N == 'C' && M == 'F')
                        std::transform(MatrixBase<K>::ja_ + prev, MatrixBase<K>::ja_ + missingCoefficients[i][0], ret->MatrixBase<K>::ja_ + prev + i, [](int j) { return j + 1; });
                    else if(N == 'F' && M == 'C')
                        std::transform(MatrixBase<K>::ja_ + prev, MatrixBase<K>::ja_ + missingCoefficients[i][0], ret->MatrixBase<K>::ja_ + prev + i, [](int j) { return j - 1; });
                    else
                        std::copy(MatrixBase<K>::ja_ + prev, MatrixBase<K>::ja_ + missingCoefficients[i][0], ret->MatrixBase<K>::ja_ + prev + i);
                    std::copy(a_ + prev, a_ + missingCoefficients[i][0], ret->a_ + prev + i);
                    if(i != missingCoefficients.size() - 1) {
                        ret->MatrixBase<K>::ja_[missingCoefficients[i][0] + i] = missingCoefficients[i][2] + (M == 'F');
                        ret->a_[missingCoefficients[i][0] + i] = K();
                        prev = missingCoefficients[i][0];
                        std::for_each(ret->MatrixBase<K>::ia_ + missingCoefficients[i][1] + 1, ret->MatrixBase<K>::ia_ + MatrixBase<K>::n_ + 1, [](int& j) { j += 1; });
                    }
                }
                return ret;
            }
        }
        constexpr bool getFree() const {
            return MatrixBase<K>::free_;
        }
        template<char N>
        std::ostream& dump(std::ostream& f) const {
            return MatrixBase<K>::template dump<N>(f, a_);
        }
};
template<class K>
inline std::ostream& operator <<(std::ostream& f, const MatrixCSR<K>& m) {
    if(m.ia_[m.n_] == m.nnz_)
        return m.template dump<'C'>(f);
    else if(m.ia_[m.n_] == m.nnz_ + 1)
        return m.template dump<'F'>(f);
    else
        return f << "Malformed CSR matrix" << std::endl;
}
} // HPDDM
#endif // HPDDM_MATRIX_HPP_
