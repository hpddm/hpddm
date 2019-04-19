/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
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

#ifndef _HPDDM_MATRIX_
#define _HPDDM_MATRIX_

#include <iterator>
#include <array>

namespace HPDDM {
template<class K>
class MatrixBase {
    private:
#if INTEL_MKL_VERSION > 110299
#endif
    public:
        /* Variable: ia
         *  Array of row pointers. */
        int*   _ia;
        /* Variable: ja
         *  Array of column indices. */
        int*   _ja;
        /* Variable: n
         *  Number of rows. */
        int     _n;
        /* Variable: m
         *  Number of columns. */
        int     _m;
        /* Variable: nnz
         *  Number of nonzero entries. */
        int   _nnz;
        /* Variable: sym
         *  Symmetry of the matrix. */
        bool  _sym;
    protected:
        /* Variable: free
         *  Sentinel value for knowing if the pointers <MatrixBase::ia>, <MatrixBase::ja> have to be freed. */
        bool _free;
    public:
        MatrixBase() : _ia(), _ja(), _n(0), _m(0), _nnz(0), _sym(true), _free(true) { }
        MatrixBase(const int& n, const int& m, const bool& sym) : _ia(new int[n + 1]), _ja(), _n(n), _m(m), _nnz(0),  _sym(sym), _free(true) { }
        MatrixBase(const int& n, const int& m, const int& nnz, const bool& sym) : _ia(new int[n + 1]), _ja(new int[nnz]), _n(n), _m(m), _nnz(nnz), _sym(sym), _free(true) { }
        MatrixBase(const int& n, const int& m, const int& nnz, int* const& ia, int* const& ja, const bool& sym, const bool& takeOwnership = false) : _ia(ia), _ja(ja), _n(n), _m(m), _nnz(nnz), _sym(sym), _free(takeOwnership) { }
        MatrixBase(const MatrixBase&) = delete;
        ~MatrixBase() {
            destroy();
        }
        /* Function: destroy
         *  Destroys the pointer <MatrixBase::ia>, and <MatrixBase::ja> using a custom deallocator if <MatrixBase::free> is true. */
        void destroy(void (*dtor)(void*) = ::operator delete[]) {
            if(_free) {
                dtor(MatrixBase<K>::_ia);
                dtor(MatrixBase<K>::_ja);
                MatrixBase<K>::_ia = MatrixBase<K>::_ja = nullptr;
            }
        }
        template<char N>
        bool structurallySymmetric() const {
            for(unsigned int i = 0; i < _n; ++i) {
                bool diagonalCoefficient = false;
                for(unsigned int j = _ia[i] - (N == 'F'); j < _ia[i + 1] - (N == 'F'); ++j) {
                    if(_ja[j] != (i + (N == 'F'))) {
                        if(!std::binary_search(_ja + _ia[_ja[j] - (N == 'F')] - (N == 'F'), _ja + _ia[_ja[j] - (N == 'F') + 1] - (N == 'F'), i + (N == 'F')))
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
            hash_range(seed, _ia, _ia + _n);
            hash_range(seed, _ja, _ja + _nnz);
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
            f << MatrixBase<K>::_n << " " << MatrixBase<K>::_m << " " << MatrixBase<K>::_sym << "  " << MatrixBase<K>::_nnz << " " << N << "\n";
            std::ios_base::fmtflags ff(f.flags());
            f << std::scientific;
            unsigned int k = MatrixBase<K>::_ia[0] - (N == 'F');
            for(unsigned int i = 0; i < MatrixBase<K>::_n; ++i)
                for(unsigned int ke = MatrixBase<K>::_ia[i + 1] - (N == 'F'); k < ke; ++k)
                    f << std::setw(9) << i + 1 << " " << std::setw(9) << MatrixBase<K>::_ja[k] + (N == 'C') << " " << pts(a, k) << "\n";
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
    private:
#if INTEL_MKL_VERSION > 110299
#endif
    public:
        /* Variable: a
         *  Array of data. */
        K*      _a;
        MatrixCSR() : MatrixBase<K>(), _a() { }
        MatrixCSR(const int& n, const int& m, const bool& sym) : MatrixBase<K>(n, m, sym), _a() { }
        MatrixCSR(const int& n, const int& m, const int& nnz, const bool& sym) : MatrixBase<K>(n, m, nnz, sym), _a(new K[nnz]) { }
        MatrixCSR(const int& n, const int& m, const int& nnz, K* const& a, int* const& ia, int* const& ja, const bool& sym, const bool& takeOwnership = false) : MatrixBase<K>(n, m, nnz, ia, ja, sym, takeOwnership), _a(a) { }
        MatrixCSR(const MatrixCSR&) = delete;
        explicit MatrixCSR(std::ifstream& file) {
            if(!file.good()) {
                _a = nullptr;
                MatrixBase<K>::_ia = MatrixBase<K>::_ja = nullptr;
                MatrixBase<K>::_n = MatrixBase<K>::_m = MatrixBase<K>::_nnz = 0;
            }
            else {
                std::string line;
                MatrixBase<K>::_n = MatrixBase<K>::_m = MatrixBase<K>::_nnz = 0;
                while(MatrixBase<K>::_nnz == 0 && std::getline(file, line)) {
                    if(line[0] != '#' && line[0] != '%') {
                        std::stringstream ss(line);
                        std::istream_iterator<std::string> begin(ss), end;
                        std::vector<std::string> vstrings(begin, end);
                        if(vstrings.size() == 1) {
                            if(MatrixBase<K>::_n == 0) {
                                MatrixBase<K>::_n = MatrixBase<K>::_m = sto<int>(vstrings[0]);
                                MatrixBase<K>::_sym = false;
                            }
                            else
                                MatrixBase<K>::_nnz = sto<int>(vstrings[0]);
                        }
                        else if(vstrings.size() == 3) {
                            MatrixBase<K>::_n = sto<int>(vstrings[0]);
                            MatrixBase<K>::_m = sto<int>(vstrings[1]);
                            MatrixBase<K>::_nnz = sto<int>(vstrings[2]);
                            MatrixBase<K>::_sym = false;
                        }
                        else if(vstrings.size() > 3) {
                            MatrixBase<K>::_n = sto<int>(vstrings[0]);
                            MatrixBase<K>::_m = sto<int>(vstrings[1]);
                            MatrixBase<K>::_sym = sto<int>(vstrings[2]);
                            MatrixBase<K>::_nnz = sto<int>(vstrings[3]);
                        }
                        else {
                            _a = nullptr;
                            MatrixBase<K>::_ia = MatrixBase<K>::_ja = nullptr;
                            MatrixBase<K>::_n = MatrixBase<K>::_m = MatrixBase<K>::_nnz = 0;
                        }
                    }
                }
                if(MatrixBase<K>::_n && MatrixBase<K>::_m) {
                    MatrixBase<K>::_ia = new int[MatrixBase<K>::_n + 1];
                    MatrixBase<K>::_ja = new int[MatrixBase<K>::_nnz];
                    _a = new K[MatrixBase<K>::_nnz];
                    MatrixBase<K>::_ia[0] = (HPDDM_NUMBERING == 'F');
                    std::fill_n(MatrixBase<K>::_ia + 1, MatrixBase<K>::_n, 0);
                    MatrixBase<K>::_nnz = 0;
                    bool order;
                    while(std::getline(file, line)) {
                        if(!line.empty() && line[0] != '#' && line[0] != '%') {
                            if(MatrixBase<K>::_nnz == 0) {
                                std::istringstream iss(line);
                                std::string word;
                                iss >> word;
                                order = Option::Arg::integer(std::string(), word, false);
                            }
                            int row;
                            if((order && MatrixBase<K>::template scan<true>(line.c_str(), &row, MatrixBase<K>::_ja + MatrixBase<K>::_nnz, _a + MatrixBase<K>::_nnz)) || (!order && MatrixBase<K>::template scan<false>(line.c_str(), &row, MatrixBase<K>::_ja + MatrixBase<K>::_nnz, _a + MatrixBase<K>::_nnz))) {
                                delete [] _a;
                                _a = nullptr;
                                delete [] MatrixBase<K>::_ja;
                                delete [] MatrixBase<K>::_ia;
                                MatrixBase<K>::_ia = MatrixBase<K>::_ja = nullptr;
                                MatrixBase<K>::_n = MatrixBase<K>::_m = MatrixBase<K>::_nnz = 0;
                                break;
                            }
                            if(HPDDM_NUMBERING == 'C')
                                MatrixBase<K>::_ja[MatrixBase<K>::_nnz]--;
                            ++MatrixBase<K>::_nnz;
                            MatrixBase<K>::_ia[row]++;
                        }
                    }
                    if(MatrixBase<K>::_ia)
                        std::partial_sum(MatrixBase<K>::_ia, MatrixBase<K>::_ia + MatrixBase<K>::_n + 1, MatrixBase<K>::_ia);
                }
            }
            MatrixBase<K>::_free = true;
        }
        MatrixCSR(const MatrixCSR<K>* const& a, const MatrixCSR<void>* const& restriction, const unsigned int* const perm) {
            MatrixBase<K>::_sym = a->MatrixBase<K>::_sym;
            MatrixBase<K>::_free = true;
            std::vector<std::pair<int, K>> tmp;
            tmp.reserve(a->MatrixBase<K>::_nnz);
            MatrixBase<K>::_n = restriction->_n;
            MatrixBase<K>::_m = MatrixBase<K>::_n;
            if(a->MatrixBase<K>::_ia) {
                MatrixBase<K>::_ia = new int[restriction->_n + 1]();
                for(int i = 0; i < MatrixBase<K>::_n; ++i) {
                    for(int j = a->MatrixBase<K>::_ia[restriction->_ja[i]]; j < a->MatrixBase<K>::_ia[restriction->_ja[i] + 1]; ++j) {
                        unsigned int col = perm[a->MatrixBase<K>::_ja[j]];
                        if(col > 0)
                            tmp.emplace_back(std::make_pair(col - 1, a->_a[j]));
                    }
                    std::sort(tmp.begin() + MatrixBase<K>::_ia[i], tmp.end(), [](const std::pair<int, K>& lhs, const std::pair<int, K>& rhs) { return lhs.first < rhs.first; });
                    MatrixBase<K>::_ia[i + 1] = tmp.size();
                }
                MatrixBase<K>::_nnz = tmp.size();
                MatrixBase<K>::_ja = new int[tmp.size()];
                _a = new K[tmp.size()];
                for(unsigned int i = 0; i < tmp.size(); ++i) {
                    MatrixBase<K>::_ja[i] = tmp[i].first;
                    _a[i] = tmp[i].second;
                }
            }
            else {
                MatrixBase<K>::_ia = nullptr;
                MatrixBase<K>::_nnz = 0;
                MatrixBase<K>::_ja = nullptr;
                _a = nullptr;
            }
        }
        ~MatrixCSR() {
            destroy();
        }
        /* Function: destroy
         *  Destroys the pointer <MatrixCSR::a> using a custom deallocator if <MatrixCSR::free> is true. */
        void destroy(void (*dtor)(void*) = ::operator delete[]) {
            if(MatrixBase<K>::_free) {
                dtor(_a);
                _a = nullptr;
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
            if(A->MatrixBase<K>::_sym == MatrixBase<K>::_sym && A->MatrixBase<K>::_nnz >= MatrixBase<K>::_nnz) {
                if(A->MatrixBase<K>::_ia == MatrixBase<K>::_ia && A->MatrixBase<K>::_ja == MatrixBase<K>::_ja)
                    return true;
                else {
                    bool same = true;
                    K* a = new K[MatrixBase<K>::_nnz];
                    for(int i = 0; i < MatrixBase<K>::_n && same; ++i) {
                        for(int j = A->MatrixBase<K>::_ia[i], k = MatrixBase<K>::_ia[i]; j < A->MatrixBase<K>::_ia[i + 1]; ++j) {
                            while(k < MatrixBase<K>::_ia[i + 1] && MatrixBase<K>::_ja[k] < A->MatrixBase<K>::_ja[j])
                                a[k++] = K();
                            if(MatrixBase<K>::_ja[k] != A->MatrixBase<K>::_ja[j]) {
                                if(std::abs(A->_a[j]) > HPDDM_EPS || !A->MatrixBase<K>::_free)
                                    same = false;
                            }
                            else
                                a[k++] = A->_a[j];
                        }
                    }
                    if(same && A->MatrixBase<K>::_free) {
                        A->MatrixBase<K>::_nnz = MatrixBase<K>::_nnz;
                        delete [] A->MatrixBase<K>::_ja;
                        delete [] A->MatrixBase<K>::_ia;
                        delete [] A->_a;
                        A->_a = a;
                        A->MatrixBase<K>::_ia = MatrixBase<K>::_ia;
                        A->MatrixBase<K>::_ja = MatrixBase<K>::_ja;
                        A->MatrixBase<K>::_free = true;
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
            for(int i = 0; i < MatrixBase<K>::_n; ++i) {
                if(MatrixBase<K>::_ia[i + 1] == MatrixBase<K>::_ia[i])
                    missingCoefficients.emplace_back(std::array<int, 3>({{ MatrixBase<K>::_ia[i] - (N == 'F'), i , i }}));
                else {
                    int* diagonal;
                    if(!MatrixBase<K>::_sym) {
                        diagonal = std::lower_bound(MatrixBase<K>::_ja + MatrixBase<K>::_ia[i] - (N == 'F'), MatrixBase<K>::_ja + MatrixBase<K>::_ia[i + 1] - (N == 'F'), i + (N == 'F'));
                        for(int j = MatrixBase<K>::_ia[i] - (N == 'F'); j < MatrixBase<K>::_ia[i + 1] - (N == 'F'); ++j) {
                            if(j != std::distance(MatrixBase<K>::_ja, diagonal)) {
                                int* it = std::lower_bound(MatrixBase<K>::_ja + MatrixBase<K>::_ia[MatrixBase<K>::_ja[j] - (N == 'F')] - (N == 'F'), MatrixBase<K>::_ja + MatrixBase<K>::_ia[MatrixBase<K>::_ja[j] - (N == 'F') + 1] - (N == 'F'), i + (N == 'F'));
                                if(it == MatrixBase<K>::_ja + MatrixBase<K>::_ia[MatrixBase<K>::_ja[j] - (N == 'F') + 1] - (N == 'F') || *it != i + (N == 'F'))
                                    missingCoefficients.emplace_back(std::array<int, 3>({{ static_cast<int>(std::distance(MatrixBase<K>::_ja, it)), MatrixBase<K>::_ja[j] - (N == 'F'), i }}));
                            }
                        }
                    }
                    else
                        diagonal = MatrixBase<K>::_ja + MatrixBase<K>::_ia[i + 1] - (N == 'F') - 1;
                    if((!MatrixBase<K>::_sym && diagonal == MatrixBase<K>::_ja + MatrixBase<K>::_ia[i + 1] - (N == 'F')) || *diagonal != i + (N == 'F'))
                        missingCoefficients.emplace_back(std::array<int, 3>({{ static_cast<int>(std::distance(MatrixBase<K>::_ja, diagonal)), i, i }}));
                }
            }
            if(missingCoefficients.empty()) {
                if(N == 'C' && M == 'F') {
                    std::for_each(MatrixBase<K>::_ia, MatrixBase<K>::_ia + MatrixBase<K>::_n + 1, [](int& i) { ++i; });
                    std::for_each(MatrixBase<K>::_ja, MatrixBase<K>::_ja + MatrixBase<K>::_nnz, [](int& i) { ++i; });
                }
                else if(N == 'F' && M == 'C') {
                    std::for_each(MatrixBase<K>::_ia, MatrixBase<K>::_ia + MatrixBase<K>::_n + 1, [](int& i) { --i; });
                    std::for_each(MatrixBase<K>::_ja, MatrixBase<K>::_ja + MatrixBase<K>::_nnz, [](int& i) { --i; });
                }
                return this;
            }
            else {
                std::sort(missingCoefficients.begin(), missingCoefficients.end());
                MatrixCSR<K>* ret = new MatrixCSR<K>(MatrixBase<K>::_n, MatrixBase<K>::_m, MatrixBase<K>::_nnz + missingCoefficients.size(), MatrixBase<K>::_sym);
                if(N == 'C' && M == 'F')
                    std::transform(MatrixBase<K>::_ia, MatrixBase<K>::_ia + MatrixBase<K>::_n + 1, ret->MatrixBase<K>::_ia, [](int i) { return i + 1; });
                else if(N == 'F' && M == 'C')
                    std::transform(MatrixBase<K>::_ia, MatrixBase<K>::_ia + MatrixBase<K>::_n + 1, ret->MatrixBase<K>::_ia, [](int i) { return i - 1; });
                else
                    std::copy_n(MatrixBase<K>::_ia, MatrixBase<K>::_n + 1, ret->MatrixBase<K>::_ia);
                missingCoefficients.emplace_back(std::array<int, 3>({{ MatrixBase<K>::_nnz, 0 , 0 }}));
                unsigned int prev = 0;
                for(unsigned int i = 0; i < missingCoefficients.size(); ++i) {
                    if(N == 'C' && M == 'F')
                        std::transform(MatrixBase<K>::_ja + prev, MatrixBase<K>::_ja + missingCoefficients[i][0], ret->MatrixBase<K>::_ja + prev + i, [](int j) { return j + 1; });
                    else if(N == 'F' && M == 'C')
                        std::transform(MatrixBase<K>::_ja + prev, MatrixBase<K>::_ja + missingCoefficients[i][0], ret->MatrixBase<K>::_ja + prev + i, [](int j) { return j - 1; });
                    else
                        std::copy(MatrixBase<K>::_ja + prev, MatrixBase<K>::_ja + missingCoefficients[i][0], ret->MatrixBase<K>::_ja + prev + i);
                    std::copy(_a + prev, _a + missingCoefficients[i][0], ret->_a + prev + i);
                    if(i != missingCoefficients.size() - 1) {
                        ret->MatrixBase<K>::_ja[missingCoefficients[i][0] + i] = missingCoefficients[i][2] + (M == 'F');
                        ret->_a[missingCoefficients[i][0] + i] = K();
                        prev = missingCoefficients[i][0];
                        std::for_each(ret->MatrixBase<K>::_ia + missingCoefficients[i][1] + 1, ret->MatrixBase<K>::_ia + MatrixBase<K>::_n + 1, [](int& j) { j += 1; });
                    }
                }
                return ret;
            }
        }
        constexpr bool getFree() const {
            return MatrixBase<K>::_free;
        }
        template<char N>
        std::ostream& dump(std::ostream& f) const {
            return MatrixBase<K>::template dump<N>(f, _a);
        }
};
template<class K>
inline std::ostream& operator <<(std::ostream& f, const MatrixCSR<K>& m) {
    if(m._ia[m._n] == m._nnz)
        return m.template dump<'C'>(f);
    else if(m._ia[m._n] == m._nnz + 1)
        return m.template dump<'F'>(f);
    else
        return f << "Malformed CSR matrix" << std::endl;
}
} // HPDDM
#endif // _HPDDM_MATRIX_
