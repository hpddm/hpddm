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
/* Class: MatrixCSR
 *
 *  A class for storing sparse matrices in Compressed Sparse Row format.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class MatrixCSR {
    private:
#if INTEL_MKL_VERSION > 110299
#endif
    public:
        /* Variable: a
         *  Array of data. */
        K*      _a;
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
    private:
        /* Variable: free
         *  Sentinel value for knowing if the pointers <MatrixCSR::a>, <MatrixCSR::ia>, <MatrixCSR::ja> have to be freed. */
        bool _free;
    public:
        MatrixCSR() : _a(), _ia(), _ja(), _n(0), _m(0), _nnz(0), _sym(true), _free(true) { }
        MatrixCSR(const int& n, const int& m, const bool& sym) : _a(), _ia(new int[n + 1]), _ja(), _n(n), _m(m), _nnz(0),  _sym(sym), _free(true) { }
        MatrixCSR(const int& n, const int& m, const int& nnz, const bool& sym) : _a(new K[nnz]), _ia(new int[n + 1]), _ja(new int[nnz]), _n(n), _m(m), _nnz(nnz), _sym(sym), _free(true) { }
        MatrixCSR(const int& n, const int& m, const int& nnz, K* const& a, int* const& ia, int* const& ja, const bool& sym, const bool& takeOwnership = false) : _a(a), _ia(ia), _ja(ja), _n(n), _m(m), _nnz(nnz), _sym(sym), _free(takeOwnership) { }
        MatrixCSR(std::ifstream& file) {
            if(!file.good()) {
                _a = nullptr;
                _ia = _ja = nullptr;
                _n = _m = _nnz = 0;
            }
            else {
                std::string line;
                _n = _m = _nnz = 0;
                while(_nnz == 0 && std::getline(file, line)) {
                    if(line[0] != '#' && line[0] != '%') {
                        std::stringstream ss(line);
                        std::istream_iterator<std::string> begin(ss), end;
                        std::vector<std::string> vstrings(begin, end);
                        if(vstrings.size() == 1) {
                            if(_n == 0) {
                                _n = _m = sto<int>(vstrings[0]);
                                _sym = false;
                            }
                            else
                                _nnz = sto<int>(vstrings[0]);
                        }
                        else if(vstrings.size() == 3) {
                            _n = sto<int>(vstrings[0]);
                            _m = sto<int>(vstrings[1]);
                            _nnz = sto<int>(vstrings[2]);
                            _sym = false;
                        }
                        else if(vstrings.size() > 3) {
                            _n = sto<int>(vstrings[0]);
                            _m = sto<int>(vstrings[1]);
                            _sym = sto<int>(vstrings[2]);
                            _nnz = sto<int>(vstrings[3]);
                        }
                        else {
                            _a = nullptr;
                            _ia = _ja = nullptr;
                            _n = _m = _nnz = 0;
                        }
                    }
                }
                if(_n && _m) {
                    _ia = new int[_n + 1];
                    _ja = new int[_nnz];
                    _a = new K[_nnz];
                    _ia[0] = (HPDDM_NUMBERING == 'F');
                    std::fill_n(_ia + 1, _n, 0);
                    _nnz = 0;
                    bool order;
                    while(std::getline(file, line)) {
                        if(!line.empty() && line[0] != '#' && line[0] != '%') {
                            if(_nnz == 0) {
                                std::istringstream iss(line);
                                std::string word;
                                iss >> word;
                                order = Option::Arg::integer(std::string(), word, false);
                            }
                            int row;
                            if((order && scan<true>(line.c_str(), &row, _ja + _nnz, _a + _nnz)) || (!order && scan<false>(line.c_str(), &row, _ja + _nnz, _a + _nnz))) {
                                delete [] _a;
                                _a = nullptr;
                                delete [] _ja;
                                delete [] _ia;
                                _ia = _ja = nullptr;
                                _n = _m = _nnz = 0;
                                break;
                            }
                            if(HPDDM_NUMBERING == 'C')
                                _ja[_nnz]--;
                            ++_nnz;
                            _ia[row]++;
                        }
                    }
                    if(_ia)
                        std::partial_sum(_ia, _ia + _n + 1, _ia);
                }
            }
            _free = true;
        }
        ~MatrixCSR() {
            destroy();
        }
        /* Function: destroy
         *  Destroys the pointer <MatrixCSR::a>, <MatrixCSR::ia>, and <MatrixCSR::ja> using a custom deallocator if <MatrixCSR::free> is true. */
        void destroy(void (*dtor)(void*) = ::operator delete[]) {
            if(_free) {
                dtor(_a);
                dtor(_ia);
                dtor(_ja);
                _a = nullptr;
                _ia = _ja = nullptr;
            }
        }
        /* Function: sameSparsity
         *
         *  Checks whether the input matrix can be modified to have the same sparsity pattern as the calling object.
         *
         * Parameter:
         *    A              - Input matrix. */
        bool sameSparsity(MatrixCSR<K>* const& A) const {
            if(A->_sym == _sym && A->_nnz >= _nnz) {
                if(A->_ia == _ia && A->_ja == _ja)
                    return true;
                else if(!A->_free)
                    return false;
                else {
                    bool same = true;
                    K* a = new K[_nnz];
                    for(int i = 0; i < _n && same; ++i) {
                        for(int j = A->_ia[i], k = _ia[i]; j < A->_ia[i + 1]; ++j) {
                            while(k < _ia[i + 1] && _ja[k] < A->_ja[j])
                                a[k++] = K();
                            if(_ja[k] != A->_ja[j]) {
                                if(std::abs(A->_a[j]) > HPDDM_EPS)
                                    same = false;
                            }
                            else
                                a[k++] = A->_a[j];
                        }
                    }
                    if(same) {
                        A->_nnz = _nnz;
                        delete [] A->_ja;
                        delete [] A->_ia;
                        delete [] A->_a;
                        A->_a = a;
                        A->_ia = _ia;
                        A->_ja = _ja;
                        A->_free = false;
                    }
                    else
                        delete [] a;
                    return same;
                }
            }
            return false;
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
        template<char N, char M>
        const MatrixCSR<K>* symmetrizedStructure() const {
            std::vector<std::array<int, 3>> missingCoefficients;
            for(int i = 0; i < _n; ++i) {
                if(_ia[i + 1] == _ia[i])
                    missingCoefficients.emplace_back(std::array<int, 3>({{ _ia[i] - (N == 'F'), i , i }}));
                else {
                    int* diagonal;
                    if(!_sym) {
                        diagonal = std::lower_bound(_ja + _ia[i] - (N == 'F'), _ja + _ia[i + 1] - (N == 'F'), i + (N == 'F'));
                        for(int j = _ia[i] - (N == 'F'); j < _ia[i + 1] - (N == 'F'); ++j) {
                            if(j != std::distance(_ja, diagonal)) {
                                int* it = std::lower_bound(_ja + _ia[_ja[j] - (N == 'F')] - (N == 'F'), _ja + _ia[_ja[j] - (N == 'F') + 1] - (N == 'F'), i + (N == 'F'));
                                if(it == _ja + _ia[_ja[j] - (N == 'F') + 1] - (N == 'F') || *it != i + (N == 'F'))
                                    missingCoefficients.emplace_back(std::array<int, 3>({{ static_cast<int>(std::distance(_ja, it)), _ja[j] - (N == 'F'), i }}));
                            }
                        }
                    }
                    else
                        diagonal = _ja + _ia[i + 1] - (N == 'F') - 1;
                    if((!_sym && diagonal == _ja + _ia[i + 1] - (N == 'F')) || *diagonal != i + (N == 'F'))
                        missingCoefficients.emplace_back(std::array<int, 3>({{ static_cast<int>(std::distance(_ja, diagonal)), i, i }}));
                }
            }
            if(missingCoefficients.empty()) {
                if(N == 'C' && M == 'F') {
                    std::for_each(_ia, _ia + _n + 1, [](int& i) { ++i; });
                    std::for_each(_ja, _ja + _nnz, [](int& i) { ++i; });
                }
                else if(N == 'F' && M == 'C') {
                    std::for_each(_ia, _ia + _n + 1, [](int& i) { --i; });
                    std::for_each(_ja, _ja + _nnz, [](int& i) { --i; });
                }
                return this;
            }
            else {
                std::sort(missingCoefficients.begin(), missingCoefficients.end());
                MatrixCSR<K>* ret = new MatrixCSR<K>(_n, _m, _nnz + missingCoefficients.size(), _sym);
                if(N == 'C' && M == 'F')
                    std::transform(_ia, _ia + _n + 1, ret->_ia, [](int i) { return i + 1; });
                else if(N == 'F' && M == 'C')
                    std::transform(_ia, _ia + _n + 1, ret->_ia, [](int i) { return i - 1; });
                else
                    std::copy_n(_ia, _n + 1, ret->_ia);
                missingCoefficients.emplace_back(std::array<int, 3>({{ _nnz, 0 , 0 }}));
                unsigned int prev = 0;
                for(unsigned int i = 0; i < missingCoefficients.size(); ++i) {
                    if(N == 'C' && M == 'F')
                        std::transform(_ja + prev, _ja + missingCoefficients[i][0], ret->_ja + prev + i, [](int j) { return j + 1; });
                    else if(N == 'F' && M == 'C')
                        std::transform(_ja + prev, _ja + missingCoefficients[i][0], ret->_ja + prev + i, [](int j) { return j - 1; });
                    else
                        std::copy(_ja + prev, _ja + missingCoefficients[i][0], ret->_ja + prev + i);
                    std::copy(_a + prev, _a + missingCoefficients[i][0], ret->_a + prev + i);
                    if(i != missingCoefficients.size() - 1) {
                        ret->_ja[missingCoefficients[i][0] + i] = missingCoefficients[i][2] + (M == 'F');
                        ret->_a[missingCoefficients[i][0] + i] = K();
                        prev = missingCoefficients[i][0];
                        std::for_each(ret->_ia + missingCoefficients[i][1] + 1, ret->_ia + _n + 1, [](int& j) { j += 1; });
                    }
                }
                return ret;
            }
        }
        std::size_t hashIndices() const {
            std::size_t seed = 0;
            hash_range(seed, _ia, _ia + _n);
            hash_range(seed, _ja, _ja + _nnz);
            return seed;
        }
        /* Function: dump
         *
         *  Outputs the matrix to an output stream.
         *
         * Template Parameter:
         *    N              - 0- or 1-based indexing. */
        template<char N>
        std::ostream& dump(std::ostream& f) const {
            static_assert(N == 'C' || N == 'F', "Unknown numbering");
            f << "# First line: n m (is symmetric) nnz indexing\n";
            f << "# For each nonzero coefficient: i j a_ij such that (i, j) \\in  {1, ..., n} x {1, ..., m}\n";
            f << _n << " " << _m << " " << _sym << "  " << _nnz << " " << N << "\n";
            std::ios_base::fmtflags ff(f.flags());
            f << std::scientific;
            unsigned int k = _ia[0] - (N == 'F');
            for(unsigned int i = 0; i < _n; ++i)
                for(unsigned int ke = _ia[i + 1] - (N == 'F'); k < ke; ++k)
                    f << std::setw(9) << i + 1 << std::setw(9) << _ja[k] + (N == 'C') << " " << std::setw(13) << _a[k] << "\n";
            f.flags(ff);
            return f;
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
