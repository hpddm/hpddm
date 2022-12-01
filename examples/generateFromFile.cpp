/*
   This file is part of HPDDM.

   Author(s): Frédéric Nataf <nataf@ann.jussieu.fr>
              Pierre Jolivet <pierre@joliv.et>
        Date: 2016-05-18

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

extern "C" {
#define __parmetis_h__
typedef int idxtype;
#include <metis.h>
}

#define HPDDM_MINIMAL
#include "schwarz.hpp"

template<class K, typename std::enable_if<!HPDDM::Wrapper<K>::is_complex>::type* = nullptr>
void assign(std::mt19937& gen, std::uniform_real_distribution<K>& dis, K& x) {
    x = dis(gen);
}
template<class K, typename std::enable_if<HPDDM::Wrapper<K>::is_complex>::type* = nullptr>
void assign(std::mt19937& gen, std::uniform_real_distribution<HPDDM::underlying_type<K>>& dis, K& x) {
    x = K(dis(gen), dis(gen));
}

void generate(int rankWorld, int sizeWorld, std::list<int>& o, std::vector<std::vector<int>>& mapping, int& ndof, HPDDM::MatrixCSR<K>*& Mat, HPDDM::MatrixCSR<K>*& MatNeumann, HPDDM::underlying_type<K>*& d, K*& f, K*& sol) {
    HPDDM::Option& opt = *HPDDM::Option::get();
    std::vector<unsigned int> idx;
    Mat = nullptr;
    if(opt.prefix("matrix_filename").size()) {
        std::ifstream file(opt.prefix("matrix_filename"));
        Mat = new HPDDM::MatrixCSR<K>(file);
        ndof = Mat->_n;
    }
    int n;
    if(!Mat || Mat->_n == 0)
        return;
    else if(sizeWorld > 1) {
        int* part = new int[Mat->_n];
        int overlap;
        if(HPDDM_NUMBERING == 'F') {
            std::for_each(Mat->_ia, Mat->_ia + Mat->_n + 1, [](int& i) { --i; });
            std::for_each(Mat->_ja, Mat->_ja + Mat->_nnz, [](int& i) { --i; });
        }
#if METIS_VER_MAJOR >= 5
        METIS_PartGraphKway(&Mat->_n, const_cast<int*>(&(HPDDM::i__1)), Mat->_ia, Mat->_ja,
                            nullptr, nullptr, nullptr, &sizeWorld, nullptr, nullptr, nullptr, &overlap, part);
#else
        METIS_PartGraphKway(&Mat->_n, Mat->_ia, Mat->_ja, nullptr, nullptr, const_cast<int*>(&(HPDDM::i__0)),
                            const_cast<int*>(&(HPDDM::i__0)), &sizeWorld, const_cast<int*>(&(HPDDM::i__0)), &overlap, part);
#endif
        if(HPDDM_NUMBERING == 'F') {
            std::for_each(Mat->_ja, Mat->_ja + Mat->_nnz, [](int& i) { ++i; });
            std::for_each(Mat->_ia, Mat->_ia + Mat->_n + 1, [](int& i) { ++i; });
        }
        K* indicator = new K[sizeWorld * Mat->_n]();
        for(unsigned int i = 0; i < sizeWorld; ++i)
            std::transform(part, part + Mat->_n, indicator + i * Mat->_n, [&](const int& p) { return p == i; });
        delete [] part;
        K* val = new K[Mat->_nnz]();
        std::fill_n(val, Mat->_nnz, 1.0);
        overlap = opt.app()["overlap"];
        K* z = new K[Mat->_n * sizeWorld];
        for(unsigned short i = 0; i < overlap; ++i) {
            HPDDM::Wrapper<K>::csrmm(Mat->_sym , &Mat->_n, &sizeWorld, val, Mat->_ia, Mat->_ja, indicator, z);
            std::transform(indicator, indicator + sizeWorld * Mat->_n, z, z, [](const K& a, const K& b) { return (b > 0.5) - (a > 0.5); });
            K alpha = i + 2;
            n = Mat->_n * sizeWorld;
            HPDDM::Blas<K>::axpy(&n, &alpha, z, &(HPDDM::i__1), indicator, &(HPDDM::i__1));
        }
        delete [] z;
        delete [] val;

        idx.reserve(Mat->_n);
        for(unsigned int k = 0; k < Mat->_n; ++k) {
            if(indicator[rankWorld * Mat->_n + k] > 0.0)
                idx.emplace_back(k);
        }
        ndof = idx.size();
        std::unordered_map<unsigned int, unsigned int> g2l;
        g2l.reserve(ndof);
        for(unsigned int k = 0; k < ndof; ++k)
            g2l[idx[k]] = k ;

        for(unsigned int i = 0; i < sizeWorld; ++i)
            if(i != rankWorld) {
                std::vector<unsigned int> neighborIdx;
                for(unsigned int k = 0; k < Mat->_n; ++k) {
                    if(indicator[i * Mat->_n + k] > 0.0)
                        neighborIdx.push_back(k);
                }
                std::vector<int> intersection;
                intersection.reserve(ndof);
                std::set_intersection(idx.cbegin(), idx.cend(), neighborIdx.cbegin() , neighborIdx.cend() , back_inserter(intersection) );
                if(intersection.size()) {
                    o.emplace_back(i);
                    std::for_each(intersection.begin(), intersection.end(), [&](int& k) { k = g2l.at(k); });
                    mapping.emplace_back(intersection) ;
                }
            }
        int nnz = 0;
        d = new HPDDM::underlying_type<K>[ndof];
        for(unsigned int k = 0, j = 0; k < Mat->_n; ++k)
            if(indicator[rankWorld * Mat->_n + k] > 0.0) {
                if(std::abs(indicator[rankWorld * Mat->_n + k] - (1.0 + overlap)) < 0.5)
                    d[j++] = 0.0;
                else
                    d[j++] = 1.0 - (indicator[rankWorld * Mat->_n + k] - 1.0) / static_cast<double>(overlap);
                for(unsigned int i = Mat->_ia[k] - (HPDDM_NUMBERING == 'F'); i < Mat->_ia[k + 1] - (HPDDM_NUMBERING == 'F'); ++i) {
                    if(indicator[rankWorld * Mat->_n + Mat->_ja[i] - (HPDDM_NUMBERING == 'F')] > 0.0)
                        ++nnz;
                }
            }
        HPDDM::MatrixCSR<K>* locMat = new HPDDM::MatrixCSR<K>(ndof, ndof, nnz, Mat->_sym);
        locMat->_ia[0] = (HPDDM_NUMBERING == 'F');
        std::fill_n(locMat->_ia + 1, locMat->_n, 0);
        for(unsigned int k = 0, nnz = 0; k < Mat->_n; ++k)
            if(indicator[rankWorld * Mat->_n + k] > 0.0) {
                for(unsigned int i = Mat->_ia[k] - (HPDDM_NUMBERING == 'F'); i < Mat->_ia[k + 1] - (HPDDM_NUMBERING == 'F'); ++i) {
                    if(indicator[rankWorld * Mat->_n + Mat->_ja[i] - (HPDDM_NUMBERING == 'F')] > 0.0) {
                        locMat->_ia[g2l.at(k) + 1]++;
                        locMat->_ja[nnz] = g2l.at(Mat->_ja[i] - (HPDDM_NUMBERING == 'F')) + (HPDDM_NUMBERING == 'F');
                        locMat->_a[nnz++] = Mat->_a[i];
                    }
                }
            }
        std::partial_sum(locMat->_ia, locMat->_ia + locMat->_n + 1, locMat->_ia);
        delete [] indicator;

        n = Mat->_n;
        delete Mat;
        Mat = locMat;
    }
    else
        n = Mat->_n;
    f = new K[ndof];
    if(opt.prefix("rhs_filename").size()) {
        std::ifstream file(opt.prefix("rhs_filename"));
        std::string line;
        unsigned int i = 0, j = 0;
        K val;
        while(std::getline(file, line) && (idx.empty() || j < ndof)) {
            if(i == 0 && j == 0) {
                std::istringstream iss(line);
                std::string word;
                iss >> word;
                if(HPDDM::Option::Arg::integer(std::string(), word, false) && HPDDM::sto<int>(word) == n)
                    std::getline(file, line);
            }
            if(idx.empty() || i++ == idx[j]) {
                std::istringstream iss(line);
                iss >> val;
                f[j++] = val;
            }
        }
    }
    else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<HPDDM::underlying_type<K>> dis(0.0, 10.0);
        std::for_each(f, f + ndof, [&](K& x) { assign(gen, dis, x); });
    }
    sol = new K[ndof]();
}
