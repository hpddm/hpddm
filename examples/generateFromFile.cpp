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

template <class K, typename std::enable_if<!HPDDM::Wrapper<K>::is_complex>::type * = nullptr>
void assign(std::mt19937 &gen, std::uniform_real_distribution<K> &dis, K &x)
{
  x = dis(gen);
}
template <class K, typename std::enable_if<HPDDM::Wrapper<K>::is_complex>::type * = nullptr>
void assign(std::mt19937 &gen, std::uniform_real_distribution<HPDDM::underlying_type<K>> &dis, K &x)
{
  x = K(dis(gen), dis(gen));
}

void generate(int rankWorld, int sizeWorld, std::list<int> &o, std::vector<std::vector<int>> &mapping, int &ndof, HPDDM::MatrixCSR<K> *&Mat, HPDDM::MatrixCSR<K> *&, HPDDM::underlying_type<K> *&d, K *&f, K *&sol)
{
  HPDDM::Option            &opt = *HPDDM::Option::get();
  std::vector<unsigned int> idx;
  Mat = nullptr;
  if (opt.prefix("matrix_filename").size()) {
    std::ifstream file(opt.prefix("matrix_filename"));
    Mat  = new HPDDM::MatrixCSR<K>(file);
    ndof = Mat->n_;
  }
  int n;
  if (!Mat || Mat->n_ == 0) return;
  else if (sizeWorld > 1) {
    int *part = new int[Mat->n_];
    int  overlap;
    if (HPDDM_NUMBERING == 'F') {
      std::for_each(Mat->ia_, Mat->ia_ + Mat->n_ + 1, [](int &i) { --i; });
      std::for_each(Mat->ja_, Mat->ja_ + Mat->nnz_, [](int &i) { --i; });
    }
#if METIS_VER_MAJOR >= 5
    METIS_PartGraphKway(&Mat->n_, const_cast<int *>(&(HPDDM::i__1)), Mat->ia_, Mat->ja_, nullptr, nullptr, nullptr, &sizeWorld, nullptr, nullptr, nullptr, &overlap, part);
#else
    METIS_PartGraphKway(&Mat->n_, Mat->ia_, Mat->ja_, nullptr, nullptr, const_cast<int *>(&(HPDDM::i__0)), const_cast<int *>(&(HPDDM::i__0)), &sizeWorld, const_cast<int *>(&(HPDDM::i__0)), &overlap, part);
#endif
    if (HPDDM_NUMBERING == 'F') {
      std::for_each(Mat->ja_, Mat->ja_ + Mat->nnz_, [](int &i) { ++i; });
      std::for_each(Mat->ia_, Mat->ia_ + Mat->n_ + 1, [](int &i) { ++i; });
    }
    K *indicator = new K[sizeWorld * Mat->n_]();
    for (unsigned int i = 0; i < sizeWorld; ++i) std::transform(part, part + Mat->n_, indicator + i * Mat->n_, [&](const int &p) { return p == i; });
    delete[] part;
    K *val = new K[Mat->nnz_]();
    std::fill_n(val, Mat->nnz_, 1.0);
    overlap = opt.app()["overlap"];
    K *z    = new K[Mat->n_ * sizeWorld];
    for (unsigned short i = 0; i < overlap; ++i) {
      HPDDM::Wrapper<K>::csrmm(Mat->sym_, &Mat->n_, &sizeWorld, val, Mat->ia_, Mat->ja_, indicator, z);
      std::transform(indicator, indicator + sizeWorld * Mat->n_, z, z, [](const K &a, const K &b) { return (b > 0.5) - (a > 0.5); });
      K alpha = i + 2;
      n       = Mat->n_ * sizeWorld;
      HPDDM::Blas<K>::axpy(&n, &alpha, z, &(HPDDM::i__1), indicator, &(HPDDM::i__1));
    }
    delete[] z;
    delete[] val;

    idx.reserve(Mat->n_);
    for (unsigned int k = 0; k < Mat->n_; ++k) {
      if (indicator[rankWorld * Mat->n_ + k] > 0.0) idx.emplace_back(k);
    }
    ndof = idx.size();
    std::unordered_map<unsigned int, unsigned int> g2l;
    g2l.reserve(ndof);
    for (unsigned int k = 0; k < ndof; ++k) g2l[idx[k]] = k;

    for (unsigned int i = 0; i < sizeWorld; ++i)
      if (i != rankWorld) {
        std::vector<unsigned int> neighborIdx;
        for (unsigned int k = 0; k < Mat->n_; ++k) {
          if (indicator[i * Mat->n_ + k] > 0.0) neighborIdx.push_back(k);
        }
        std::vector<int> intersection;
        intersection.reserve(ndof);
        std::set_intersection(idx.cbegin(), idx.cend(), neighborIdx.cbegin(), neighborIdx.cend(), back_inserter(intersection));
        if (intersection.size()) {
          o.emplace_back(i);
          std::for_each(intersection.begin(), intersection.end(), [&](int &k) { k = g2l.at(k); });
          mapping.emplace_back(intersection);
        }
      }
    int nnz = 0;
    d       = new HPDDM::underlying_type<K>[ndof];
    for (unsigned int k = 0, j = 0; k < Mat->n_; ++k)
      if (indicator[rankWorld * Mat->n_ + k] > 0.0) {
        if (std::abs(indicator[rankWorld * Mat->n_ + k] - (1.0 + overlap)) < 0.5) d[j++] = 0.0;
        else d[j++] = 1.0 - (indicator[rankWorld * Mat->n_ + k] - 1.0) / static_cast<double>(overlap);
        for (unsigned int i = Mat->ia_[k] - (HPDDM_NUMBERING == 'F'); i < Mat->ia_[k + 1] - (HPDDM_NUMBERING == 'F'); ++i) {
          if (indicator[rankWorld * Mat->n_ + Mat->ja_[i] - (HPDDM_NUMBERING == 'F')] > 0.0) ++nnz;
        }
      }
    HPDDM::MatrixCSR<K> *locMat = new HPDDM::MatrixCSR<K>(ndof, ndof, nnz, Mat->sym_);
    locMat->ia_[0]              = (HPDDM_NUMBERING == 'F');
    std::fill_n(locMat->ia_ + 1, locMat->n_, 0);
    for (unsigned int k = 0, nnz = 0; k < Mat->n_; ++k)
      if (indicator[rankWorld * Mat->n_ + k] > 0.0) {
        for (unsigned int i = Mat->ia_[k] - (HPDDM_NUMBERING == 'F'); i < Mat->ia_[k + 1] - (HPDDM_NUMBERING == 'F'); ++i) {
          if (indicator[rankWorld * Mat->n_ + Mat->ja_[i] - (HPDDM_NUMBERING == 'F')] > 0.0) {
            locMat->ia_[g2l.at(k) + 1]++;
            locMat->ja_[nnz]  = g2l.at(Mat->ja_[i] - (HPDDM_NUMBERING == 'F')) + (HPDDM_NUMBERING == 'F');
            locMat->a_[nnz++] = Mat->a_[i];
          }
        }
      }
    std::partial_sum(locMat->ia_, locMat->ia_ + locMat->n_ + 1, locMat->ia_);
    delete[] indicator;

    n = Mat->n_;
    delete Mat;
    Mat = locMat;
  } else n = Mat->n_;
  f = new K[ndof];
  if (opt.prefix("rhs_filename").size()) {
    std::ifstream file(opt.prefix("rhs_filename"));
    std::string   line;
    unsigned int  i = 0, j = 0;
    K             val;
    while (std::getline(file, line) && (idx.empty() || j < ndof)) {
      if (i == 0 && j == 0) {
        std::istringstream iss(line);
        std::string        word;
        iss >> word;
        if (HPDDM::Option::Arg::integer(std::string(), word, false) && HPDDM::sto<int>(word) == n) std::getline(file, line);
      }
      if (idx.empty() || i++ == idx[j]) {
        std::istringstream iss(line);
        iss >> val;
        f[j++] = val;
      }
    }
    if (rankWorld == 0) {
      std::ifstream       empty("foobar.txt");
      HPDDM::MatrixCSR<K> A(empty);
      std::cout << A;
      A.nnz_ = 1;
      std::cout << A;
    }
  } else {
    std::random_device                                        rd;
    std::mt19937                                              gen(rd());
    std::uniform_real_distribution<HPDDM::underlying_type<K>> dis(0.0, 10.0);
    std::for_each(f, f + ndof, [&](K &x) { assign(gen, dis, x); });
  }
  sol = new K[ndof]();
}
