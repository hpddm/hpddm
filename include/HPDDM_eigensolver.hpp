/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2012-12-15

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

#ifndef HPDDM_EIGENSOLVER_HPP_
#define HPDDM_EIGENSOLVER_HPP_

#include <random>

#include "HPDDM_iterative.hpp"

#if !HPDDM_PETSC
namespace HPDDM
{
/* Class: Eigensolver
 *
 *  A base class used to interface eigenvalue problem solvers such as <Arpack> or <Lapack>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template <class K>
class Eigensolver {
private:
  template <class T, typename std::enable_if<Wrapper<T>::is_complex>::type * = nullptr>
  void imag(T &t, underlying_type<T> v)
  {
    t.imag(v);
  }
  template <class T, typename std::enable_if<!Wrapper<T>::is_complex>::type * = nullptr>
  void imag(T &, underlying_type<T>)
  {
  }

protected:
  /* Variable: tol
         *  Relative tolerance of the eigenvalue problem solver. */
  underlying_type<K> tol_;
  /* Variable: threshold
         *  Threshold criterion. */
  underlying_type<K> threshold_;
  /* Variable: n
         *  Number of rows of the eigenvalue problem. */
  int n_;

public:
  /* Variable: nu
         *  Number of desired eigenvalues. */
  int nu_;
  explicit Eigensolver(int n) : tol_(), threshold_(), n_(n), nu_() { }
  Eigensolver(int n, int nu) : tol_(Option::get()->val("eigensolver_tol", 1.0e-6)), threshold_(), n_(n), nu_(std::min(nu, n)) { }
  Eigensolver(underlying_type<K> threshold, int n, int nu) : tol_(threshold > 0.0 ? HPDDM_EPS : Option::get()->val("eigensolver_tol", 1.0e-6)), threshold_(threshold), n_(n), nu_(std::min(nu, n)) { }
  Eigensolver(underlying_type<K> tol, underlying_type<K> threshold, int n, int nu) : tol_(threshold > 0.0 ? HPDDM_EPS : tol), threshold_(threshold), n_(n), nu_(std::min(nu, n)) { }
  std::string dump(const K *const eigenvalues, const K *const *const eigenvectors, const MPI_Comm &communicator, std::ios_base::openmode mode = std::ios_base::out) const
  {
    int rankWorld;
    MPI_Comm_rank(communicator, &rankWorld);
    const Option &opt      = *Option::get();
    std::string   filename = opt.prefix("dump_eigenvectors", true);
    if (filename.size() == 0) filename = opt.prefix("dump_eigenvectors_" + to_string(rankWorld), true);
    if (filename.size() != 0) {
      int sizeWorld;
      MPI_Comm_size(communicator, &sizeWorld);
      std::string   name = filename + "_" + to_string(rankWorld) + "_" + to_string(sizeWorld) + ".txt";
      std::ofstream output(name, mode);
      output << std::scientific;
      for (unsigned short col = 0; col < nu_; col += 5) {
        for (unsigned short i = col; i < std::min(col + 5, nu_); ++i) output << std::setw(13) << i + 1 << "\t";
        output << "\n";
        for (unsigned short i = col; i < std::min(col + 5, nu_); ++i) output << std::setw(13) << eigenvalues[i] << "\t";
        output << "\n\n";
        for (unsigned int j = 0; j < n_; ++j) {
          for (unsigned short i = col; i < std::min(col + 5, nu_); ++i) output << std::setw(13) << eigenvectors[i][j] << "\t";
          output << "\n";
        }
        output << "\n";
      }
      return name;
    } else return std::string();
  }
  /* Function: selectNu
         *
         *  Computes a uniform threshold criterion.
         *
         * Parameters:
         *    eigenvalues   - Input array used to store eigenvalues in ascending order.
         *    communicator  - MPI communicator (usually <Subdomain::communicator>) on which the criterion <Eigensolver::nu> has to be uniformized. */
  template <class T, bool min = false>
  void selectNu(const T *const eigenvalues, K **&eigenvectors, const MPI_Comm &communicator, unsigned short m = 0)
  {
    static_assert(std::is_same<T, K>::value || std::is_same<T, underlying_type<K>>::value, "Wrong types");
    const Option &opt = *Option::get();
    unsigned short nev = nu_ ? std::min(static_cast<int>(std::distance(eigenvalues, std::upper_bound(eigenvalues + 1, eigenvalues + nu_, threshold_, [](const T &lhs, const T &rhs) { return std::real(lhs) < std::real(rhs); }))), nu_) : (min ? std::numeric_limits<unsigned short>::max() : 0);
    switch (opt.val<char>("geneo_force_uniformity")) {
    case HPDDM_GENEO_FORCE_UNIFORMITY_MIN:
      if (!min) MPI_Allreduce(MPI_IN_PLACE, &nev, 1, MPI_UNSIGNED_SHORT, MPI_MIN, communicator);
      else {
        unsigned short r[2] = {nev, static_cast<unsigned short>(std::numeric_limits<unsigned short>::max() - m)};
        MPI_Allreduce(MPI_IN_PLACE, r, 2, MPI_UNSIGNED_SHORT, MPI_MIN, communicator);
        nev = std::max(r[0], static_cast<unsigned short>(std::numeric_limits<unsigned short>::max() - r[1]));
      }
      break;
    case HPDDM_GENEO_FORCE_UNIFORMITY_MAX:
      if (!nu_) nev = 0;
      MPI_Allreduce(MPI_IN_PLACE, &nev, 1, MPI_UNSIGNED_SHORT, MPI_MAX, communicator);
      if (nev > nu_) {
        K **basis = new K *[nev];
        *basis    = new K[nev * n_];
        for (unsigned short i = 1; i < nev; ++i) basis[i] = *basis + i * n_;
        std::random_device                                 rd;
        std::default_random_engine                         generator(rd());
        std::uniform_real_distribution<underlying_type<K>> uniform;
        if (eigenvectors && *eigenvectors) {
          std::copy_n(*eigenvectors, nu_ * n_, *basis);
          std::pair<K *, K *> result = std::minmax_element(*eigenvectors, *eigenvectors + nu_ * n_, [](const K &lhs, const K &rhs) { return std::real(lhs) < std::real(rhs); });
          uniform                    = std::uniform_real_distribution<underlying_type<K>>(std::real(*result.first), std::real(*result.second));
          delete[] *eigenvectors;
        } else uniform = std::uniform_real_distribution<underlying_type<K>>(0.0, 1.0);
        delete[] eigenvectors;
        std::for_each(basis[nu_], basis[nev - 1] + n_, [&](K &v) { v = uniform(generator); });
        if (Wrapper<K>::is_complex) std::for_each(basis[nu_], basis[nev - 1] + n_, [&](K &v) { imag(v, uniform(generator)); });
        eigenvectors = basis;
        if (nu_ == 0) {
          underlying_type<K> nrm = Blas<K>::nrm2(&n_, *basis, &i__1);
          std::for_each(*basis, *basis + n_, [&](K &v) { v /= nrm; });
          nu_ = 1;
        }
        const char id = opt.val<char>("orthogonalization", HPDDM_ORTHOGONALIZATION_CGS);
        for (unsigned short i = nu_; i < nev; ++i) IterativeMethod::orthogonalization(id, n_, i - 1, *basis, basis[i]);
      }
      nu_ = nev;
      break;
    }
    nu_ = std::min(nu_, static_cast<int>(nev));
    if (!nu_ && eigenvectors) {
      delete[] *eigenvectors;
      delete[] eigenvectors;
      eigenvectors  = new K *[1];
      *eigenvectors = nullptr;
    }
  }
  /* Function: getTol
         *  Returns the value of <Eigensolver::tol>. */
  underlying_type<K> getTol() const { return tol_; }
};
} // namespace HPDDM
#endif
#endif // HPDDM_EIGENSOLVER_HPP_
