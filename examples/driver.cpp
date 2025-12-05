/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2016-01-20

   Copyright (C) 2016-     Centre National de la Recherche Scientifique

   Note:      Reference MATLAB implementation available at
                                    http://www.sandia.gov/~mlparks/GCRODR.zip
              Sequence of linear systems constructed by
                                       Philippe H. Geubelle and Spandan Maiti

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

#undef HPDDM_NUMBERING
#undef HPDDM_SCHWARZ
#undef HPDDM_FETI
#undef HPDDM_BDD

#define HPDDM_NUMBERING 'F'
#define HPDDM_SCHWARZ   0
#define HPDDM_FETI      0
#define HPDDM_BDD       0

#include "HPDDM.hpp"

#ifdef FORCE_COMPLEX
typedef std::complex<double> K;
#else
typedef double K;
#endif

struct CustomOperator : HPDDM::CustomOperator<HPDDM::MatrixCSR<K>, K> {
  using HPDDM::CustomOperator<HPDDM::MatrixCSR<K>, K>::CustomOperator;
  template <bool>
  int apply(const K *const in, K *const out, const unsigned short &mu = 1, K * = nullptr, const unsigned short & = 0) const
  {
    HPDDM::Option &opt = *HPDDM::Option::get();
    if (opt.app()["diagonal_scaling"] == 0) std::copy_n(in, mu * n_, out);
    else {
      const HPDDM::MatrixCSR<K> *const A = getMatrix();
      for (int i = 0; i < n_; ++i) {
        int mid = std::distance(A->ja_, std::upper_bound(A->ja_ + A->ia_[i] - A->ia_[0], A->ja_ + A->ia_[i + 1] - A->ia_[0], i + A->ia_[0])) - 1;
        for (unsigned short nu = 0; nu < mu; ++nu) out[nu * n_ + i] = in[nu * n_ + i] / A->a_[mid];
      }
    }
    return 0;
  }
};

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  HPDDM::Option &opt   = *HPDDM::Option::get();
  opt["gmres_restart"] = 40;
  opt["recycle"]       = 20;
  opt["max_it"]        = 1000;
  opt["tol"]           = 1.0e-10;
  opt.parse(argc, argv, 1,
            {
              std::forward_as_tuple("mu=<1>", "Number of right-hand sides.", HPDDM::Option::Arg::integer),                                       //
              std::forward_as_tuple("diagonal_scaling=<0>", "Use the diagonal of the matrix as a preconditioner.", HPDDM::Option::Arg::integer), //
              std::forward_as_tuple("path=<./examples/data>", "Relative path to the different .txt files.", HPDDM::Option::Arg::argument)        //
            });
  unsigned short no = 0;
  std::ifstream  t(opt.prefix("path") + "/40" + HPDDM::to_string(no++) + ".txt");
  if (!t.good()) {
    std::cerr << "Please specity a correct -path=<./examples/data>" << std::endl;
    return 1;
  }
  int                  status = 0;
  unsigned int         it     = 0;
  CustomOperator      *A      = nullptr;
  HPDDM::MatrixCSR<K> *Mat    = nullptr;
  do {
    K  *rhs = nullptr;
    int mu  = opt.app()["mu"];
    {
      t.seekg(0, std::ios::end);
      size_t      size = t.tellg();
      std::string buffer(size, ' ');
      t.seekg(0);
      t.read(&buffer[0], size);
      std::vector<std::string> s;
      s.reserve(size);
      std::stringstream ss(buffer);
      std::string       item;
      while (std::getline(ss, item, ' ')) {
        item.erase(std::remove(item.begin(), item.end(), '\n'), item.end());
        if (item.size() > 0) s.emplace_back(item);
      }
      std::cout << "Solving system #" << no << ": " << s[0] << "x" << s[0] << ", " << s[1] << " nnz" << std::endl;
      int n   = HPDDM::template sto<int>(s[0]);
      int nnz = HPDDM::template sto<int>(s[1]);
      Mat     = new HPDDM::MatrixCSR<K>(n, n, nnz, false);
      for (unsigned int i = 0; i < nnz; ++i) Mat->a_[i] = HPDDM::template sto<double>(s[3 + i]);
      for (unsigned int i = 0; i < nnz; ++i) Mat->ja_[i] = HPDDM::template sto<int>(s[nnz + 3 + i]);
      for (unsigned int i = 0; i < n + 1; ++i) Mat->ia_[i] = HPDDM::template sto<int>(s[2 * nnz + 3 + i]);
      rhs = new K[mu * n];
      for (unsigned int i = 0; i < n; ++i) rhs[i] = HPDDM::template sto<double>(s[2 * nnz + n + 4 + i]);
      t.close();
    }
    if (A) A->setMatrix(Mat);
    else {
      A = new CustomOperator(Mat);
    }
    K *x = new K[mu * Mat->n_]();
    if (mu > 1)
      for (unsigned short nu = 1; nu < mu; ++nu) std::copy_n(rhs, Mat->n_, rhs + nu * Mat->n_);
    it += HPDDM::IterativeMethod::solve(*A, rhs, x, mu, MPI_COMM_SELF);
    HPDDM::underlying_type<K> *nrmb = new HPDDM::underlying_type<K>[2 * mu];
    int                        n    = Mat->n_;
    for (unsigned short nu = 0; nu < mu; ++nu) nrmb[nu] = HPDDM::Blas<K>::nrm2(&n, rhs + nu * n, &(HPDDM::i__1));
    K *tmp = new K[mu * n];
    HPDDM::Wrapper<K>::csrmm(Mat->sym_, &n, &mu, Mat->a_, Mat->ia_, Mat->ja_, x, tmp);
    n *= mu;
    HPDDM::Blas<K>::axpy(&n, &(HPDDM::Wrapper<K>::d__2), rhs, &(HPDDM::i__1), tmp, &(HPDDM::i__1));
    n /= mu;
    HPDDM::underlying_type<K> *nrmAx = nrmb + mu;
    for (unsigned short nu = 0; nu < mu; ++nu) {
      nrmAx[nu] = HPDDM::Blas<K>::nrm2(&n, tmp + nu * n, &(HPDDM::i__1));
      if (nu == 0) std::cout << " --- residual = ";
      else std::cout << "                ";
      std::cout << std::scientific << nrmAx[nu] << " / " << nrmb[nu];
      if (mu > 1) std::cout << " (rhs #" << nu + 1 << ")";
      std::cout << std::endl;
      if (nrmAx[nu] / nrmb[nu] > 1.0e-7) status = 1;
    }
    delete[] tmp;
    delete[] nrmb;
    delete[] x;
    delete[] rhs;
  } while (t.open(opt.prefix("path") + "/40" + HPDDM::to_string(no++) + ".txt"), t.good());
  delete Mat;
  delete A;
  std::cout << "Total number of iterations: " << it << std::endl;
  MPI_Finalize();
  if (status == 0 && opt.any_of("krylov_method", {HPDDM_KRYLOV_METHOD_GCRODR, HPDDM_KRYLOV_METHOD_BGCRODR})) {
    const char variant = (!opt.set("variant") ? 'R' : opt["variant"] == HPDDM_VARIANT_LEFT ? 'L' : 'F');
    if (opt.app()["diagonal_scaling"] == 0) status = !(it > 2346 && it < 2366);
    else if (variant == 'L') status = !(it > 2052 && it < 2072);
    else if (variant == 'R') status = !(it > 2055 && it < 2075);
  }
  return status;
}
