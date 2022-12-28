/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2016-04-29

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

#undef HPDDM_NUMBERING
#undef HPDDM_SCHWARZ
#undef HPDDM_FETI
#undef HPDDM_BDD

#define HPDDM_NUMBERING 'F'
#define HPDDM_SCHWARZ    0
#define HPDDM_FETI       0
#define HPDDM_BDD        0

#include <HPDDM.hpp>

#ifdef FORCE_SINGLE
#ifdef FORCE_COMPLEX
typedef std::complex<float> K;
#else
typedef float K;
#endif
#else
#ifdef FORCE_COMPLEX
typedef std::complex<double> K;
#else
typedef double K;
#endif
#endif

template<class K>
struct CustomOperator : public HPDDM::EmptyOperator<K> {
    void      (*mv_)(const int*, const K*, K*, const int*);
    void (*precond_)(const int*, const K*, K*, const int*);
    CustomOperator(int n, void (*mv)(const int*, const K*, K*, const int*), void (*precond)(const int*, const K*, K*, const int*)) : HPDDM::EmptyOperator<K>(n), mv_(mv), precond_(precond) { }
    int GMV(const K* const in, K* const out, const int& mu = 1) const {
        mv_(&(HPDDM::EmptyOperator<K>::n_), in, out, &mu);
        return 0;
    }
    template<bool>
    int apply(const K* const in, K* const out, const unsigned short& mu = 1, K* = nullptr, const unsigned short& = 0) const {
        int m = mu;
        precond_(&(HPDDM::EmptyOperator<K>::n_), in, out, &m);
        return 0;
    }
};

extern "C" {
int HPDDM_F77(hpddmparseconfig)(const char* str) {
    std::string cfg(str);
    std::shared_ptr<HPDDM::Option> opt = HPDDM::Option::get();
    std::ifstream stream(cfg);
    return opt->parse(stream);
}
void HPDDM_F77(hpddmoptionremove)(const char* str) {
    HPDDM::Option::get()->remove(str);
}
int HPDDM_F77(hpddmcustomoperatorsolve)(const int* n, void (**mv)(const int*, const K*, K*, const int*), void (**precond)(const int*, const K*, K*, const int*), const K* const b, K* const sol, const int* mu, const int* comm) {
    return HPDDM::IterativeMethod::solve(CustomOperator<K>(*n, *mv, *precond), b, sol, *mu, MPI_Comm_f2c(*comm));
}
}
