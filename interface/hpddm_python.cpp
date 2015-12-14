/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-11-06

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
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

#ifndef HPDDM_NUMBERING
#define HPDDM_NUMBERING       'C'
#endif
#include <Python.h>
#include <HPDDM.hpp>

#ifdef FORCE_SINGLE
#ifdef FORCE_COMPLEX
typedef std::complex<float> K;
#ifndef GENERAL_CO
#define GENERAL_CO
#endif
#else
typedef float K;
#endif
#else
#ifdef FORCE_COMPLEX
typedef std::complex<double> K;
#ifndef GENERAL_CO
#define GENERAL_CO
#endif
#else
typedef double K;
#endif
#endif

#ifdef GENERAL_CO
const char symCoarse = 'G';
#else
const char symCoarse = 'S';
#endif

extern "C" {
char numbering = HPDDM_NUMBERING;
unsigned short scalar = std::is_same<K, float>::value ? 0 : std::is_same<K, double>::value ? 1 : std::is_same<K, std::complex<float>>::value ? 2 : 3;

void* const optionGet() {
    return &*HPDDM::Option::get();
}
int optionParse(void* option, char* args, bool display) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>(option);
    std::string arg(args);
    return opt.parse(arg, display);
}
int optionParseInts(void* option, char* args, int size, char** str, char** desc) {
    std::vector<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>> pack;
    pack.reserve(size);
    for(int i = 0; i < size; ++i) {
        pack.emplace_back(str[i], desc[i], HPDDM::Option::Arg::integer);
    }
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>(option);
    std::string arg(args);
    return opt.parse(arg, false, pack);
}
int optionParseArgs(void* option, char* args, int size, char** str, char** desc) {
    std::vector<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>> pack;
    pack.reserve(size);
    for(int i = 0; i < size; ++i) {
        pack.emplace_back(str[i], desc[i], HPDDM::Option::Arg::argument);
    }
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>(option);
    std::string arg(args);
    return opt.parse(arg, false, pack);
}
bool optionSet(void* option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>(option);
    return opt.set(str);
}
void optionRemove(void* option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>(option);
    opt.remove(str);
}
double optionVal(void* option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>(option);
    return opt.val(str);
}
double* optionAddr(void* option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>(option);
    return &(opt[str]);
}
double optionApp(void* option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>(option);
    if(opt.app().find(str) != opt.app().cend())
        return opt.app()[str];
    else
        return 0;
}

void* matrixCSRCreate(int n, int m, int nnz, HPDDM::pod_type<K>* a, int* ia, int* ja, bool sym) {
    int* icopy = new int[n + 1];
    std::copy_n(ia, n + 1, icopy);
    int* jcopy = new int[nnz];
    std::copy_n(ja, nnz, jcopy);
    K* acopy = new K[nnz];
    std::copy_n(reinterpret_cast<K*>(a), nnz, acopy);
    return new HPDDM::MatrixCSR<K>(n, m, nnz, acopy, icopy, jcopy, sym, true);
}
void matrixCSRDestroy(void** Mat) {
    if(*Mat != NULL) {
        delete reinterpret_cast<HPDDM::MatrixCSR<K>*>(*Mat);
        *Mat = NULL;
    }
}
void csrmm(void* Mat, HPDDM::pod_type<K>* x, HPDDM::pod_type<K>* prod, int m) {
    HPDDM::MatrixCSR<K>* A = reinterpret_cast<HPDDM::MatrixCSR<K>*>(Mat);
    HPDDM::Wrapper<K>::csrmm(A->_sym, &(A->_n), &m, A->_a, A->_ia, A->_ja, reinterpret_cast<K*>(x), reinterpret_cast<K*>(prod));
}

void subdomainNumfact(void** S, void* Mat) {
    if(Mat) {
        if(*S == NULL)
            *S = new SUBDOMAIN<K>();
        reinterpret_cast<SUBDOMAIN<K>*>(*S)->numfact(reinterpret_cast<HPDDM::MatrixCSR<K>*>(Mat));
    }
}
void subdomainSolve(void* S, HPDDM::pod_type<K>* b, HPDDM::pod_type<K>* x, unsigned short n) {
    reinterpret_cast<SUBDOMAIN<K>*>(S)->solve(reinterpret_cast<K*>(b), reinterpret_cast<K*>(x), n);
}
void subdomainDestroy(void** S) {
    if(*S != NULL) {
        delete reinterpret_cast<SUBDOMAIN<K>*>(*S);
        *S = NULL;
    }
}

void initializeCoarseOperator(void* A, unsigned short nu) {
    reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, K>, K>*>(A)->initialize(nu);
}
void setVectors(void* A, int nu, HPDDM::pod_type<K>* v) {
    K** array = new K*[nu];
    int dof = reinterpret_cast<HPDDM::Subdomain<K>*>(A)->getDof();
    *array = new K[nu * dof];
    for(int i = 0; i < nu; ++i) {
        array[i] = *array + i * dof;
        std::copy_n(reinterpret_cast<K*>(v + i * dof), dof, array[i]);
    }
    reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, K>, K>*>(A)->setVectors(array);
}
const MPI_Comm* getCommunicator(void* A) {
    return &(reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, K>, K>*>(A)->getCommunicator());
}

void* schwarzCreate(void* Mat, PyObject* neighbors, PyObject* connectivity) {
    HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>* A = new HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>;
    std::vector<int> o;
    o.reserve(PyList_Size(neighbors));
    for(int i = 0; i < PyList_Size(neighbors); ++i)
        o.emplace_back(PyLong_AsLong(PyList_GET_ITEM(neighbors, i)));
    std::vector<std::vector<int>> r;
    r.reserve(o.size());
    for(int i = 0; i < PyList_Size(connectivity); ++i) {
        r.emplace_back(std::vector<int>());
        PyObject* neighbor = PyList_GET_ITEM(connectivity, i);
        r.back().reserve(PyList_Size(neighbor));
        for(int j = 0; j < PyList_Size(neighbor); ++j)
            r.back().emplace_back(PyLong_AsLong(PyList_GET_ITEM(neighbor, j)));
    }
    A->Subdomain::initialize(reinterpret_cast<HPDDM::MatrixCSR<K>*>(Mat), o, r);
    return A;
}
void schwarzInitialize(void* A, HPDDM::underlying_type<K>* d) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)->initialize(d);
}
void* schwarzPreconditioner(void* A) {
    return static_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, K>, K>*>(reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A));
}
void schwarzMultiplicityScaling(void* A, HPDDM::underlying_type<K>* d) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)->multiplicityScaling(d);
}
void schwarzScaledExchange(void* A, HPDDM::pod_type<K>* x, unsigned short mu) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)->scaledExchange<true>(reinterpret_cast<K*>(x), mu);
}
void schwarzCallNumfact(void* A) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)->callNumfact();
}
void schwarzSolveGEVP(void* A, void* neumann, unsigned short* nu, HPDDM::underlying_type<K> threshold) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)->solveGEVP<HPDDM::Arpack>(reinterpret_cast<HPDDM::MatrixCSR<K>*>(neumann), *nu, threshold);
}
void schwarzBuildCoarseOperator(void* A, MPI_Comm comm) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)->buildTwo(comm);
}
void schwarzComputeError(void* A, HPDDM::pod_type<K>* sol, HPDDM::pod_type<K>* f, HPDDM::underlying_type<K>* storage, unsigned short mu) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)->computeError(reinterpret_cast<K*>(sol), reinterpret_cast<K*>(f), storage, mu);
}
void schwarzDestroy(void** schwarz) {
    if(*schwarz != NULL) {
        delete reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(*schwarz);
        *schwarz = NULL;
    }
}

int CG(void* A, HPDDM::pod_type<K>* sol, HPDDM::pod_type<K>* f, MPI_Comm* comm) {
    return HPDDM::IterativeMethod::CG(*(reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)), reinterpret_cast<K*>(sol), reinterpret_cast<K*>(f), *comm);
}
int GMRES(void* A, HPDDM::pod_type<K>* sol, HPDDM::pod_type<K>* f, int mu, MPI_Comm* comm) {
    return HPDDM::IterativeMethod::GMRES(*(reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, K>*>(A)), reinterpret_cast<K*>(sol), reinterpret_cast<K*>(f), mu, *comm);
}
}
