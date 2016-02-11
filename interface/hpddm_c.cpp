/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-10-29

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

#include <HPDDM.hpp>

extern "C" {
#include "HPDDM.h"
}

#ifdef GENERAL_CO
const char symCoarse = 'G';
#else
const char symCoarse = 'S';
#endif

typedef typename std::conditional<std::is_same<K, underlying_type>::value, K, std::complex<underlying_type>>::type cpp_type;

const HpddmOption* HpddmOptionGet() {
    return (const HpddmOption* const)&*HPDDM::Option::get();
}
int HpddmOptionParse(const HpddmOption* const option, int argc, char** argv, bool display) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.parse(argc, argv, display);
}
int HpddmOptionParseInt(const HpddmOption* const option, int argc, char** argv, char* str, char* desc) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.parse(argc, argv, false, { std::forward_as_tuple(str, desc, HPDDM::Option::Arg::integer) });
}
int HpddmOptionParseInts(const HpddmOption* const option, int argc, char** argv, int size, char* str[], char* desc[]) {
    std::vector<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>> pack;
    pack.reserve(size);
    for(int i = 0; i < size; ++i)
        pack.emplace_back(str[i], desc[i], HPDDM::Option::Arg::integer);
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.parse(argc, argv, false, pack);
}
int HpddmOptionParseArg(const HpddmOption* const option, int argc, char** argv, char* str, char* desc) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.parse(argc, argv, false, { std::forward_as_tuple(str, desc, HPDDM::Option::Arg::argument) });
}
int HpddmOptionParseArgs(const HpddmOption* const option, int argc, char** argv, int size, char* str[], char* desc[]) {
    std::vector<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>> pack;
    pack.reserve(size);
    for(int i = 0; i < size; ++i)
        pack.emplace_back(str[i], desc[i], HPDDM::Option::Arg::argument);
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.parse(argc, argv, false, pack);
}
bool HpddmOptionSet(const HpddmOption* const option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.set(str);
}
void HpddmOptionRemove(const HpddmOption* const option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    opt.remove(str);
}
double HpddmOptionVal(const HpddmOption* const option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.val(str);
}
double* HpddmOptionAddr(const HpddmOption* const option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return &(opt[str]);
}
double HpddmOptionApp(const HpddmOption* const option, char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    if(opt.app().find(str) != opt.app().cend())
        return opt.app()[str];
    else
        return 0;
}

HpddmMatrixCSR* HpddmMatrixCSRCreate(int n, int m, int nnz, K* const a, int* const ia, int* const ja, bool sym, bool takeOwnership) {
    return reinterpret_cast<HpddmMatrixCSR*>(new HPDDM::MatrixCSR<cpp_type>(n, m, nnz, reinterpret_cast<cpp_type*>(a), ia, ja, sym, takeOwnership));
}
void HpddmMatrixCSRDestroy(HpddmMatrixCSR* a) {
    reinterpret_cast<HPDDM::MatrixCSR<cpp_type>*>(a)->destroy(std::free);
    delete reinterpret_cast<HPDDM::MatrixCSR<cpp_type>*>(a);
}
void HpddmCsrmm(HpddmMatrixCSR* a, const K* const x, K* prod, int m) {
    HPDDM::MatrixCSR<cpp_type>* A = reinterpret_cast<HPDDM::MatrixCSR<cpp_type>*>(a);
    HPDDM::Wrapper<cpp_type>::csrmm(A->_sym, &(A->_n), &m, A->_a, A->_ia, A->_ja, reinterpret_cast<const cpp_type*>(x), reinterpret_cast<cpp_type*>(prod));
}

void HpddmSubdomainNumfact(HpddmSubdomain** S, HpddmMatrixCSR* Mat) {
    if(Mat) {
        if(*S == NULL)
            *S = reinterpret_cast<HpddmSubdomain*>(new SUBDOMAIN<cpp_type>());
        reinterpret_cast<SUBDOMAIN<cpp_type>*>(*S)->numfact(reinterpret_cast<HPDDM::MatrixCSR<cpp_type>*>(Mat));
    }
}
void HpddmSubdomainSolve(HpddmSubdomain* S, const K* const b, K* x, unsigned short n) {
    reinterpret_cast<SUBDOMAIN<cpp_type>*>(S)->solve(reinterpret_cast<const cpp_type*>(b), reinterpret_cast<cpp_type*>(x), n);
}
void HpddmSubdomainDestroy(HpddmSubdomain* S) {
    delete reinterpret_cast<SUBDOMAIN<cpp_type>*>(S);
}

void HpddmInitializeCoarseOperator(HpddmPreconditioner* A, unsigned short nu) {
    reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type>, cpp_type>*>(A)->initialize(nu);
}
void HpddmSetVectors(HpddmPreconditioner* A, K** v) {
    reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type>, cpp_type>*>(A)->setVectors(reinterpret_cast<cpp_type**>(v));
}
void HpddmDestroyVectors(HpddmPreconditioner* A) {
    reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type>, cpp_type>*>(A)->destroyVectors(std::free);
}
const MPI_Comm* HpddmGetCommunicator(HpddmPreconditioner* A) {
    return &(reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type>, cpp_type>*>(A)->getCommunicator());
}

HpddmSchwarz* HpddmSchwarzCreate(HpddmMatrixCSR* Mat, int neighbors, int* list, int* sizes, int** connectivity) {
    HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>* A = new HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>;
    A->Subdomain::initialize(reinterpret_cast<HPDDM::MatrixCSR<cpp_type>*>(Mat), neighbors, list, sizes, connectivity);
    return reinterpret_cast<HpddmSchwarz*>(A);
}
void HpddmSchwarzInitialize(HpddmSchwarz* A, underlying_type* d) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)->initialize(d);
}
HpddmPreconditioner* HpddmSchwarzPreconditioner(HpddmSchwarz* A) {
    return reinterpret_cast<HpddmPreconditioner*>(static_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type>, cpp_type>*>(reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)));
}
void HpddmSchwarzMultiplicityScaling(HpddmSchwarz* A, underlying_type* d) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)->multiplicityScaling(d);
}
void HpddmSchwarzScaledExchange(HpddmSchwarz* A, K* const x, unsigned short mu) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)->scaledExchange<true>(reinterpret_cast<cpp_type*>(x), mu);
}
void HpddmSchwarzCallNumfact(HpddmSchwarz* A) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)->callNumfact();
}
void HpddmSchwarzSolveGEVP(HpddmSchwarz* A, HpddmMatrixCSR* neumann, unsigned short* nu, underlying_type threshold) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)->solveGEVP<HPDDM::Arpack>(reinterpret_cast<HPDDM::MatrixCSR<cpp_type>*>(neumann), *nu, threshold);
}
void HpddmSchwarzBuildCoarseOperator(HpddmSchwarz* A, MPI_Comm comm) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)->buildTwo(comm);
}
void HpddmSchwarzComputeError(HpddmSchwarz* A, const K* const sol, const K* const f, underlying_type* storage, unsigned short mu) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)->computeError(reinterpret_cast<const cpp_type*>(sol), reinterpret_cast<const cpp_type*>(f), storage, mu);
}
void HpddmSchwarzDestroy(HpddmSchwarz* A) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)->destroyMatrix(std::free);
    delete reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A);
}

int HpddmSolve(HpddmSchwarz* A, const K* const b, K* const sol, int nu, const MPI_Comm* comm) {
    return HPDDM::IterativeMethod::solve(*(reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type>*>(A)), reinterpret_cast<const cpp_type*>(b), reinterpret_cast<cpp_type*>(sol), nu, *comm);
}

underlying_type nrm2(const int* n, const K* x, const int* inc) {
    return HPDDM::Blas<cpp_type>::nrm2(n, reinterpret_cast<const cpp_type*>(x), inc);
}
void axpy(const int* n, const K* const a, const K* const x, const int* incx, K* const y, const int* incy) {
    return HPDDM::Blas<cpp_type>::axpy(n, reinterpret_cast<const cpp_type*>(a), reinterpret_cast<const cpp_type*>(x), incx, reinterpret_cast<cpp_type*>(y), incy);
}
