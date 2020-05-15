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

#if defined(SUBDOMAIN) && defined(COARSEOPERATOR)
#ifdef GENERAL_CO
const char symCoarse = 'G';
#else
const char symCoarse = 'S';
#endif
#endif

template<class T>
using cpp_type = typename std::conditional<std::is_same<T, underlying_type>::value, T, std::complex<underlying_type>>::type;

template<class Operator, class K>
struct CustomOperator : public HPDDM::EmptyOperator<cpp_type<K>> {
    const Operator* const                                  _A;
    int      (*_mv)(const Operator* const, const K*, K*, int);
    int (*_precond)(const Operator* const, const K*, K*, int);
    CustomOperator(const Operator* const A, int n, int (*mv)(const Operator* const, const K*, K*, int), int (*precond)(const Operator* const, const K*, K*, int)) : HPDDM::EmptyOperator<cpp_type<K>>(n), _A(A), _mv(mv), _precond(precond) { }
    int GMV(const cpp_type<K>* const in, cpp_type<K>* const out, const int& mu = 1) const {
        return _mv(_A, reinterpret_cast<const K*>(in), reinterpret_cast<K*>(out), mu);
    }
    template<bool>
    int apply(const cpp_type<K>* const in, cpp_type<K>* const out, const unsigned short& mu = 1, cpp_type<K>* = nullptr, const unsigned short& = 0) const {
        return _precond(_A, reinterpret_cast<const K*>(in), reinterpret_cast<K*>(out), mu);
    }
};

const HpddmOption* HpddmOptionGet() {
    return (const HpddmOption*)HPDDM::Option::get().get();
}
int HpddmOptionParse(const HpddmOption* const option, int argc, char** argv, bool display) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.parse(argc, argv, display);
}
int HpddmOptionParseString(const HpddmOption* const option, const char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.parse(str, false);
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
int HpddmOptionParseArgs(const HpddmOption* const option, int argc, char** argv, int size, char* str[], char* desc[]) {
    std::vector<std::tuple<std::string, std::string, std::function<bool(const std::string&, const std::string&, bool)>>> pack;
    pack.reserve(size);
    for(int i = 0; i < size; ++i)
        pack.emplace_back(str[i], desc[i], HPDDM::Option::Arg::argument);
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.parse(argc, argv, false, pack);
}
bool HpddmOptionSet(const HpddmOption* const option, const char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.set(str);
}
void HpddmOptionRemove(const HpddmOption* const option, const char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    opt.remove(str);
}
double HpddmOptionVal(const HpddmOption* const option, const char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return opt.val(str);
}
double* HpddmOptionAddr(const HpddmOption* const option, const char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    return &opt[str];
}
double HpddmOptionApp(const HpddmOption* const option, const char* str) {
    HPDDM::Option& opt = *reinterpret_cast<HPDDM::Option*>((HpddmOption*)option);
    if(opt.app().find(str) != opt.app().cend())
        return opt.app()[str];
    else
        return 0;
}

HpddmMatrixCSR* HpddmMatrixCSRCreate(int n, int m, int nnz, K* const a, int* const ia, int* const ja, bool sym, bool takeOwnership) {
    return reinterpret_cast<HpddmMatrixCSR*>(new HPDDM::MatrixCSR<cpp_type<K>>(n, m, nnz, reinterpret_cast<cpp_type<K>*>(a), ia, ja, sym, takeOwnership));
}
void HpddmMatrixCSRDestroy(HpddmMatrixCSR* a) {
    reinterpret_cast<HPDDM::MatrixCSR<cpp_type<K>>*>(a)->destroy(std::free);
    delete reinterpret_cast<HPDDM::MatrixCSR<cpp_type<K>>*>(a);
}
void HpddmCSRMM(HpddmMatrixCSR* a, const K* const x, K* prod, int m) {
    HPDDM::MatrixCSR<cpp_type<K>>* A = reinterpret_cast<HPDDM::MatrixCSR<cpp_type<K>>*>(a);
    HPDDM::Wrapper<cpp_type<K>>::csrmm(A->_sym, &A->_n, &m, A->_a, A->_ia, A->_ja, reinterpret_cast<const cpp_type<K>*>(x), reinterpret_cast<cpp_type<K>*>(prod));
}

#if defined(SUBDOMAIN) && defined(COARSEOPERATOR)
void HpddmSubdomainNumfact(HpddmSubdomain** S, HpddmMatrixCSR* Mat) {
    if(Mat) {
        if(*S == NULL)
            *S = reinterpret_cast<HpddmSubdomain*>(new SUBDOMAIN<cpp_type<K>>());
        reinterpret_cast<SUBDOMAIN<cpp_type<K>>*>(*S)->numfact(reinterpret_cast<HPDDM::MatrixCSR<cpp_type<K>>*>(Mat));
    }
}
void HpddmSubdomainSolve(HpddmSubdomain* S, const K* const b, K* x, unsigned short n) {
    reinterpret_cast<SUBDOMAIN<cpp_type<K>>*>(S)->solve(reinterpret_cast<const cpp_type<K>*>(b), reinterpret_cast<cpp_type<K>*>(x), n);
}
void HpddmSubdomainDestroy(HpddmSubdomain* S) {
    delete reinterpret_cast<SUBDOMAIN<cpp_type<K>>*>(S);
}

void HpddmInitializeCoarseOperator(HpddmPreconditioner* A, unsigned short nu) {
    reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type<K>>, cpp_type<K>>*>(A)->initialize(nu);
}
void HpddmSetVectors(HpddmPreconditioner* A, K** v) {
    reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type<K>>, cpp_type<K>>*>(A)->setVectors(reinterpret_cast<cpp_type<K>**>(v));
}
void HpddmDestroyVectors(HpddmPreconditioner* A) {
    reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type<K>>, cpp_type<K>>*>(A)->destroyVectors(std::free);
}
const MPI_Comm* HpddmGetCommunicator(HpddmPreconditioner* A) {
    return &(reinterpret_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type<K>>, cpp_type<K>>*>(A)->getCommunicator());
}

HpddmSchwarz* HpddmSchwarzCreate(HpddmMatrixCSR* Mat, int neighbors, int* list, int* sizes, int** connectivity) {
    HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>* A = new HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>;
    A->Subdomain::initialize(reinterpret_cast<HPDDM::MatrixCSR<cpp_type<K>>*>(Mat), neighbors, list, sizes, connectivity);
    return reinterpret_cast<HpddmSchwarz*>(A);
}
void HpddmSchwarzInitialize(HpddmSchwarz* A, underlying_type* d) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->initialize(d);
}
HpddmPreconditioner* HpddmSchwarzPreconditioner(HpddmSchwarz* A) {
    return reinterpret_cast<HpddmPreconditioner*>(static_cast<HPDDM::Preconditioner<SUBDOMAIN, HPDDM::CoarseOperator<COARSEOPERATOR, symCoarse, cpp_type<K>>, cpp_type<K>>*>(reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)));
}
void HpddmSchwarzMultiplicityScaling(HpddmSchwarz* A, underlying_type* d) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->multiplicityScaling(d);
}
void HpddmSchwarzExchange(HpddmSchwarz* A, K* const x, unsigned short mu) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->exchange<true>(reinterpret_cast<cpp_type<K>*>(x), mu);
}
void HpddmSchwarzCallNumfact(HpddmSchwarz* A) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->callNumfact();
}
#ifdef EIGENSOLVER
void HpddmSchwarzSolveGEVP(HpddmSchwarz* A, HpddmMatrixCSR* neumann) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->solveGEVP<EIGENSOLVER>(reinterpret_cast<HPDDM::MatrixCSR<cpp_type<K>>*>(neumann));
}
#endif
void HpddmSchwarzBuildCoarseOperator(HpddmSchwarz* A, MPI_Comm comm) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->buildTwo(comm);
}
void HpddmSchwarzComputeResidual(HpddmSchwarz* A, const K* const sol, const K* const f, underlying_type* storage, unsigned short mu) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->computeResidual(reinterpret_cast<const cpp_type<K>*>(sol), reinterpret_cast<const cpp_type<K>*>(f), storage, mu);
}
void HpddmSchwarzDestroy(HpddmSchwarz* A) {
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->destroyMatrix(std::free);
    reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)->destroyVectors(std::free);
    delete reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A);
}

int HpddmSolve(HpddmSchwarz* A, const K* const b, K* const sol, int mu, const MPI_Comm* comm) {
    return HPDDM::IterativeMethod::solve(*(reinterpret_cast<HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symCoarse, cpp_type<K>>*>(A)), reinterpret_cast<const cpp_type<K>*>(b), reinterpret_cast<cpp_type<K>*>(sol), mu, *comm);
}
#endif
int HpddmCustomOperatorSolve(const HpddmCustomOperator* const A, int n, int (*mv)(const HpddmCustomOperator* const, const K*, K*, int), int (*precond)(const HpddmCustomOperator* const, const K*, K*, int), const K* const b, K* const sol, int mu, const MPI_Comm* comm) {
    return HPDDM::IterativeMethod::solve(CustomOperator<HpddmCustomOperator, K>(A, n, mv, precond), reinterpret_cast<const cpp_type<K>*>(b), reinterpret_cast<cpp_type<K>*>(sol), mu, *comm);
}

underlying_type nrm2(const int* n, const K* x, const int* inc) {
    return HPDDM::Blas<cpp_type<K>>::nrm2(n, reinterpret_cast<const cpp_type<K>*>(x), inc);
}
void axpy(const int* n, const K* const a, const K* const x, const int* incx, K* const y, const int* incy) {
    return HPDDM::Blas<cpp_type<K>>::axpy(n, reinterpret_cast<const cpp_type<K>*>(a), reinterpret_cast<const cpp_type<K>*>(x), incx, reinterpret_cast<cpp_type<K>*>(y), incy);
}

#if HPDDM_PETSC
PetscErrorCode HpddmRegisterKSP() {
    return KSPRegister(KSPHPDDM, HPDDM::KSPCreate_HPDDM);
}
#endif
