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

#ifndef _HPDDM_H_
#define _HPDDM_H_

#ifndef MKL_Complex16
# define MKL_Complex16 void*
#endif
#ifndef MKL_Complex8
# define MKL_Complex8 void*
#endif

#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#ifndef MPI_VERSION
# include <mpi.h>
#endif
#include "../include/HPDDM_define.hpp"

#ifdef FORCE_SINGLE
#ifdef FORCE_COMPLEX
typedef float _Complex K;
typedef float underlying_type;
#ifndef GENERAL_CO
#define GENERAL_CO
#endif
#else
typedef float K;
typedef float underlying_type;
#endif
#else
#ifdef FORCE_COMPLEX
typedef double _Complex K;
typedef double underlying_type;
#ifndef GENERAL_CO
#define GENERAL_CO
#endif
#else
typedef double K;
typedef double underlying_type;
#endif
#endif

struct HpddmOption;
typedef struct HpddmOption HpddmOption;
const HpddmOption* HpddmOptionGet();
int HpddmOptionParse(const HpddmOption* const, int, char**, bool);
int HpddmOptionParseString(const HpddmOption* const, const char*);
int HpddmOptionParseInt(const HpddmOption* const, int, char**, char*, char*);
int HpddmOptionParseInts(const HpddmOption* const, int, char**, int, char*[], char*[]);
int HpddmOptionParseArg(const HpddmOption* const, int, char**, char*, char*);
int HpddmOptionParseArgs(const HpddmOption* const, int, char**, int, char*[], char*[]);
bool HpddmOptionSet(const HpddmOption* const, const char*);
void HpddmOptionRemove(const HpddmOption* const, const char*);
double HpddmOptionVal(const HpddmOption* const, const char*);
double* HpddmOptionAddr(const HpddmOption* const, const char*);
double HpddmOptionApp(const HpddmOption* const, const char*);

struct HpddmMatrixCSR;
typedef struct HpddmMatrixCSR HpddmMatrixCSR;
HpddmMatrixCSR* HpddmMatrixCSRCreate(int, int, int, K*, int*, int*, bool, bool);
void HpddmMatrixCSRDestroy(HpddmMatrixCSR*);
void HpddmCSRMM(HpddmMatrixCSR*, const K* const, K*, int);

struct HpddmSubdomain;
typedef struct HpddmSubdomain HpddmSubdomain;
void HpddmSubdomainNumfact(HpddmSubdomain**, HpddmMatrixCSR*);
void HpddmSubdomainSolve(HpddmSubdomain*, const K* const, K*, unsigned short);
void HpddmSubdomainDestroy(HpddmSubdomain* S);

struct HpddmPreconditioner;
typedef struct HpddmPreconditioner HpddmPreconditioner;
void HpddmInitializeCoarseOperator(HpddmPreconditioner*, unsigned short);
void HpddmSetVectors(HpddmPreconditioner*, K**);
void HpddmDestroyVectors(HpddmPreconditioner*);
const MPI_Comm* HpddmGetCommunicator(HpddmPreconditioner*);

struct HpddmSchwarz;
typedef struct HpddmSchwarz HpddmSchwarz;
HpddmSchwarz* HpddmSchwarzCreate(HpddmMatrixCSR*, int, int*, int*, int**);
void HpddmSchwarzInitialize(HpddmSchwarz*, underlying_type*);
HpddmPreconditioner* HpddmSchwarzPreconditioner(HpddmSchwarz*);
void HpddmSchwarzMultiplicityScaling(HpddmSchwarz*, underlying_type*);
void HpddmSchwarzExchange(HpddmSchwarz*, K* const, unsigned short);
void HpddmSchwarzCallNumfact(HpddmSchwarz*);
void HpddmSchwarzSolveGEVP(HpddmSchwarz*, HpddmMatrixCSR*);
void HpddmSchwarzBuildCoarseOperator(HpddmSchwarz*, MPI_Comm);
void HpddmSchwarzComputeResidual(HpddmSchwarz*, const K* const, const K* const, underlying_type*, unsigned short);
void HpddmSchwarzDestroy(HpddmSchwarz*);

int HpddmSolve(HpddmSchwarz*, const K* const, K* const, int, const MPI_Comm*);
struct HpddmCustomOperator;
typedef struct HpddmCustomOperator HpddmCustomOperator;
int HpddmCustomOperatorSolve(const HpddmCustomOperator* const, int, void (*)(const HpddmCustomOperator* const, const K*, K*, int), void (*)(const HpddmCustomOperator* const, const K*, K*, int), const K* const, K* const, int, const MPI_Comm*);

underlying_type nrm2(const int*, const K* const, const int*);
void axpy(const int*, const K* const, const K* const, const int*, K* const, const int*);

#if HPDDM_PETSC
#include <petscsys.h>
PetscErrorCode HpddmRegisterKSP();
#endif

#endif // _HPDDM_H_
