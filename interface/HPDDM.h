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

#include <complex.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

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

#include "HPDDM.hpp"

struct HpddmOption;
typedef struct HpddmOption HpddmOption;
const HpddmOption* const HpddmOptionGet();
int HpddmOptionParse(const HpddmOption* const, int, char**, bool);
int HpddmOptionParseInt(const HpddmOption* const, int, char**, char*, char*);
int HpddmOptionParseInts(const HpddmOption* const, int, char**, int, char*[], char*[]);
int HpddmOptionParseArg(const HpddmOption* const, int, char**, char*, char*);
int HpddmOptionParseArgs(const HpddmOption* const, int, char**, int, char*[], char*[]);
bool HpddmOptionSet(const HpddmOption* const, char*);
void HpddmOptionRemove(const HpddmOption* const, char*);
double HpddmOptionVal(const HpddmOption* const, char*);
double* const HpddmOptionAddr(const HpddmOption* const, char*);
double HpddmOptionApp(const HpddmOption* const, char*);

struct HpddmMatrixCSR;
typedef struct HpddmMatrixCSR HpddmMatrixCSR;
HpddmMatrixCSR* HpddmMatrixCSRCreate(int, int, int, K*, int*, int*, bool, bool);
void HpddmMatrixCSRDestroy(HpddmMatrixCSR*);
void HpddmCsrmm(HpddmMatrixCSR*, const K* const, K*, int);

struct HpddmSubdomain;
typedef struct HpddmSubdomain HpddmSubdomain;
void HpddmSubdomainNumfact(HpddmSubdomain**, HpddmMatrixCSR*);
void HpddmSubdomainSolve(HpddmSubdomain*, const K* const, K*, unsigned short);
void HpddmSubdomainDestroy(HpddmSubdomain* S);

struct HpddmPreconditioner;
typedef struct HpddmPreconditioner HpddmPreconditioner;
void HpddmInitializeCoarseOperator(HpddmPreconditioner*, unsigned short);
void HpddmSetVectors(HpddmPreconditioner*, K**);
const MPI_Comm* HpddmGetCommunicator(HpddmPreconditioner*);

struct HpddmSchwarz;
typedef struct HpddmSchwarz HpddmSchwarz;
HpddmSchwarz* HpddmSchwarzCreate(HpddmMatrixCSR*, int, int*, int*, int**);
void HpddmSchwarzInitialize(HpddmSchwarz*, underlying_type*);
HpddmPreconditioner* HpddmSchwarzPreconditioner(HpddmSchwarz*);
void HpddmSchwarzMultiplicityScaling(HpddmSchwarz*, underlying_type*);
void HpddmSchwarzScaledExchange(HpddmSchwarz*, K* const, unsigned short);
void HpddmSchwarzCallNumfact(HpddmSchwarz*);
void HpddmSchwarzSolveGEVP(HpddmSchwarz*, HpddmMatrixCSR*, unsigned short*, underlying_type);
void HpddmSchwarzBuildCoarseOperator(HpddmSchwarz*, MPI_Comm);
void HpddmSchwarzComputeError(HpddmSchwarz*, const K* const, const K* const, underlying_type*, unsigned short);
void HpddmSchwarzDestroy(HpddmSchwarz*);

int HpddmCG(HpddmSchwarz*, K* const, const K* const, const MPI_Comm*);
int HpddmGMRES(HpddmSchwarz*, K* const, const K* const, int, const MPI_Comm*);

underlying_type nrm2(const int*, const K* const, const int*);
void axpy(const int*, const K* const, const K* const, const int*, K* const, const int*);

#endif // _HPDDM_H_
