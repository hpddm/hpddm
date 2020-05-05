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

#ifndef _SCHWARZ_
#define _SCHWARZ_

#if HPDDM_MKL
#include <complex>
#define MKL_Complex16         std::complex<double>
#define MKL_Complex8          std::complex<float>
#define MKL_INT               int
#endif
#ifndef HPDDM_NUMBERING
#define HPDDM_NUMBERING       'C'
#endif
#ifdef PETSCSUB
#define HPDDM_BDD             0
#define HPDDM_FETI            0
#ifdef HPDDM_NUMBERING
#undef HPDDM_NUMBERING
#endif
#define HPDDM_NUMBERING       'C'
#endif
#include <HPDDM.hpp>
#include <random>
#include <list>

#ifndef PETSCSUB
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
#else
typedef PetscScalar K;
#ifdef PETSC_USE_COMPLEX
#ifndef GENERAL_CO
#define GENERAL_CO
#endif
#endif
#endif

#ifdef GENERAL_CO
const char symCoarse = 'G';
#else
const char symCoarse = 'S';
#endif

const HPDDM::underlying_type<K> pi = 3.141592653589793238463;

void generate(int, int, std::list<int>&, std::vector<std::vector<int>>&, int&, HPDDM::MatrixCSR<K>*&, HPDDM::MatrixCSR<K>*&, HPDDM::underlying_type<K>*&, K*&, K*&);

#endif // _SCHWARZ_
