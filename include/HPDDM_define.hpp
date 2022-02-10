/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2018-04-11

   Copyright (C) 2018-     Centre National de la Recherche Scientifique

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

#ifndef _HPDDM_DEFINE_
#define _HPDDM_DEFINE_

/* Constants: C-style preprocessor variables
 *
 *    HPDDM_VERSION       - Version of the framework.
 *    HPDDM_EPS           - Small positive number used internally for dropping values.
 *    HPDDM_PEN           - Large positive number used externally for penalization, e.g. for imposing Dirichlet boundary conditions.
 *    HPDDM_GRANULARITY   - Granularity for OpenMP scheduling.
 *    HPDDM_MPI           - If not set to zero, MPI is supposed to be activated during compilation and for running the library.
 *    HPDDM_MKL           - If not set to zero, Intel MKL is chosen as the linear algebra backend.
 *    HPDDM_NUMBERING     - 0- or 1-based indexing of user-supplied matrices.
 *    HPDDM_SCHWARZ       - Overlapping Schwarz methods enabled.
 *    HPDDM_FETI          - FETI methods enabled.
 *    HPDDM_BDD           - BDD methods enabled.
 *    HPDDM_DENSE         - Methods for dense matrices enabled. Users must provide their own matrix--vector products.
 *    HPDDM_PETSC         - PETSc interface enabled.
 *    HPDDM_SLEPC         - PETSc compiled with SLEPc.
 *    HPDDM_QR            - If not set to zero, pseudo-inverses of Schur complements are computed using dense QR decompositions (with pivoting if set to one, without pivoting otherwise).
 *    HPDDM_ICOLLECTIVE   - If possible, use nonblocking MPI collective operations.
 *    HPDDM_MIXED_PRECISION - Use mixed precision arithmetic for the assembly of coarse operators.
 *    HPDDM_INEXACT_COARSE_OPERATOR - Solve coarse systems using a Krylov method.
 *    HPDDM_LIBXSMM       - Block sparse matrices products are computed using LIBXSMM. */
#define HPDDM_VERSION                                   "2.2.0"
#define HPDDM_EPS                                       1.0e-12
#define HPDDM_PEN                                       1.0e+30
#define HPDDM_GRANULARITY                               50000
#if !defined(HPDDM_PETSC) && defined(PETSC_PCHPDDM_MAXLEVELS)
# define HPDDM_PETSC                                    1
#endif
#if defined(PETSC_HAVE_MKL_LIBS) && !defined(HPDDM_MKL)
# define HPDDM_MKL                                      1
#endif
#if defined(HPDDM_PETSC) && HPDDM_PETSC
# ifndef HPDDM_NUMBERING
#  define HPDDM_NUMBERING                              'C'
# endif
# define HPDDM_SCHWARZ                                  0
# define HPDDM_BDD                                      0
# define HPDDM_FETI                                     0
# if defined(PETSCHPDDM_H)
#  define HPDDM_INEXACT_COARSE_OPERATOR                 1
# endif
#elif !defined(HPDDM_PETSC)
# define HPDDM_PETSC                                    0
#endif
#ifndef HPDDM_NUMBERING
# if HPDDM_PETSC || HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
#  pragma message("The numbering of user-supplied matrices has not been set, assuming 0-based indexing")
# endif
# define HPDDM_NUMBERING                               'C'
#elif defined(__cplusplus)
static_assert(HPDDM_NUMBERING == 'C' || HPDDM_NUMBERING == 'F', "Unknown numbering");
#endif
#ifndef HPDDM_MPI
# define HPDDM_MPI                                      1
#elif !HPDDM_MPI && defined(MPI_VERSION)
# pragma message("You cannot deactivate MPI support and still include MPI headers")
# undef HPDDM_MPI
# define HPDDM_MPI                                      1
#endif
#ifndef HPDDM_MKL
# ifdef INTEL_MKL_VERSION
#  define HPDDM_MKL                                     1
# else
#  define HPDDM_MKL                                     0
# endif
#endif
#ifndef HPDDM_SCHWARZ
# define HPDDM_SCHWARZ                                  1
#endif
#ifndef HPDDM_FETI
# define HPDDM_FETI                                     1
#endif
#ifndef HPDDM_BDD
# define HPDDM_BDD                                      1
#endif
#ifndef HPDDM_DENSE
# define HPDDM_DENSE                                    0
#elif HPDDM_DENSE
# if defined(HPDDM_INEXACT_COARSE_OPERATOR) && HPDDM_INEXACT_COARSE_OPERATOR
#  undef HPDDM_INEXACT_COARSE_OPERATOR
#  pragma message("HPDDM_DENSE and HPDDM_INEXACT_COARSE_OPERATOR are mutually exclusive")
# endif
# if defined(HPDDM_SCHWARZ) && !HPDDM_SCHWARZ
#  undef HPDDM_SCHWARZ
# endif
# ifndef HPDDM_SCHWARZ
#  define HPDDM_SCHWARZ                                 1
# endif
#endif
#ifndef HPDDM_SLEPC
# if (defined(SLEPCVERSION_H) || (HPDDM_PETSC && defined(PETSC_HAVE_SLEPC))) && defined(_PCIMPL_H)
#  define HPDDM_SLEPC                                   1
# else
#  define HPDDM_SLEPC                                   0
#endif
#endif
#define HPDDM_QR                                        2
#ifndef HPDDM_ICOLLECTIVE
# define HPDDM_ICOLLECTIVE                              0
#endif
#ifndef HPDDM_MIXED_PRECISION
# define HPDDM_MIXED_PRECISION                          0
#endif
#ifndef HPDDM_INEXACT_COARSE_OPERATOR
# define HPDDM_INEXACT_COARSE_OPERATOR                  0
#endif
#ifndef HPDDM_LIBXSMM
# define HPDDM_LIBXSMM                                  0
#endif

#define HPDDM_COMPUTE_RESIDUAL_L2                       0
#define HPDDM_COMPUTE_RESIDUAL_L1                       1
#define HPDDM_COMPUTE_RESIDUAL_LINFTY                   2

#define HPDDM_ORTHOGONALIZATION_CGS                     0
#define HPDDM_ORTHOGONALIZATION_MGS                     1

#define HPDDM_KRYLOV_METHOD_GMRES                       0
#define HPDDM_KRYLOV_METHOD_BGMRES                      1
#define HPDDM_KRYLOV_METHOD_CG                          2
#define HPDDM_KRYLOV_METHOD_BCG                         3
#define HPDDM_KRYLOV_METHOD_GCRODR                      4
#define HPDDM_KRYLOV_METHOD_BGCRODR                     5
#define HPDDM_KRYLOV_METHOD_BFBCG                       6
#define HPDDM_KRYLOV_METHOD_RICHARDSON                  7
#define HPDDM_KRYLOV_METHOD_NONE                        8

#define HPDDM_VARIANT_LEFT                              0
#define HPDDM_VARIANT_RIGHT                             1
#define HPDDM_VARIANT_FLEXIBLE                          2

#define HPDDM_QR_CHOLQR                                 0
#define HPDDM_QR_CGS                                    1
#define HPDDM_QR_MGS                                    2

#define HPDDM_RECYCLE_STRATEGY_A                        0
#define HPDDM_RECYCLE_STRATEGY_B                        1

#define HPDDM_RECYCLE_TARGET_SM                         0
#define HPDDM_RECYCLE_TARGET_LM                         1
#define HPDDM_RECYCLE_TARGET_SR                         2
#define HPDDM_RECYCLE_TARGET_LR                         3
#define HPDDM_RECYCLE_TARGET_SI                         4
#define HPDDM_RECYCLE_TARGET_LI                         5

#define HPDDM_GENEO_FORCE_UNIFORMITY_MIN                0
#define HPDDM_GENEO_FORCE_UNIFORMITY_MAX                1

#define HPDDM_DISTRIBUTION_CENTRALIZED                  0
#define HPDDM_DISTRIBUTION_SOL                          1

#define HPDDM_SCHWARZ_METHOD_RAS                        0
#define HPDDM_SCHWARZ_METHOD_ORAS                       1
#define HPDDM_SCHWARZ_METHOD_SORAS                      2
#define HPDDM_SCHWARZ_METHOD_ASM                        3
#define HPDDM_SCHWARZ_METHOD_OSM                        4
#define HPDDM_SCHWARZ_METHOD_NONE                       5

#define HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED        0
#define HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE        1
#define HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED        2

#define HPDDM_SUBSTRUCTURING_SCALING_MULTIPLICITY       0
#define HPDDM_SUBSTRUCTURING_SCALING_STIFFNESS          1
#define HPDDM_SUBSTRUCTURING_SCALING_COEFFICIENT        2

#define HPDDM_HYPRE_SOLVER_FGMRES                       0
#define HPDDM_HYPRE_SOLVER_PCG                          1
#define HPDDM_HYPRE_SOLVER_AMG                          2

#endif // _HPDDM_DEFINE_
