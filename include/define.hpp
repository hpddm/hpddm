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

#define HPDDM_COMPUTE_RESIDUAL_L2                               0
#define HPDDM_COMPUTE_RESIDUAL_L1                               1
#define HPDDM_COMPUTE_RESIDUAL_LINFTY                           2

#define HPDDM_ORTHOGONALIZATION_CGS                             0
#define HPDDM_ORTHOGONALIZATION_MGS                             1

#define HPDDM_KRYLOV_METHOD_GMRES                               0
#define HPDDM_KRYLOV_METHOD_BGMRES                              1
#define HPDDM_KRYLOV_METHOD_CG                                  2
#define HPDDM_KRYLOV_METHOD_BCG                                 3
#define HPDDM_KRYLOV_METHOD_GCRODR                              4
#define HPDDM_KRYLOV_METHOD_BGCRODR                             5
#define HPDDM_KRYLOV_METHOD_BFBCG                               6
#define HPDDM_KRYLOV_METHOD_RICHARDSON                          7
#define HPDDM_KRYLOV_METHOD_NONE                                8

#define HPDDM_VARIANT_LEFT                                      0
#define HPDDM_VARIANT_RIGHT                                     1
#define HPDDM_VARIANT_FLEXIBLE                                  2

#define HPDDM_QR_CHOLQR                                         0
#define HPDDM_QR_CGS                                            1
#define HPDDM_QR_MGS                                            2

#define HPDDM_RECYCLE_STRATEGY_A                                0
#define HPDDM_RECYCLE_STRATEGY_B                                1

#define HPDDM_RECYCLE_TARGET_SM                                 0
#define HPDDM_RECYCLE_TARGET_LM                                 1
#define HPDDM_RECYCLE_TARGET_SR                                 2
#define HPDDM_RECYCLE_TARGET_LR                                 3
#define HPDDM_RECYCLE_TARGET_SI                                 4
#define HPDDM_RECYCLE_TARGET_LI                                 5

#define HPDDM_GENEO_FORCE_UNIFORMITY_MIN                        0
#define HPDDM_GENEO_FORCE_UNIFORMITY_MAX                        1

#define HPDDM_DISTRIBUTION_CENTRALIZED                          0
#define HPDDM_DISTRIBUTION_SOL                                  1

#define HPDDM_SCHWARZ_METHOD_RAS                                0
#define HPDDM_SCHWARZ_METHOD_ORAS                               1
#define HPDDM_SCHWARZ_METHOD_SORAS                              2
#define HPDDM_SCHWARZ_METHOD_ASM                                3
#define HPDDM_SCHWARZ_METHOD_OSM                                4
#define HPDDM_SCHWARZ_METHOD_NONE                               5

#define HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED                0
#define HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE                1
#define HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED                2

#define HPDDM_SUBSTRUCTURING_SCALING_MULTIPLICITY               0
#define HPDDM_SUBSTRUCTURING_SCALING_STIFFNESS                  1
#define HPDDM_SUBSTRUCTURING_SCALING_COEFFICIENT                2

#define HPDDM_HYPRE_SOLVER_FGMRES                               0
#define HPDDM_HYPRE_SOLVER_PCG                                  1
#define HPDDM_HYPRE_SOLVER_AMG                                  2

#define KSPHPDDM                                                "hpddm"

#endif // _HPDDM_DEFINE_
