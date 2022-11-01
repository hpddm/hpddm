#! /bin/bash

#
#  This file is part of HPDDM.
#
#  Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
#       Date: 2022-11-01
#
#  Copyright (C) 2022-     Centre National de la Recherche Scientifique
#
#  HPDDM is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  HPDDM is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with HPDDM.  If not, see <http://www.gnu.org/licenses/>.
#

cat << EOF > src/ksp/ksp/tutorials/m.c
#include <petsc.h>

static char help[] = "Solves a linear system after having repartitioned its symmetric part.\n\n";

int main(int argc,char **args)
{
  Vec             b;
  Mat             A,T,perm;
  KSP             ksp;
  IS              is,rows;
  PetscBool       flg;
  PetscViewer     viewer;
  char            name[PETSC_MAX_PATH_LEN];
  MatPartitioning mpart;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-mat_name",name,sizeof(name),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Missing -mat_name");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer));
  PetscCall(MatLoad(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD,&mpart));
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&T));
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&perm));
  PetscCall(MatAXPY(perm,1.0,T,DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatPartitioningSetAdjacency(mpart,perm)); // partition A^T+A
  PetscCall(MatPartitioningSetFromOptions(mpart));
  PetscCall(MatPartitioningApply(mpart,&is));
  PetscCall(MatDestroy(&perm));
  PetscCall(MatDestroy(&T));
  PetscCall(MatPartitioningDestroy(&mpart));
  PetscCall(ISBuildTwoSided(is,NULL,&rows));
  PetscCall(ISDestroy(&is));
  PetscCall(MatCreateSubMatrix(A,rows,rows,MAT_INITIAL_MATRIX,&perm));
  PetscCall(ISDestroy(&rows));
  PetscCall(MatHeaderReplace(A,&perm)); // only keep the permuted matrix
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp)); // parse command-line options
  PetscCall(MatCreateVecs(A,NULL,&b)); // vector with a compatible dimension
  PetscCall(PetscOptionsGetString(NULL,NULL,"-rhs_name",name,sizeof(name),&flg));
  if (!flg) PetscCall(VecSetRandom(b,NULL)); // random right-hand side
  else {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer));
    PetscCall(VecLoad(b,viewer)); // right-hand side from file
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscCall(KSPSolve(ksp,b,b)); // in-place solve
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: hpddm double mumps datafilespath !complex !defined(PETSC_USE_64BIT_INDICES)
      nsize: 4
      args: -pc_type hpddm -pc_hpddm_levels_1_eps_nev 80 -pc_hpddm_block_splitting -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_gen_non_hermitian -pc_hpddm_levels_1_sub_pc_factor_mat_solver_type mumps -pc_hpddm_levels_1_sub_mat_mumps_icntl_24 1 -pc_hpddm_levels_1_st_share_sub_ksp {{true false}shared output} -pc_hpddm_coarse_mat_type baij -ksp_error_if_not_converged -ksp_pc_side right -ksp_max_it 20 -ksp_gmres_modifiedgramschmidt
      args: -mat_name \${DATAFILESPATH}/matrices/power
      output_file: output/ex77_preonly.out

TEST*/
EOF
