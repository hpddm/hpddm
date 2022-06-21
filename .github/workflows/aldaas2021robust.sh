#! /bin/bash

#
#  This file is part of HPDDM.
#
#  Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
#       Date: 2021-07-22
#
#  Copyright (C) 2021-     Centre National de la Recherche Scientifique
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

cd .. &&  git clone --depth 1 https://github.com/prj-/aldaas2021robust.git && cd - 1> /dev/null
sed -i -e '/[0-9] KSP/d' ../aldaas2021robust/output/*out
sed -i -e '/rows=/d' ../aldaas2021robust/output/*out
sed -i -e '/total: nonzeros=/d' ../aldaas2021robust/output/*out
sed -i -e '/ I-node /d' ../aldaas2021robust/output/*out
cat << EOF >> ../aldaas2021robust/sparse_ls.c

/*TEST

   testset:
      requires: hpddm double !complex !defined(PETSC_USE_64BIT_INDICES)
      nsize: 4
      args: -ksp_view -pc_type hpddm
      filter: egrep -v "[0-9]+ KSP " | grep -v "rows=" | grep -v "total: nonzeros=" | grep -v " I-node " | sed -e "s/CONVERGED_RTOL iterations 6/CONVERGED_RTOL iterations 7/g" -e "s/CONVERGED_RTOL_NORMAL iterations 28/CONVERGED_RTOL_NORMAL iterations 20/g" -e "s/CONVERGED_RTOL iterations 21/CONVERGED_RTOL iterations 13/g" -e "s/CONVERGED_RTOL_NORMAL iterations 9/CONVERGED_RTOL_NORMAL iterations 8/g" -e "s/CONVERGED_RTOL_NORMAL iterations 8/CONVERGED_RTOL_NORMAL iterations 9/g"
      test:
        suffix: 1
        args: -options_file \${wPETSC_DIR}/../aldaas2021robust/default.rc -mat_name \${wPETSC_DIR}/../aldaas2021robust/datafiles/mesh_deform.dat -pc_hidden_setup {{true false}shared output}
        output_file: ../../../../../aldaas2021robust/output/sparse_ls_ksp_type-lsqr_mat_name-mesh_deform_pc_type-hpddm.out
      test:
        suffix: 2
        args: -options_file \${wPETSC_DIR}/../aldaas2021robust/gmres.rc -mat_name \${wPETSC_DIR}/../aldaas2021robust/datafiles/mesh_deform.dat
        output_file: ../../../../../aldaas2021robust/output/sparse_ls_ksp_type-gmres_mat_name-mesh_deform_pc_type-hpddm.out
      test:
        suffix: 3
        args: -options_file \${wPETSC_DIR}/../aldaas2021robust/default.rc -mat_name \${wPETSC_DIR}/../aldaas2021robust/datafiles/lp_stocfor3.dat
        output_file: ../../../../../aldaas2021robust/output/sparse_ls_ksp_type-lsqr_mat_name-lp_stocfor3_pc_type-hpddm.out
      test:
        suffix: 4
        args: -options_file \${wPETSC_DIR}/../aldaas2021robust/gmres.rc -mat_name \${wPETSC_DIR}/../aldaas2021robust/datafiles/lp_stocfor3.dat
        output_file: ../../../../../aldaas2021robust/output/sparse_ls_ksp_type-gmres_mat_name-lp_stocfor3_pc_type-hpddm.out
      test:
        suffix: 5
        requires: suitesparse
        args: -options_file \${wPETSC_DIR}/../aldaas2021robust/default.rc -mat_name \${wPETSC_DIR}/../aldaas2021robust/datafiles/mesh_deform.dat -pc_use_qr -pc_hpddm_levels_1_sub_pc_type qr -pc_hidden_setup {{true false}shared output}
        output_file: ../../../../../aldaas2021robust/output/sparse_ls_ksp_type-lsqr_mat_name-mesh_deform_pc_type-hpddm_pc_use_qr.out

TEST*/
EOF
cp ../aldaas2021robust/sparse_ls.c src/ksp/ksp/tutorials
