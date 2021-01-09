#
#  This file is part of HPDDM.
#
#  Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
#       Date: 2015-11-12
#
#  Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
#                2016-     Centre National de la Recherche Scientifique
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

-include Makefile.inc

TOP_DIR ?= ${PWD}
BIN_DIR = bin
LIB_DIR = lib
TRASH_DIR = .trash

$(shell mkdir -p ${TOP_DIR}/${BIN_DIR} > /dev/null)
$(shell mkdir -p ${TOP_DIR}/${LIB_DIR} > /dev/null)
$(shell mkdir -p ${TOP_DIR}/${TRASH_DIR} > /dev/null)

DEPFLAGS ?= -MT $@ -MMD -MP -MF ${TOP_DIR}/${TRASH_DIR}/$(notdir $(basename $@)).Td
POSTCOMPILE = mv -f ${TOP_DIR}/${TRASH_DIR}/$(notdir $(basename $@)).Td ${TOP_DIR}/${TRASH_DIR}/$(notdir $(basename $@)).d || true

INCS += -I./include -I./interface

ifdef SOLVER
    override HPDDMFLAGS += -DD${SOLVER}
endif
ifdef SUBSOLVER
    override HPDDMFLAGS += -D${SUBSOLVER}SUB
endif
ifdef EIGENSOLVER
    override HPDDMFLAGS += -DMU_${EIGENSOLVER}
endif

ifeq (${SUBSOLVER}, PETSC)
    INCS += ${PETSC_INCS}
    LIBS += ${PETSC_LIBS}
endif
ifeq (${SOLVER}, MUMPS)
    INCS += ${MUMPS_INCS}
    LIBS += ${MUMPS_LIBS}
else
    ifeq (${SUBSOLVER}, MUMPS)
        INCS += ${MUMPS_INCS}
        LIBS += ${MUMPS_LIBS}
    endif
endif
ifeq (${SOLVER}, MKL_PARDISO)
    INCS += ${MKL_PARDISO_INCS}
    LIBS += ${MKL_PARDISO_LIBS}
else
    ifeq (${SUBSOLVER}, MKL_PARDISO)
        INCS += ${MKL_PARDISO_INCS}
        LIBS += ${MKL_PARDISO_LIBS}
    endif
endif
ifeq (${SOLVER}, PASTIX)
    INCS += ${PASTIX_INCS}
    LIBS += ${PASTIX_LIBS}
else
    ifeq (${SUBSOLVER}, PASTIX)
        INCS += ${PASTIX_INCS}
        LIBS += ${PASTIX_LIBS}
    endif
endif
ifeq (${SOLVER}, HYPRE)
    INCS += ${HYPRE_INCS}
    LIBS += ${HYPRE_LIBS}
endif
ifeq (${SOLVER}, SUITESPARSE)
    INCS += ${SUITESPARSE_INCS}
    LIBS += ${SUITESPARSE_LIBS}
else
    ifeq (${SUBSOLVER}, SUITESPARSE)
        INCS += ${SUITESPARSE_INCS}
        LIBS += ${SUITESPARSE_LIBS}
    endif
endif
ifeq (${SUBSOLVER}, DISSECTION)
    INCS += ${DISSECTION_INCS}
    LIBS += ${DISSECTION_LIBS}
endif
ifeq (${SOLVER}, ELEMENTAL)
    INCS += ${EL_INCS}
    LIBS += ${EL_LIBS}
endif
ifeq (${EIGENSOLVER}, ARPACK)
    LIBS += ${ARPACK_LIBS}
endif
ifeq (${EIGENSOLVER}, SLEPC)
    INCS += ${SLEPC_INCS}
    LIBS += ${SLEPC_LIBS}
endif
ifdef LIBXSMM_INCS
    ifdef LIBXSMM_LIBS
        INCS += ${LIBXSMM_INCS}
        LIBS += ${LIBXSMM_LIBS}
        override HPDDMFLAGS += -DHPDDM_LIBXSMM=1
    endif
endif

LIBS += ${SCALAPACK_LIBS}
ifdef MKL_INCS
    INCS += ${MKL_INCS}
    LIBS += ${MKL_LIBS}
    override HPDDMFLAGS += -DHPDDM_MKL=1
else
    LIBS += ${BLAS_LIBS}
endif

ifeq (${OS}, Windows_NT)
    MAKE_OS = Windows
    EXTENSION_LIB = dll
    ifeq (,$(findstring ${TOP_DIR}/${LIB_DIR},${PATH}))
        PATH := ${PATH}:${TOP_DIR}/${LIB_DIR}
    endif
else
    UNAME_S := ${shell uname -s}
    ifeq (${UNAME_S}, Linux)
        MAKE_OS = Linux
        EXTENSION_LIB = so
        override LDFLAGS += -Wl,-rpath,${TOP_DIR}/${LIB_DIR}
    endif
    ifeq (${UNAME_S}, Darwin)
        MAKE_OS = macOS
        EXTENSION_LIB = dylib
    endif
endif

ifneq (, $(shell which mpixlc 2> /dev/null))
    SEP = : 
endif

LIST_COMPILATION ?= cpp c python fortran

.PHONY: all cpp c python fortran clean test test_cpp test_c test_python test_bin/schwarz_cpp test_bin/schwarz_c test_examples/schwarz.py test_bin/schwarz_cpp_custom_operator test_bin/schwarzFromFile_cpp test_bin/driver force

.PRECIOUS: ${TOP_DIR}/${BIN_DIR}/%_cpp.o ${TOP_DIR}/${BIN_DIR}/%_c.o ${TOP_DIR}/${BIN_DIR}/%.o

all: Makefile.inc ${LIST_COMPILATION}

${TOP_DIR}/${TRASH_DIR}/compiler_flags_cpp: force
	@echo "${CXXFLAGS} ${HPDDMFLAGS}" | cmp -s - $@ || echo "${CXXFLAGS} ${HPDDMFLAGS}" > $@

${TOP_DIR}/${TRASH_DIR}/compiler_flags_c: force
	@echo "${CFLAGS} ${HPDDMFLAGS}" | cmp -s - $@ || echo "${CFLAGS} ${HPDDMFLAGS}" > $@

cpp: ${TOP_DIR}/${BIN_DIR}/schwarz_cpp
c: ${TOP_DIR}/${BIN_DIR}/schwarz_c ${TOP_DIR}/${LIB_DIR}/libhpddm_c.${EXTENSION_LIB}
python: ${TOP_DIR}/${LIB_DIR}/libhpddm_python.${EXTENSION_LIB}

ifneq (,$(findstring gfortran,${OMPI_FC}${MPICH_F90}))
    F90MOD = -J
else
    F90MOD = -module
endif
fortran: ${TOP_DIR}/${LIB_DIR}/libhpddm_fortran.${EXTENSION_LIB} ${TOP_DIR}/${BIN_DIR}/custom_operator_fortran

Makefile.inc:
	@echo "No Makefile.inc found, please choose one from directory Make.inc"
	@echo ""
	@echo "Be sure to define at least ONE of the following variables, unless you are only building the C/Python/Fortran libraries:"
	@echo " - SUITESPARSE_INCS and SUITESPARSE_LIBS,"
	@echo " - MUMPS_INCS and MUMPS_LIBS,"
	@echo " - MKL_PARDISO_INCS and MKL_PARDISO_LIBS."
	@echo ""
	@echo "BLAS_LIBS must always be defined!"
	@echo ""
	@if [ -z ${MAKE_OS} ]; then \
		exit 1; \
	else \
		echo "${MAKE_OS} detected, trying to use Make.inc/Makefile.${MAKE_OS}"; \
		cp Make.inc/Makefile.${MAKE_OS} Makefile.inc; \
	fi

clean:
	rm -rf ${TOP_DIR}/${BIN_DIR} ${TOP_DIR}/${LIB_DIR} ${TOP_DIR}/${TRASH_DIR}
	find ${TOP_DIR} \( -name "*.o" -o -name "*.${EXTENSION_LIB}" -o -name "*.mod" -o -name "__pycache__" -type d -o -name "*.pyc" -o -name "*.gcov" \) -exec rm -rvf '{}' '+'

${TOP_DIR}/${BIN_DIR}/%_cpp.o: examples/%.cpp ${TOP_DIR}/${TRASH_DIR}/%.d ${TOP_DIR}/${TRASH_DIR}/compiler_flags_cpp
	@if test $(findstring FromFile, $<); then \
		echo ${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} ${METIS_INCS} -DHPDDM_FROMFILE -c $(subst schwarzFromFile,schwarz,$<) -o $@; \
		${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} ${METIS_INCS} -DHPDDM_FROMFILE -c $(subst schwarzFromFile,schwarz,$<) -o $@; \
	else \
		echo ${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -c $< -o $@; \
		${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -c $< -o $@; \
	fi;
	${POSTCOMPILE}

${TOP_DIR}/${BIN_DIR}/%_c.o: examples/%.c ${TOP_DIR}/${TRASH_DIR}/%.d ${TOP_DIR}/${TRASH_DIR}/compiler_flags_c
	${MPICC} ${DEPFLAGS} ${CFLAGS} ${HPDDMFLAGS} ${INCS} -c $< -o $@
	${POSTCOMPILE}

${TOP_DIR}/${BIN_DIR}/%.o: interface/%.cpp ${TOP_DIR}/${TRASH_DIR}/%.d ${TOP_DIR}/${TRASH_DIR}/compiler_flags_cpp
	${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -c $< -o $@
	${POSTCOMPILE}

${TOP_DIR}/${BIN_DIR}/%_cpp.o: benchmark/%.cpp ${TOP_DIR}/${TRASH_DIR}/%.d ${TOP_DIR}/${TRASH_DIR}/compiler_flags_cpp
	${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -c $< -o $@
	${POSTCOMPILE}

${TOP_DIR}/${BIN_DIR}/schwarz_c: ${TOP_DIR}/${BIN_DIR}/schwarz_c.o ${TOP_DIR}/${BIN_DIR}/generate_c.o ${TOP_DIR}/${BIN_DIR}/hpddm_c.o
	${MPICXX} $^ -o $@ ${LIBS}

${TOP_DIR}/${BIN_DIR}/schwarz%cpp: ${TOP_DIR}/${BIN_DIR}/schwarz%cpp.o ${TOP_DIR}/${BIN_DIR}/generate%cpp.o
	@if test $(findstring FromFile, $@); then \
		echo ${MPICXX} $^ -o $@ ${LIBS} ${METIS_LIBS}; \
		${MPICXX} $^ -o $@ ${LIBS} ${METIS_LIBS}; \
	else \
		echo ${MPICXX} $^ -o $@ ${LIBS}; \
		${MPICXX} $^ -o $@ ${LIBS}; \
	fi;

${TOP_DIR}/${BIN_DIR}/driver: ${TOP_DIR}/${BIN_DIR}/driver_cpp.o
	${MPICXX} $^ -o $@ ${LIBS}

${TOP_DIR}/${BIN_DIR}/local_%: ${TOP_DIR}/${BIN_DIR}/local_%_cpp.o
	${MPICXX} $^ -o $@ ${LIBS}

${TOP_DIR}/${BIN_DIR}/custom_operator_fortran.o: examples/custom_operator.f90
	${MPIF90} -c interface/HPDDM.f90 -o ${TOP_DIR}/${BIN_DIR}/HPDDM.o ${F90MOD} ${TOP_DIR}/${BIN_DIR}
	${MPIF90} -I${TOP_DIR}/${BIN_DIR} -c $< -o $@ ${F90MOD} ${TOP_DIR}/${BIN_DIR}

${TOP_DIR}/${BIN_DIR}/custom_operator_c.o: examples/custom_operator.c
	${MPICC} ${INCS} -c $< -o $@

${TOP_DIR}/${BIN_DIR}/custom_operator_%: ${TOP_DIR}/${BIN_DIR}/custom_operator_%.o ${TOP_DIR}/${LIB_DIR}/libhpddm_%.${EXTENSION_LIB}
	@if test $(findstring fortran,$@); then \
		CMD="${MPIF90} -I${TOP_DIR}/${BIN_DIR} -o $@ ${F90MOD} ${TOP_DIR}/${BIN_DIR} $@.o ${TOP_DIR}/${BIN_DIR}/HPDDM.o ${LDFLAGS} -L${TOP_DIR}/${LIB_DIR} -lhpddm_fortran ${LIBS}"; \
	else \
		CMD="${MPICC} -o $@ $@.o ${LDFLAGS} -L${TOP_DIR}/${LIB_DIR} -lhpddm_c ${LIBS}"; \
	fi; \
	echo "$${CMD}"; \
	$${CMD} || exit; \

test_bin/custom_operator_c: ${TOP_DIR}/${BIN_DIR}/custom_operator_c
	${MPIRUN} 4 ${TOP_DIR}/${BIN_DIR}/custom_operator_c -n 1000 -mu 4 -hpddm_krylov_method bcg
	${MPIRUN} 4 ${TOP_DIR}/${BIN_DIR}/custom_operator_c -n 1000 -mu 4 -hpddm_krylov_method cg -hpddm_variant flexible

benchmark/local_solver:
	@if [ -z ${MTX_FILE} ]; then \
		echo "MTX_FILE is not set, no matrix to benchmark ${TOP_DIR}/${BIN_DIR}/local_solver"; \
		exit 1; \
	fi
	@$@.py ${TOP_DIR}/${BIN_DIR}/local_solver ${MTX_FILE} ${BENCHMARKFLAGS}

${TOP_DIR}/${LIB_DIR}/lib%.${EXTENSION_LIB}: interface/%.cpp ${TOP_DIR}/${TRASH_DIR}/lib%.d ${TOP_DIR}/${TRASH_DIR}/compiler_flags_cpp
	@if [ "$<" = "interface/hpddm_python.cpp" ]; then \
		echo ${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} ${PYTHON_INCS} -shared $< -o $@ ${LIBS} ${PYTHON_LIBS}; \
		${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} ${PYTHON_INCS} -shared $< -o $@ ${LIBS} ${PYTHON_LIBS}; \
	else \
		echo ${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -shared $< -o $@ ${LIBS}; \
		${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -shared $< -o $@ ${LIBS}; \
	fi
	${POSTCOMPILE}

lib: $(addprefix ${TOP_DIR}/${LIB_DIR}/libhpddm_, $(addsuffix .${EXTENSION_LIB}, $(filter-out cpp, ${LIST_COMPILATION})))

test: all $(addprefix test_, ${LIST_COMPILATION})

test_cpp: ${TOP_DIR}/${BIN_DIR}/schwarz_cpp test_bin/schwarz_cpp test_bin/schwarz_cpp_custom_operator
test_c: ${TOP_DIR}/${BIN_DIR}/schwarz_c test_bin/schwarz_c
test_python: python test_examples/schwarz.py
test_fortran: examples/hpddm_f90.cfg ${TOP_DIR}/${BIN_DIR}/custom_operator_fortran
	cp examples/hpddm_f90.cfg ${TOP_DIR}/${BIN_DIR}
	cd ${TOP_DIR}/${BIN_DIR} && echo 100 2 | ${MPIRUN} 4 ./custom_operator_fortran && cd -

test_bin/schwarz_cpp test_bin/schwarz_c test_examples/schwarz.py:
	${MPIRUN} 1 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_verbosity -hpddm_dump_matrices=${TRASH_DIR}/output.txt -hpddm_version
	@if [ -f ${LIB_DIR}/libhpddm_python.${EXTENSION_LIB} ] && [ -f ${TRASH_DIR}/output.txt ] && [ "$@" = "test_bin/schwarz_cpp" ]; then \
		CMD="${MPIRUN} 1 examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
		CMD="${MPIRUN} 1 examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity -hpddm_krylov_method=bgmres"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
		CMD="${MPIRUN} 1 examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity -generate_random_rhs 4"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
		CMD="${MPIRUN} 1 examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity 1 -hpddm_krylov_method=bgmres -generate_random_rhs=4 -hpddm_gmres_restart 5 -hpddm_deflation_tol 1e-6"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
		CMD="${MPIRUN} 1 examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity 1 -hpddm_krylov_method=bgmres -generate_random_rhs=4 -hpddm_gmres_restart 5 -hpddm_deflation_tol 1e-6 -hpddm_qr    cgs"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
	fi
	${MPIRUN} 1 $(subst test_,${SEP} ${TOP_DIR}/,$@) -symmetric_csr -hpddm_verbosity 4 -hpddm_help
	${MPIRUN} 1 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_verbosity -generate_random_rhs 8
	${MPIRUN} 1 $(subst test_,${SEP} ${TOP_DIR}/,$@) -symmetric_csr -hpddm_verbosity -generate_random_rhs 8
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_verbosity=1 --hpddm_gmres_restart=25 -hpddm_max_it 80 -generate_random_rhs 4 -hpddm_orthogonalization=mgs
	@if test ! $(findstring -DHPDDM_MIXED_PRECISION=1, ${HPDDMFLAGS}); then \
		CMD="${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_verbosity=1 --hpddm_gmres_restart=25 -hpddm_max_it 80 -generate_random_rhs 4 -hpddm_schwarz_coarse_correction deflated"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
	fi
ifdef EIGENSOLVER
	${MPIRUN} 2 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=2 -hpddm_verbosity=2 -symmetric_csr --hpddm_gmres_restart    20 -hpddm_dump_eigenvectors ${TRASH_DIR}/ev
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=4 --hpddm_gmres_restart=15 -hpddm_max_it 80 -hpddm_dump_matrices=${TRASH_DIR}/output -hpddm_level_2_push_prefix -hpddm_dump_matrix=${TRASH_DIR}/co -hpddm_assembly_hierarchy 2 -hpddm_pop_prefix
	@if [ -f ${LIB_DIR}/libhpddm_python.${EXTENSION_LIB} ]; then \
		CMD="${MPIRUN} 1 examples/solver.py ${TRASH_DIR}/output_1_4.txt"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
	fi
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -Nx 50 -Ny 50 -symmetric_csr -hpddm_level_2_p 2 -hpddm_level_2_distribution sol -hpddm_orthogonalization   mgs -hpddm_gmres_restart=25 -hpddm_level_2_hypre_solver=amg
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_level_2_p 2 -hpddm_gmres_restart=25
	@if [ "$@" = "test_bin/schwarz_cpp" ]; then \
		CMD="${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_myPrefix_schwarz_coarse_correction deflated -hpddm_myPrefix_geneo_nu=10 -hpddm_myPrefix_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_myPrefix_level_2_p 2 -hpddm_myPrefix_gmres_restart=25 -hpddm_verbosity=2 -prefix=myPrefix_ -hpddm_myPrefix_level_2_hypre_solver=pcg"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
		if [ "${SOLVER}" != "HYPRE" ]; then \
			CMD="${MPIRUN} 5 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_myPrefix_schwarz_coarse_correction deflated -hpddm_myPrefix_geneo_nu=10 -hpddm_myPrefix_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_myPrefix_level_2_p 2 -hpddm_myPrefix_gmres_restart=25 -hpddm_verbosity=2 -prefix=myPrefix_ -hpddm_myPrefix_geneo_threshold 0.2 -hpddm_myPrefix_geneo_force_uniformity min"; \
			echo "$${CMD}"; \
			$${CMD} || exit; \
			CMD="${MPIRUN} 5 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_myPrefix_schwarz_coarse_correction deflated -hpddm_myPrefix_geneo_nu=10 -hpddm_myPrefix_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_myPrefix_level_2_p 2 -hpddm_myPrefix_gmres_restart=25 -hpddm_verbosity=2 -prefix=myPrefix_ -hpddm_myPrefix_geneo_threshold 0.2 -hpddm_myPrefix_geneo_force_uniformity max"; \
			echo "$${CMD}"; \
			$${CMD} || exit; \
		fi \
	fi
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_level_2_p 2 -generate_random_rhs 8 -hpddm_krylov_method=bgmres -hpddm_gmres_restart=10 -hpddm_deflation_tol=1e-4 -hpddm_gmres_restart=25
	@if test ! $(findstring -DHPDDM_MIXED_PRECISION=1, ${HPDDMFLAGS}) && test ! $(findstring -DFORCE_SINGLE, ${HPDDMFLAGS}); then \
		CMD="${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction additive -hpddm_geneo_nu=1 -hpddm_verbosity=2 -Nx 20 -Ny 20 -symmetric_csr -hpddm_level_2_p 2 -generate_random_rhs 4 -hpddm_krylov_method=bfbcg -hpddm_deflation_tol=1e-4 -hpddm_schwarz_method asm -hpddm_geneo_threshold 1e+1"; \
		echo "$${CMD}"; \
		$${CMD} || exit; \
	fi
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_level_2_p 2 -generate_random_rhs 8 -hpddm_krylov_method=bgmres -hpddm_gmres_restart=10 -hpddm_deflation_tol=1e-4 -hpddm_qr=mgs -hpddm_gmres_restart=25
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_level_2_p 2 -generate_random_rhs 8 -hpddm_krylov_method=bgmres -hpddm_deflation_tol=1e-4 -hpddm_qr=cgs -hpddm_gmres_restart=25 -hpddm_orthogonalization=mgs
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr --hpddm_dump_matrices ${TRASH_DIR}/output -hpddm_gmres_restart=25
	@if [ -f ${LIB_DIR}/libhpddm_python.${EXTENSION_LIB} ]; then \
		${MPIRUN} 1 examples/solver.py ${TRASH_DIR}/output_2_4.txt; \
	fi
endif

test_bin/schwarz_cpp_custom_operator: ${TOP_DIR}/${BIN_DIR}/schwarz_cpp
	${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarz_cpp -hpddm_verbosity -hpddm_schwarz_method none -Nx 10 -Ny 10
	${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarz_cpp -symmetric_csr -hpddm_verbosity -hpddm_schwarz_method=none -Nx 10 -Ny 10
	${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarz_cpp -hpddm_verbosity -hpddm_schwarz_method none -Nx 10 -Ny 10 -hpddm_krylov_method bgmres
	${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarz_cpp -symmetric_csr -hpddm_verbosity -hpddm_schwarz_method=none -Nx 10 -Ny 10 ---hpddm_krylov_method bgmres

test_bin/schwarzFromFile_cpp: ${TOP_DIR}/${BIN_DIR}/schwarzFromFile_cpp
	@if [ -f ./examples/data/mini.tar.gz ]; then \
		mkdir -p ${TOP_DIR}/${TRASH_DIR}/data; \
		tar xzf ./examples/data/mini.tar.gz -C ${TOP_DIR}/${TRASH_DIR}/data; \
		for NP in 2 4; do \
			for OVERLAP in 1 3; do \
				CMD="${MPIRUN} $${NP} ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarzFromFile_cpp -matrix_filename=${TOP_DIR}/${TRASH_DIR}/data/mini.mtx -hpddm_verbosity 2 -overlap $${OVERLAP}"; \
				if [ "$${NP}" = "4" ] && [ "$${OVERLAP}" = "1" ]; then CMD="$${CMD} -rhs_filename=${TOP_DIR}/${TRASH_DIR}/data/ones.txt"; fi; \
				echo "$${CMD}"; \
				$${CMD} || exit; \
			done \
		done \
	fi


test_bin/driver: ${TOP_DIR}/${BIN_DIR}/driver
	@if [ -f ./examples/data/40X.tar.gz ]; then \
		mkdir -p ${TOP_DIR}/${TRASH_DIR}/data; \
		tar xzf ./examples/data/40X.tar.gz -C ${TOP_DIR}/${TRASH_DIR}/data; \
		for SCALING in 0 1; do \
			for VARIANT in left right; do \
				for MU in 1 3; do \
					CMD="${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/driver -path=${TOP_DIR}/${TRASH_DIR}/data -hpddm_max_it 1000 -hpddm_krylov_method gcrodr -hpddm_gmres_restart 40 -hpddm_recycle 20 -hpddm_tol 1e-10 -diagonal_scaling $${SCALING} -hpddm_variant $${VARIANT} -mu $${MU}"; \
					echo "$${CMD}"; \
					$${CMD} || exit; \
				done; \
				CMD="${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/driver -path=${TOP_DIR}/${TRASH_DIR}/data -hpddm_max_it 1000 -hpddm_krylov_method bgcrodr -hpddm_gmres_restart 40 -hpddm_recycle 20 -hpddm_tol 1e-10 -diagonal_scaling $${SCALING} -hpddm_variant $${VARIANT}"; \
				echo "$${CMD}"; \
				$${CMD} || exit; \
			done \
		done \
	fi

${TOP_DIR}/${TRASH_DIR}/%.d: ;

SOURCES = schwarz.cpp schwarzFromFile.cpp generate.cpp generateFromFile.cpp driver.cpp local_solver.cpp local_eigensolver.cpp schwarz.c generate.c
INTERFACES = hpddm_c.cpp hpddm_python.cpp hpddm_fortran.cpp
-include $(patsubst %,${TOP_DIR}/${TRASH_DIR}/%.d,$(subst .,_,${SOURCES}))
-include $(patsubst %,${TOP_DIR}/${TRASH_DIR}/%.d,$(basename ${INTERFACES}))
-include $(patsubst %,${TOP_DIR}/${TRASH_DIR}/lib%.d,$(basename ${INTERFACES}))
