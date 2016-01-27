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

TOP_DIR ?= .
BIN_DIR = bin
LIB_DIR = lib
TRASH_DIR = .trash

$(shell mkdir -p ${TOP_DIR}/${BIN_DIR} > /dev/null)
$(shell mkdir -p ${TOP_DIR}/${LIB_DIR} > /dev/null)
$(shell mkdir -p ${TOP_DIR}/${TRASH_DIR} > /dev/null)

-include Makefile.inc

DEPFLAGS = -MT $@ -MMD -MP -MF ${TOP_DIR}/${TRASH_DIR}/$(notdir $(basename $@)).Td
POSTCOMPILE = mv -f ${TOP_DIR}/${TRASH_DIR}/$(notdir $(basename $@)).Td ${TOP_DIR}/${TRASH_DIR}/$(subst lib,,$(notdir $(basename $@))).d

INCS += -I./include -I./interface

SOLVER ?= SUITESPARSE
SUBSOLVER ?= SUITESPARSE
override HPDDMFLAGS += -DD${SOLVER} -D${SUBSOLVER}SUB
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

LIBS += ${ARPACK_LIBS} ${SCALAPACK_LIBS}
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
else
    UNAME_S := ${shell uname -s}
    ifeq (${UNAME_S}, Linux)
        MAKE_OS = Linux
        EXTENSION_LIB = so
    endif
    ifeq (${UNAME_S}, Darwin)
        MAKE_OS = OSX
        EXTENSION_LIB = dylib
    endif
endif

ifneq (, $(shell which mpixlc 2> /dev/null))
    SEP = : 
endif

LIST_COMPILATION ?= cpp c python

all: Makefile.inc ${LIST_COMPILATION}

cpp: ${TOP_DIR}/${BIN_DIR}/schwarz_cpp
c: ${TOP_DIR}/${BIN_DIR}/schwarz_c
python: ${TOP_DIR}/${LIB_DIR}/libhpddm_python.${EXTENSION_LIB}

Makefile.inc:
	@echo "No Makefile.inc found, please choose one from directory Make.inc"
	@if [ -z ${MAKE_OS} ]; then \
		exit 1; \
	else \
		echo "${MAKE_OS} detected, trying to use Make.inc/Makefile.${MAKE_OS}"; \
		cp Make.inc/Makefile.${MAKE_OS} Makefile.inc; \
	fi

clean:
	rm -rf ${TOP_DIR}/${BIN_DIR} ${TOP_DIR}/${LIB_DIR} ${TOP_DIR}/${TRASH_DIR}
	find ${TOP_DIR} \( -name "*.o" -o -name "*.${EXTENSION_LIB}" -o -name "*.pyc" -o -name "*.gcov" \) -exec rm -vf '{}' ';'

${TOP_DIR}/${BIN_DIR}/%_cpp.o: examples/%.cpp ${TOP_DIR}/${TRASH_DIR}/%.d
	${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -c $< -o $@
	${POSTCOMPILE}

${TOP_DIR}/${BIN_DIR}/%_c.o: examples/%.c ${TOP_DIR}/${TRASH_DIR}/%.d
	${MPICC} ${DEPFLAGS} ${CFLAGS} ${HPDDMFLAGS} ${INCS} -c $< -o $@
	${POSTCOMPILE}

${TOP_DIR}/${BIN_DIR}/%.o: interface/%.cpp ${TOP_DIR}/${TRASH_DIR}/%.d
	${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -c $< -o $@
	${POSTCOMPILE}

${TOP_DIR}/${BIN_DIR}/schwarz_c: ${TOP_DIR}/${BIN_DIR}/schwarz_c.o ${TOP_DIR}/${BIN_DIR}/generate_c.o ${TOP_DIR}/${BIN_DIR}/hpddm_c.o
	${MPICXX} $^ -o $@ ${LIBS}

${TOP_DIR}/${BIN_DIR}/schwarz_cpp: ${TOP_DIR}/${BIN_DIR}/schwarz_cpp.o ${TOP_DIR}/${BIN_DIR}/generate_cpp.o
	${MPICXX} $^ -o $@ ${LIBS}

${TOP_DIR}/${BIN_DIR}/driver: ${TOP_DIR}/${BIN_DIR}/driver_cpp.o
	${MPICXX} $^ -o $@ ${LIBS}

${TOP_DIR}/${LIB_DIR}/lib%.${EXTENSION_LIB}: interface/%.cpp ${TOP_DIR}/${TRASH_DIR}/%.d
	${MPICXX} ${DEPFLAGS} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} ${PYTHON_INCS} -shared $< -o $@ ${LIBS} ${PYTHON_LIBS}
	${POSTCOMPILE}

test: all $(addprefix test_, ${LIST_COMPILATION})

test_cpp: ${TOP_DIR}/${BIN_DIR}/schwarz_cpp test_bin/schwarz_cpp test_bin/schwarz_cpp_custom_op
test_c: ${TOP_DIR}/${BIN_DIR}/schwarz_c test_bin/schwarz_c
test_python: ${TOP_DIR}/${LIB_DIR}/libhpddm_python.${EXTENSION_LIB} test_examples/schwarz.py
test_bin/schwarz_cpp test_bin/schwarz_c test_examples/schwarz.py:
	${MPIRUN} 1 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_verbosity -hpddm_dump_local_matrices=${TRASH_DIR}/output.txt
	@if [ -f ${LIB_DIR}/libhpddm_python.${EXTENSION_LIB} ] && [ -f ${TRASH_DIR}/output.txt ] && [ "$@" = "test_bin/schwarz_cpp" ] ; then \
		examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity ; \
		examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity -hpddm_krylov_method=bgmres; \
		examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity -generate_random_rhs 4; \
		examples/iterative.py -matrix_filename ${TRASH_DIR}/output.txt -hpddm_verbosity 1 -hpddm_krylov_method=bgmres -generate_random_rhs=4 -hpddm_gmres_restart 5 -hpddm_initial_deflation_tol 1e-6; \
	fi
	${MPIRUN} 1 $(subst test_,${SEP} ${TOP_DIR}/,$@) -symmetric_csr -hpddm_verbosity
	${MPIRUN} 1 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_verbosity -generate_random_rhs 8
	${MPIRUN} 1 $(subst test_,${SEP} ${TOP_DIR}/,$@) -symmetric_csr -hpddm_verbosity -generate_random_rhs 8
	${MPIRUN} 2 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=2 -hpddm_verbosity=2 -symmetric_csr --hpddm_gmres_restart    20
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_tol=1.0e-6 -hpddm_verbosity=1 --hpddm_gmres_restart=25 -hpddm_max_it 80 -generate_random_rhs 4 -hpddm_orthogonalization=mgs
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 --hpddm_gmres_restart=15 -hpddm_max_it 80 -hpddm_dump_local_matrix_1=${TRASH_DIR}/output
	@if [ -f ${LIB_DIR}/libhpddm_python.${EXTENSION_LIB} ]; then \
		examples/solver.py ${TRASH_DIR}/output_1_4.txt; \
	fi
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -Nx 50 -Ny 50 -symmetric_csr -hpddm_master_p 2 -distributed_sol -hpddm_orthogonalization   mgs
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_master_p 2
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_master_p 2 -generate_random_rhs 8 -hpddm_krylov_method=bgmres -hpddm_gmres_restart=10 -hpddm_initial_deflation_tol=1e-4
	${MPIRUN} 4 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr --hpddm_dump_local_matrix_2 ${TRASH_DIR}/output
	@if [ -f ${LIB_DIR}/libhpddm_python.${EXTENSION_LIB} ]; then \
		examples/solver.py ${TRASH_DIR}/output_2_4.txt; \
	fi
	${MPIRUN} 8 $(subst test_,${SEP} ${TOP_DIR}/,$@) -hpddm_tol=1.0e-4 -hpddm_schwarz_coarse_correction balanced -hpddm_geneo_nu=0 -hpddm_verbosity=2 -Nx 40 -Ny 40 -hpddm_variant=right -symmetric_csr -hpddm_dump_local_matrix_2 ${TRASH_DIR}/output
	@if [ -f ${LIB_DIR}/libhpddm_python.${EXTENSION_LIB} ]; then \
		examples/solver.py ${TRASH_DIR}/output_2_8.txt; \
	fi

test_bin/schwarz_cpp_custom_op: ${TOP_DIR}/${BIN_DIR}/schwarz_cpp
	${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarz_cpp -hpddm_verbosity -hpddm_schwarz_method none -Nx 10 -Ny 10
	${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarz_cpp -symmetric_csr -hpddm_verbosity -hpddm_schwarz_method=none -Nx 10 -Ny 10
	${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarz_cpp -hpddm_verbosity -hpddm_schwarz_method none -Nx 10 -Ny 10 -hpddm_krylov_method bgmres
	${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/schwarz_cpp -symmetric_csr -hpddm_verbosity -hpddm_schwarz_method=none -Nx 10 -Ny 10 ---hpddm_krylov_method bgmres

test_bin/driver: ${TOP_DIR}/${BIN_DIR}/driver
	mkdir -p ${TOP_DIR}/${TRASH_DIR}/data
	tar xzf ./examples/data/40X.tar.gz -C ${TOP_DIR}/${TRASH_DIR}/data
	for SCALING in 0 1; do \
		for VARIANT in left right; do \
			for MU in 1 3; do \
				${MPIRUN} 1 ${SEP} ${TOP_DIR}/${BIN_DIR}/driver -path=${TOP_DIR}/${TRASH_DIR}/data -hpddm_max_it 1000 -hpddm_krylov_method gcrodr -hpddm_gmres_restart 40 -hpddm_gmres_recycle 20 -hpddm_tol 1e-10 -diagonal_scaling $${SCALING} -hpddm_variant $${VARIANT} -mu $${MU}; \
			done \
		done \
	done

${TOP_DIR}/${TRASH_DIR}/%.d: ;

SOURCES = schwarz.cpp generate.cpp driver.cpp schwarz.c generate.c
INTERFACES = hpddm_c.cpp hpddm_python.cpp
-include $(patsubst %,${TOP_DIR}/${TRASH_DIR}/%.d,$(subst .,_,${SOURCES}))
-include $(patsubst %,${TOP_DIR}/${TRASH_DIR}/%.d,$(basename ${INTERFACES}))
