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

TOP_DIR = .
LIST_COMPILATION ?= cpp c python
TRASH_DIR = .trash

all: Makefile.inc ${LIST_COMPILATION}

cpp: bin/schwarz_cpp
c: bin/schwarz_c
python: lib/libhpddm_python.${EXTENSION_LIB}

Makefile.inc:
	@echo "No Makefile.inc found, please choose one from directory Make.inc"
	@if [ -z ${MAKE_OS} ]; then \
		exit 1; \
	else \
		echo "${MAKE_OS} detected, trying to use Make.inc/Makefile.${MAKE_OS}"; \
		cp Make.inc/Makefile.${MAKE_OS} Makefile.inc; \
	fi

clean:
	rm -rf bin lib ${TRASH_DIR}
	find ${TOP_DIR} \( -name "*.o" -o -name "*.${EXTENSION_LIB}" -o -name "*.pyc" \) -exec rm -vf '{}' ';'

bin/%_cpp.o: examples/%.cpp
	@mkdir -p bin
	${MPICXX} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -c $? -o $@

bin/%_c.o: examples/%.c
	@mkdir -p bin
	${MPICC} ${CFLAGS} ${HPDDMFLAGS} ${INCS} -c $? -o $@

bin/%.o: interface/%.cpp
	@mkdir -p bin
	${MPICXX} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} -c $? -o $@

bin/schwarz_c: bin/schwarz_c.o bin/generate_c.o bin/hpddm_c.o
	${MPICXX} $^ -o $@ ${LIBS}

bin/schwarz_cpp: bin/schwarz_cpp.o bin/generate_cpp.o
	${MPICXX} $^ -o $@ ${LIBS}

lib/libhpddm_python.${EXTENSION_LIB}: interface/hpddm_python.cpp
	@mkdir -p lib
	${MPICXX} ${CXXFLAGS} ${HPDDMFLAGS} ${INCS} ${PYTHON_INCS} -shared $? -o $@ ${LIBS} ${PYTHON_LIBS}

test: all $(addprefix test_, $(LIST_COMPILATION))

test_cpp: bin/schwarz_cpp test_bin/schwarz_cpp test_bin/schwarz_cpp_custom_op
test_c: bin/schwarz_c test_bin/schwarz_c
test_python: lib/libhpddm_python.${EXTENSION_LIB} test_examples/schwarz.py
test_bin/schwarz_cpp test_bin/schwarz_c test_examples/schwarz.py:
	@mkdir -p ${TRASH_DIR}
	${MPIRUN} 1 $(subst test_,${SEP},$@) -hpddm_verbosity
	${MPIRUN} 1 $(subst test_,${SEP},$@) -symmetric_csr -hpddm_verbosity
	${MPIRUN} 1 $(subst test_,${SEP},$@) -hpddm_verbosity -generate_random_rhs 8
	${MPIRUN} 1 $(subst test_,${SEP},$@) -symmetric_csr -hpddm_verbosity -generate_random_rhs 8
	${MPIRUN} 2 $(subst test_,${SEP},$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=2 -hpddm_verbosity=2 -symmetric_csr
	${MPIRUN} 4 $(subst test_,${SEP},$@) -hpddm_tol=1.0e-6 -hpddm_verbosity=2 --hpddm_gmres_restart=25 -hpddm_max_it 80 -generate_random_rhs 4
	${MPIRUN} 4 $(subst test_,${SEP},$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 --hpddm_gmres_restart=15 -hpddm_max_it 80 -hpddm_dump_local_matrix_1=${TRASH_DIR}/output
	@if [ -f lib/libhpddm_python.${EXTENSION_LIB} ]; then \
		examples/solver.py ${TRASH_DIR}/output_1_4.txt; \
	fi
	${MPIRUN} 4 $(subst test_,${SEP},$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -Nx 50 -Ny 50 -symmetric_csr -hpddm_master_p 2 -distributed_sol
	${MPIRUN} 4 $(subst test_,${SEP},$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_master_p 2
	${MPIRUN} 4 $(subst test_,${SEP},$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr -hpddm_master_p 2 -generate_random_rhs 8 -hpddm_krylov_method=bgmres -hpddm_gmres_restart=10 -hpddm_initial_deflation_tol=1e-4
	${MPIRUN} 4 $(subst test_,${SEP},$@) -hpddm_tol=1.0e-6 -hpddm_schwarz_coarse_correction deflated -hpddm_geneo_nu=10 -hpddm_verbosity=2 -nonuniform -Nx 50 -Ny 50 -symmetric_csr --hpddm_dump_local_matrix_2 ${TRASH_DIR}/output
	@if [ -f lib/libhpddm_python.${EXTENSION_LIB} ]; then \
		examples/solver.py ${TRASH_DIR}/output_2_4.txt; \
	fi
	${MPIRUN} 8 $(subst test_,${SEP},$@) -hpddm_tol=1.0e-4 -hpddm_schwarz_coarse_correction balanced -hpddm_geneo_nu=0 -hpddm_verbosity=2 -Nx 40 -Ny 40 -hpddm_variant=right -symmetric_csr -hpddm_dump_local_matrix_2 ${TRASH_DIR}/output
	@if [ -f lib/libhpddm_python.${EXTENSION_LIB} ]; then \
		examples/solver.py ${TRASH_DIR}/output_2_8.txt; \
	fi

test_bin/schwarz_cpp_custom_op: bin/schwarz_cpp
	${MPIRUN} 1 bin/schwarz_cpp -hpddm_verbosity -hpddm_schwarz_method none -Nx 10 -Ny 10
	${MPIRUN} 1 bin/schwarz_cpp -symmetric_csr -hpddm_verbosity -hpddm_schwarz_method=none -Nx 10 -Ny 10
	${MPIRUN} 1 bin/schwarz_cpp -hpddm_verbosity -hpddm_schwarz_method none -Nx 10 -Ny 10 -hpddm_krylov_method bgmres
	${MPIRUN} 1 bin/schwarz_cpp -symmetric_csr -hpddm_verbosity -hpddm_schwarz_method=none -Nx 10 -Ny 10 ---hpddm_krylov_method bgmres
