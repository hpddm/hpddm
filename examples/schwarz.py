#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-11-06

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
"""

from mpi4py import MPI
import sys
sys.path.append('interface')
import ctypes
import numpy
import hpddm
from generate import *

rankWorld = MPI.COMM_WORLD.Get_rank()
sizeWorld = MPI.COMM_WORLD.Get_size()
opt = hpddm.optionGet()
args = ctypes.create_string_buffer(" ".join(sys.argv[1:]))
hpddm.optionParse(opt, args, rankWorld == 0)
def appArgs():
    val = (ctypes.c_char_p * 3)()
    (val[0], val[1], val[2]) = [ "Nx=<100>", "Ny=<100>", "overlap=<1>" ]
    desc = (ctypes.c_char_p * 3)()
    (desc[0], desc[1], desc[2]) = [ "Number of grid points in the x-direction.", "Number of grid points in the y-direction.", "Number of grid points in the overlap." ]
    hpddm.optionParseInts(opt, args, 3, ctypes.cast(val, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(desc, ctypes.POINTER(ctypes.c_char_p)))
    (val[0], val[1]) = [ "symmetric_csr=(0|1)", "nonuniform=(0|1)" ]
    (desc[0], desc[1]) = [ "Assemble symmetric matrices.", "Use a different number of eigenpairs to compute on each subdomain." ]
    hpddm.optionParseArgs(opt, args, 2, ctypes.cast(val, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(desc, ctypes.POINTER(ctypes.c_char_p)))
    val = None
    desc = None
appArgs()
if rankWorld != 0:
    hpddm.optionRemove(opt, 'verbosity')
o, connectivity, dof, Mat, MatNeumann, d, f, sol = generate(rankWorld, sizeWorld)
status = 0
if sizeWorld > 1:
    A = hpddm.schwarzCreate(Mat, o, connectivity)
    hpddm.schwarzMultiplicityScaling(A, d)
    hpddm.schwarzInitialize(A, d)
    if hpddm.optionSet(opt, "schwarz_coarse_correction"):
        nu = ctypes.c_ushort(int(hpddm.optionVal(opt, "geneo_nu")))
        if nu.value > 0:
            if hpddm.optionApp(opt, "nonuniform"):
                nu.value += max(int(-hpddm.optionVal(opt, "geneo_nu") + 1), (-1)**rankWorld * rankWorld)
            threshold = hpddm.underlying(max(0, hpddm.optionVal(opt, "geneo_threshold")))
            hpddm.schwarzSolveGEVP(A, MatNeumann, ctypes.byref(nu), threshold)
            addr = hpddm.optionAddr(opt, "geneo_nu")
            addr.contents.value = nu.value
        else:
            nu = 1
            deflation = numpy.ones((nu, dof), dtype = hpddm.scalar)
            hpddm.setVectors(hpddm.schwarzPreconditioner(A), nu, deflation)
        hpddm.initializeCoarseOperator(hpddm.schwarzPreconditioner(A), nu)
        hpddm.schwarzBuildCoarseOperator(A, hpddm.MPI_Comm.from_address(MPI._addressof(MPI.COMM_WORLD)))
    hpddm.schwarzCallNumfact(A)
    comm = hpddm.getCommunicator(hpddm.schwarzPreconditioner(A))
    if hpddm.optionVal(opt, "krylov_method") == 1:
        it = hpddm.CG(A, sol, f, comm)
    else:
        it = hpddm.GMRES(A, sol, f, 1, comm)
    storage = numpy.empty(2, dtype = hpddm.underlying)
    hpddm.schwarzComputeError(A, sol, f, storage)
    if rankWorld == 0:
        print " --- error = {:e} / {:e}".format(storage[1], storage[0])
    if it > 45 or storage[1] / storage[0] > 1.0e-2:
        status = 1
    hpddm.schwarzDestroy(ctypes.byref(A))
else:
    S = ctypes.POINTER(hpddm.Subdomain)()
    hpddm.subdomainNumfact(ctypes.byref(S), Mat)
    hpddm.subdomainSolve(S, f, sol)
    nrmb = numpy.linalg.norm(f)
    tmp = numpy.empty_like(f)
    hpddm.csrmv(Mat, sol, tmp)
    tmp -= f
    nrmAx = numpy.linalg.norm(tmp)
    print " --- error = {:e} / {:e}".format(nrmAx, nrmb)
    if nrmAx / nrmb > (1.0e-8 if ctypes.sizeof(hpddm.underlying) == ctypes.sizeof(ctypes.c_double) else 1.0e-2):
        status = 1
    hpddm.subdomainDestroy(ctypes.byref(S))
    hpddm.matrixCSRDestroy(ctypes.byref(Mat))
hpddm.matrixCSRDestroy(ctypes.byref(MatNeumann))
sys.exit(status)
