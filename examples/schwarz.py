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

from __future__ import print_function
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
args = ctypes.create_string_buffer(' '.join(sys.argv[1:]).encode('ascii', 'ignore'))
hpddm.optionParse(opt, args, rankWorld == 0)
def appArgs():
    val = (ctypes.c_char_p * 4)()
    (val[0], val[1], val[2], val[3]) = [ b'Nx=<100>', b'Ny=<100>', b'overlap=<1>', b'generate_random_rhs=<0>' ]
    desc = (ctypes.c_char_p * 4)()
    (desc[0], desc[1], desc[2], desc[3]) = [ b'Number of grid points in the x-direction.', b'Number of grid points in the y-direction.', b'Number of grid points in the overlap.', b'Number of generated random right-hand sides.' ]
    hpddm.optionParseInts(opt, args, 4, ctypes.cast(val, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(desc, ctypes.POINTER(ctypes.c_char_p)))
    (val[0], val[1]) = [ b'symmetric_csr=(0|1)', b'nonuniform=(0|1)' ]
    (desc[0], desc[1]) = [ b'Assemble symmetric matrices.', b'Use a different number of eigenpairs to compute on each subdomain.' ]
    hpddm.optionParseArgs(opt, args, 2, ctypes.cast(val, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(desc, ctypes.POINTER(ctypes.c_char_p)))
    val = None
    desc = None
appArgs()
if rankWorld != 0:
    hpddm.optionRemove(opt, b'verbosity')
o, connectivity, dof, Mat, MatNeumann, d, f, sol, mu = generate(rankWorld, sizeWorld)
status = 0
if sizeWorld > 1:
    A = hpddm.schwarzCreate(Mat, o, connectivity)
    hpddm.schwarzMultiplicityScaling(A, d)
    hpddm.schwarzInitialize(A, d)
    if mu != 0:
        hpddm.schwarzScaledExchange(A, f)
    else:
        mu = 1
    if hpddm.optionSet(opt, b'schwarz_coarse_correction'):
        nu = ctypes.c_ushort(int(hpddm.optionVal(opt, b'geneo_nu')))
        if nu.value > 0:
            if hpddm.optionApp(opt, b'nonuniform'):
                nu.value += max(int(-hpddm.optionVal(opt, b'geneo_nu') + 1), (-1)**rankWorld * rankWorld)
            threshold = hpddm.underlying(max(0, hpddm.optionVal(opt, b'geneo_threshold')))
            hpddm.schwarzSolveGEVP(A, MatNeumann, ctypes.byref(nu), threshold)
            addr = hpddm.optionAddr(opt, b'geneo_nu')
            addr.contents.value = nu.value
        else:
            nu = 1
            deflation = numpy.ones((dof, nu), order = 'F', dtype = hpddm.scalar)
            hpddm.setVectors(hpddm.schwarzPreconditioner(A), nu, deflation)
        hpddm.initializeCoarseOperator(hpddm.schwarzPreconditioner(A), nu)
        hpddm.schwarzBuildCoarseOperator(A, hpddm.MPI_Comm.from_address(MPI._addressof(MPI.COMM_WORLD)))
    hpddm.schwarzCallNumfact(A)
    comm = hpddm.getCommunicator(hpddm.schwarzPreconditioner(A))
    if hpddm.optionVal(opt, b'krylov_method') == 2:
        it = hpddm.CG(A, f, sol, comm)
    elif hpddm.optionVal(opt, b'krylov_method') == 1:
        it = hpddm.BGMRES(A, f, sol, comm)
    else:
        it = hpddm.GMRES(A, f, sol, comm)
    storage = numpy.empty(2 * mu, order = 'F', dtype = hpddm.underlying)
    hpddm.schwarzComputeError(A, sol, f, storage)
    if rankWorld == 0:
        for nu in xrange(mu):
            if nu == 0:
                print(' --- error = ', end = '')
            else:
                print('             ', end = '')
            print('{:e} / {:e}'.format(storage[1 + 2 * nu], storage[2 * nu]), end = '')
            if mu > 1:
                print(' (rhs #{:d})'.format(nu + 1), end = '')
            print('')
    if it > 45:
        status = 1
    else:
        for nu in xrange(mu):
            if storage[1 + 2 * nu] / storage[2 * nu] > 1.0e-2:
                status = 1
    hpddm.schwarzDestroy(ctypes.byref(A))
else:
    S = ctypes.POINTER(hpddm.Subdomain)()
    hpddm.subdomainNumfact(ctypes.byref(S), Mat)
    hpddm.subdomainSolve(S, f, sol)
    nrmb = numpy.linalg.norm(f, axis = 0)
    tmp = numpy.empty_like(f)
    hpddm.csrmv(Mat, sol, tmp)
    tmp -= f
    nrmAx = numpy.linalg.norm(tmp, axis = 0)
    if mu == 0:
        nrmb = [ nrmb ]
        nrmAx = [ nrmAx ]
        mu = 1
    for nu in xrange(mu):
        if nu == 0:
            print(' --- error = ', end = '')
        else:
            print('             ', end = '')
        print('{:e} / {:e}'.format(nrmAx[nu], nrmb[nu]), end = '')
        if mu > 1:
            print(' (rhs #{:d})'.format(nu + 1), end = '')
        print('')
        if nrmAx[nu] / nrmb[nu] > (1.0e-8 if ctypes.sizeof(hpddm.underlying) == ctypes.sizeof(ctypes.c_double) else 1.0e-2):
            status = 1
    hpddm.subdomainDestroy(ctypes.byref(S))
    hpddm.matrixCSRDestroy(ctypes.byref(Mat))
hpddm.matrixCSRDestroy(ctypes.byref(MatNeumann))
sys.exit(status)
