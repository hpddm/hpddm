#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2016-01-08

   Copyright (C) 2016-     Centre National de la Recherche Scientifique

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
import sys
sys.path.append('interface')
import ctypes
import numpy
import scipy.sparse
import scipy.sparse.linalg
import hpddm
import re

try:
    xrange
except NameError:
    xrange = range

opt = hpddm.optionGet()
args = ctypes.create_string_buffer(' '.join(sys.argv[1:]).encode('ascii', 'ignore'))
hpddm.optionParse(opt, args)
def appArgs():
    val = (ctypes.c_char_p * 1)()
    val[0] = b'generate_random_rhs=<1>'
    desc = (ctypes.c_char_p * 1)()
    desc[0] =  b'Number of generated random right-hand sides.'
    hpddm.optionParseInts(opt, args, 1, ctypes.cast(val, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(desc, ctypes.POINTER(ctypes.c_char_p)))
    val[0] = b'matrix_filename=<input_file>'
    desc[0] = b'Name of the file in which the matrix is stored.'
    hpddm.optionParseArgs(opt, args, 1, ctypes.cast(val, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(desc, ctypes.POINTER(ctypes.c_char_p)))
    val = None
    desc = None
appArgs()

filename = hpddm.optionPrefix(opt, b'matrix_filename')
if len(filename) == 0:
    print('Please specity a -matrix_filename=<input_file>')
    sys.exit(1)

n, m, nnz, a, ia, ja, sym = hpddm.parse_file(filename)

Mat = hpddm.matrixCSRCreate(n, m, nnz, a, ia, ja, sym)
if hpddm.numbering.value == b'F':
    ia[:] -= 1
    ja[:] -= 1
csr = scipy.sparse.csr_matrix((a, ja, ia), shape = (n, m), dtype = hpddm.scalar)

mu = int(hpddm.optionApp(opt, b'generate_random_rhs'))
shape = n if mu == 1 else (n, mu)
sol = numpy.zeros(shape, order = 'F', dtype = hpddm.scalar)
f = numpy.empty_like(sol)
f[:] = numpy.random.random_sample(shape)
if hpddm.scalar == numpy.complex64 or hpddm.scalar == numpy.complex128:
    f[:] += numpy.random.random_sample(shape) * 1j

lu = scipy.sparse.linalg.spilu(csr.tocsc(), drop_tol = 1e-6, fill_factor = 14)
@hpddm.precondFunc
def precond(y, x, n, m):
    if m == 1:
        x._shape_ = (n,)
        y._shape_ = (n,)
        x = numpy.ctypeslib.as_array(x, (n,))
        y = numpy.ctypeslib.as_array(y, (n,))
    else:
        x._shape_ = (m, n)
        y._shape_ = (m, n)
        x = numpy.ctypeslib.as_array(x, (m, n)).transpose()
        y = numpy.ctypeslib.as_array(y, (m, n)).transpose()
    x[:] = lu.solve(y[:])

if hpddm.optionVal(opt, b'krylov_method') == 1:
    hpddm.BGMRES(Mat, precond, f, sol)
else:
    hpddm.GMRES(Mat, precond, f, sol)

status = 0
nrmb = numpy.linalg.norm(f, axis = 0)
tmp = numpy.empty_like(f)
hpddm.csrmv(Mat, sol, tmp)
tmp -= f
nrmAx = numpy.linalg.norm(tmp, axis = 0)
if mu == 1:
    nrmb = [ nrmb ]
    nrmAx = [ nrmAx ]
for nu in xrange(mu):
    if nu == 0:
        print(' --- error = ', end = '')
    else:
        print('             ', end = '')
    print('{:e} / {:e}'.format(nrmAx[nu], nrmb[nu]), end = '')
    if mu > 1:
        print(' (rhs #{:d})'.format(nu + 1), end = '')
    print('')
    if nrmAx[nu] / nrmb[nu] > (1.0e-4 if ctypes.sizeof(hpddm.underlying) == ctypes.sizeof(ctypes.c_double) else 1.0e-2):
        status = 1
hpddm.matrixCSRDestroy(ctypes.byref(Mat))
sys.exit(status)
