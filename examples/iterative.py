#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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
    val = (ctypes.c_char_p * 2)()
    (val[0], val[1]) = [ b'generate_random_rhs=<1>', b'fill_factor=<18>' ]
    desc = (ctypes.c_char_p * 2)()
    (desc[0], desc[1]) = [ b'Number of generated random right-hand sides.', b'Specifies the fill ratio upper bound (>= 1.0) for ILU.' ]
    hpddm.optionParseInts(opt, args, 2, ctypes.cast(val, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(desc, ctypes.POINTER(ctypes.c_char_p)))
    val[0] = b'drop_tol=<1.0e-4>'
    desc[0] = b'Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.'
    hpddm.optionParseDoubles(opt, args, 1, ctypes.cast(val, ctypes.POINTER(ctypes.c_char_p)), ctypes.cast(desc, ctypes.POINTER(ctypes.c_char_p)))
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

lu = scipy.sparse.linalg.spilu(csr.tocsc(), drop_tol = hpddm.optionApp(opt, b'drop_tol'), fill_factor = hpddm.optionApp(opt, b'fill_factor'))
@hpddm.precondFunc
def precond(y, x, n, m):
    if hpddm.scalar is not hpddm.underlying:
        factor = 2
    else:
        factor = 1
    ptr = ctypes.cast(x, ctypes.POINTER(hpddm.underlying * factor * m * n))
    x = numpy.frombuffer(ptr.contents)
    ptr = ctypes.cast(y, ctypes.POINTER(hpddm.underlying * factor * m * n))
    y = numpy.frombuffer(ptr.contents)
    x = x.view(dtype = hpddm.scalar)
    y = y.view(dtype = hpddm.scalar)
    if m > 1:
        x = numpy.reshape(x, (n, m), order = 'F')
        y = numpy.reshape(y, (n, m), order = 'F')
    x[:] = lu.solve(y[:])

it = hpddm.solve(Mat, precond, f, sol)

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
        print(' --- residual = ', end = '')
    else:
        print('                ', end = '')
    print('{:e} / {:e}'.format(nrmAx[nu], nrmb[nu]), end = '')
    if mu > 1:
        print(' (rhs #{:d})'.format(nu + 1), end = '')
    print('')
    if nrmAx[nu] / nrmb[nu] > (1.0e-4 if ctypes.sizeof(hpddm.underlying) == ctypes.sizeof(ctypes.c_double) else 1.0e-2):
        status = 1
if it > 50:
    status = 1
hpddm.matrixCSRDestroy(ctypes.byref(Mat))
sys.exit(status)
