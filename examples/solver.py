#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-11-17

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

import sys
sys.path.append('interface')
import ctypes
import numpy
import hpddm
import re

pair = re.compile(r'\(([^,\)]+),([^,\)]+)\)')
def parse_pair(s):
    match = pair.match(s)
    if match is None:
        return float(s) + 0j
    else:
        return complex(*map(float, match.groups()))

with open(sys.argv[1], "r") as input:
    k = 0
    for line in input:
        if not line.startswith('# '):
            words = line.split()
            y = 0
            for w in words:
                if k == 0:
                    if y == 0:
                        n = int(w)
                    elif y == 1:
                        m = int(w)
                    elif y == 2:
                        sym = bool(int(w) == 1)
                    elif y == 3:
                        nnz = int(w)
                        ia = numpy.zeros(n + 1, dtype = ctypes.c_int)
                        ja = numpy.empty(nnz, dtype = ctypes.c_int)
                        a = numpy.empty(nnz, dtype = hpddm.scalar)
                        ia[0] = (hpddm.numbering.value == 'F')
                        ia[n] = (hpddm.numbering.value == 'F')
                else:
                    if y == 0:
                        ia[int(w)] += 1
                    elif y == 1:
                        ja[k - 1] = int(w) - (hpddm.numbering.value == 'C')
                    else:
                        if hpddm.scalar == ctypes.c_double or hpddm.scalar == ctypes.c_float:
                            a[k - 1] = w
                        else:
                            a[k - 1] = parse_pair(w)
                y += 1
            k += 1
ia[:] = numpy.cumsum(ia[:])
Mat = hpddm.matrixCSRCreate(n, m, nnz, a, ia, ja, sym)
S = ctypes.POINTER(hpddm.Subdomain)()
hpddm.subdomainNumfact(ctypes.byref(S), Mat)
f = numpy.empty(n, hpddm.scalar)
f[:] = numpy.random.random_sample(n)
if hpddm.scalar == numpy.complex64 or hpddm.scalar == numpy.complex128:
    f[:] += numpy.random.random_sample(n) * 1j
sol = numpy.empty_like(f)
hpddm.subdomainSolve(S, f, sol)
nrmb = numpy.linalg.norm(f)
tmp = numpy.empty_like(f)
hpddm.csrmv(Mat, sol, tmp)
tmp -= f
nrmAx = numpy.linalg.norm(tmp)
print " --- error = {:e} / {:e}".format(nrmAx, nrmb)
hpddm.matrixCSRDestroy(ctypes.byref(Mat))
