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

Mat = hpddm.matrixCSRParseFile(ctypes.create_string_buffer(sys.argv[1].encode('ascii', 'ignore')))
n = hpddm.matrixCSRnRows(Mat)
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
print(' --- residual = {:e} / {:e}'.format(nrmAx, nrmb))
if nrmAx / nrmb > (1.0e-8 if ctypes.sizeof(hpddm.underlying) == ctypes.sizeof(ctypes.c_double) else 1.0e-2):
    status = 1
hpddm.subdomainDestroy(ctypes.byref(S))
hpddm.matrixCSRDestroy(ctypes.byref(Mat))
