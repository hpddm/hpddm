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
import ctypes
import numpy
import hpddm
import math

try:
    xrange
except NameError:
    xrange = range

def generate(rankWorld, sizeWorld):
    opt = hpddm.optionGet()
    Nx = int(hpddm.optionApp(opt, b'Nx'))
    Ny = int(hpddm.optionApp(opt, b'Ny'))
    overlap = int(hpddm.optionApp(opt, b'overlap'))
    sym = bool(hpddm.optionApp(opt, b'symmetric_csr'))
    xGrid = int(math.sqrt(sizeWorld))
    while sizeWorld % xGrid != 0:
        --xGrid
    yGrid = int(sizeWorld / xGrid)

    y = int(rankWorld / xGrid)
    x = rankWorld - xGrid * y
    iStart = max(x * int(Nx / xGrid) - overlap, 0)
    iEnd   = min((x + 1) * int(Nx / xGrid) + overlap, Nx)
    jStart = max(y * int(Ny / yGrid) - overlap, 0)
    jEnd   = min((y + 1) * int(Ny / yGrid) + overlap, Ny)
    ndof = (iEnd - iStart) * (jEnd - jStart)
    nnz = ndof * 3 - (iEnd - iStart) - (jEnd - jStart)
    if not(sym):
        nnz = 2 * nnz - ndof
    sol = numpy.zeros(ndof, dtype = hpddm.scalar)
    f = numpy.empty_like(sol)
    xdim = [ 0.0, 10.0 ]
    ydim = [ 0.0, 10.0 ]
    Nf = 3
    xsc = [ 6.5, 2.0, 7.0 ]
    ysc = [ 8.0, 7.0, 3.0 ]
    rsc = [ 0.3, 0.3, 0.4 ]
    asc = [ 0.3, 0.2, -0.1 ]
    dx = (xdim[1] - xdim[0]) / float(Nx)
    dy = (ydim[1] - ydim[0]) / float(Ny)
    k = 0
    for j in xrange(jStart, jEnd):
        for i in xrange(iStart, iEnd):
            frs = 1.0
            for n in xrange(Nf):
                (xdist, ydist) = [ xdim[0] + dx * (i + 0.5) - xsc[n], ydim[0] + dy * (j + 0.5) - ysc[n] ]
                if math.sqrt(xdist * xdist + ydist * ydist) <= rsc[n]:
                    frs -= asc[n] * math.cos(0.5 * math.pi * xdist / rsc[n]) * math.cos(0.5 * math.pi * ydist / rsc[n])
                f[k] = frs
            k += 1
    d = numpy.ones(ndof, dtype = hpddm.underlying)
    o = []
    connectivity = []
    if jStart != 0:
        if iStart != 0:
            o.append(rankWorld - xGrid - 1)
            connectivity.append([None] * 4 * overlap * overlap)
            k = 0
            for j in xrange(2 * overlap):
                for i in xrange(iStart, iStart + 2 * overlap):
                    connectivity[-1][k] = i - iStart + (iEnd - iStart) * j
                    k += 1
            for j in xrange(overlap):
                for i in xrange(overlap - j):
                    d[i + j + j * (iEnd - iStart)] = j / float(overlap)
                for i in xrange(j):
                    d[i + j * (iEnd - iStart)] = i / float(overlap)
        else:
            for j in xrange(overlap):
                for i in xrange(overlap):
                    d[i + j * (iEnd - iStart)] = j / float(overlap)
        o.append(rankWorld - xGrid)
        connectivity.append([None] * 2 * overlap * (iEnd - iStart))
        k = 0
        for j in xrange(2 * overlap):
            for i in xrange(iStart, iEnd):
                connectivity[-1][k] = i - iStart + (iEnd - iStart) * j
                k += 1
        for j in xrange(overlap):
            for i in xrange(iStart + overlap, iEnd - overlap):
                d[i - iStart + (iEnd - iStart) * j] = j / float(overlap)
        if iEnd != Nx:
            o.append(rankWorld - xGrid + 1)
            connectivity.append([None] * 4 * overlap * overlap)
            k = 0
            for i in xrange(2 * overlap):
                for j in xrange(2 * overlap):
                    connectivity[-1][k] = (iEnd - iStart) * (i + 1) - 2 * overlap + j
                    k += 1
            for j in xrange(0, overlap):
                for i in xrange(overlap - j):
                    d[(iEnd - iStart) * (j + 1) - overlap + i] = j / float(overlap)
                for i in xrange(j):
                    d[(iEnd - iStart) * (j + 1) - i - 1] = i / float(overlap)
        else:
            for j in xrange(overlap):
                for i in xrange(overlap):
                    d[(iEnd - iStart) * (j + 1) - overlap + i] = j / float(overlap)
    if iStart != 0:
        o.append(rankWorld - 1)
        connectivity.append([None] * 2 * overlap * (jEnd - jStart))
        k = 0
        for i in xrange(jStart, jEnd):
            for j in xrange(2 * overlap):
                connectivity[-1][k] = j + (i - jStart) * (iEnd - iStart)
                k += 1
        for i in xrange(jStart + (jStart != 0) * overlap, jEnd - (jEnd != Ny) * overlap):
            for j in xrange(overlap):
                d[j + (i - jStart) * (iEnd - iStart)] = j / float(overlap)
    if iEnd != Nx:
        o.append(rankWorld + 1)
        connectivity.append([None] * 2 * overlap * (jEnd - jStart))
        k = 0
        for i in xrange(jStart, jEnd):
            for j in xrange(2 * overlap):
                connectivity[-1][k] = (iEnd - iStart) * (i + 1 - jStart) - 2 * overlap + j
                k += 1
        for i in xrange(jStart + (jStart != 0) * overlap, jEnd - (jEnd != Ny) * overlap):
            for j in xrange(overlap):
                d[(iEnd - iStart) * (i + 1 - jStart) - j - 1] = j / float(overlap)
    if jEnd != Ny:
        if iStart != 0:
            o.append(rankWorld + xGrid - 1)
            connectivity.append([None] * 4 * overlap * overlap)
            k = 0
            for j in xrange(2 * overlap):
                for i in xrange(iStart, iStart + 2 * overlap):
                    connectivity[-1][k] = ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j
                    k += 1
            for j in xrange(overlap):
                for i in xrange(overlap - j):
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * j] = i / float(overlap)
                for i in xrange(overlap - j, overlap):
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * j] = (overlap - 1 - j) / float(overlap)
        else:
            for j in xrange(overlap):
                for i in xrange(overlap):
                    d[ndof - overlap * (iEnd - iStart) + (iEnd - iStart) * j + i] = (overlap - j - 1) / float(overlap)
        o.append(rankWorld + xGrid);
        connectivity.append([None] * 2 * overlap * (iEnd - iStart))
        k = 0
        for j in xrange(2 * overlap):
            for i in xrange(iStart, iEnd):
                connectivity[-1][k] = ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j
                k += 1
        for j in xrange(overlap):
            for i in xrange(iStart + overlap, iEnd - overlap):
                d[ndof - overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j] = (overlap - 1 - j) / float(overlap)
        if iEnd != Nx:
            o.append(rankWorld + xGrid + 1)
            connectivity.append([None] * 4 * overlap * overlap)
            k = 0
            for j in xrange(2 * overlap):
                for i in xrange(iStart, iStart + 2 * overlap):
                    connectivity[-1][k] = ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j + (iEnd - iStart - 2 * overlap)
                    k += 1
            for j in xrange(overlap):
                for i in xrange(j, overlap):
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - 1 - i) / float(overlap)
                for i in xrange(j):
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - 1 - j) / float(overlap)
        else:
            for j in xrange(overlap):
                for i in xrange(overlap):
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - j - 1) / float(overlap)

    ia = numpy.empty(ndof + 1, dtype = ctypes.c_int)
    ja = numpy.empty(nnz, dtype = ctypes.c_int)
    a = numpy.empty(nnz, dtype = hpddm.scalar)
    ia[0] = (hpddm.numbering.value == b'F')
    ia[ndof] = nnz + (hpddm.numbering.value == b'F')
    if sym:
        k = 0
        nnz = 0
        for j in xrange(jStart, jEnd):
            for i in xrange(iStart, iEnd):
                if j > jStart:
                    a[nnz] = -1 / (dy * dy)
                    ja[nnz] = k - int(Ny / yGrid) + (hpddm.numbering.value == b'F')
                    nnz += 1
                if i > iStart:
                    a[nnz] = -1 / (dx * dx)
                    ja[nnz] = k - (hpddm.numbering.value == b'C')
                    nnz += 1
                a[nnz] = 2 / (dx * dx) + 2 / (dy * dy)
                ja[nnz] = k + (hpddm.numbering.value == b'F')
                nnz += 1
                k += 1
                ia[k] = nnz + (hpddm.numbering.value == b'F')
    else:
        k = 0
        nnz = 0
        for j in xrange(jStart, jEnd):
            for i in xrange(iStart, iEnd):
                if j > jStart:
                    a[nnz] = -1 / (dy * dy)
                    ja[nnz] = k - int(Ny / yGrid) + (hpddm.numbering.value == b'F')
                    nnz += 1
                if i > iStart:
                    a[nnz] = -1 / (dx * dx)
                    ja[nnz] = k - (hpddm.numbering.value == b'C')
                    nnz += 1
                a[nnz] = 2 / (dx * dx) + 2 / (dy * dy)
                ja[nnz] = k + (hpddm.numbering.value == b'F')
                nnz += 1
                if i < iEnd - 1:
                    a[nnz] = -1 / (dx * dx)
                    ja[nnz] = k + 1 + (hpddm.numbering.value == b'F')
                    nnz += 1
                if j < jEnd - 1:
                    a[nnz] = -1 / (dy * dy)
                    ja[nnz] = k + int(Ny / yGrid) + (hpddm.numbering.value == b'F')
                    nnz += 1
                k += 1
                ia[k] = nnz + (hpddm.numbering.value == b'F')
    if sizeWorld > 1 and hpddm.optionSet(opt, b'schwarz_coarse_correction') and hpddm.optionVal(opt, b'geneo_nu') > 0:
        if sym:
            nnzNeumann = 2 * nnz - ndof
            iNeumann = numpy.empty_like(ia)
            jNeumann = numpy.empty(nnzNeumann, dtype = ctypes.c_int)
            aNeumann = numpy.empty(nnzNeumann, dtype = hpddm.scalar)
            iNeumann[0] = (hpddm.numbering.value == b'F')
            iNeumann[ndof] = nnzNeumann + (hpddm.numbering.value == b'F')
            k = 0
            nnzNeumann = 0
            for j in xrange(jStart, jEnd):
                for i in xrange(iStart, iEnd):
                    if j > jStart:
                        aNeumann[nnzNeumann] = -1 / (dy * dy) + (-1 / (dx * dx) if i == iStart else 0)
                        jNeumann[nnzNeumann] = k - int(Ny / yGrid) + (hpddm.numbering.value == b'F')
                        nnzNeumann += 1
                    if i > iStart:
                        aNeumann[nnzNeumann] = -1 / (dx * dx) + (-1 / (dy * dy) if j == jStart else 0)
                        jNeumann[nnzNeumann] = k - (hpddm.numbering.value == b'C')
                        nnzNeumann += 1
                    aNeumann[nnzNeumann] = 2 / (dx * dx) + 2 / (dy * dy)
                    jNeumann[nnzNeumann] = k + (hpddm.numbering.value == b'F')
                    nnzNeumann += 1
                    if i < iEnd - 1:
                        aNeumann[nnzNeumann] = -1 / (dx * dx) + (-1 / (dy * dy) if j == jEnd - 1 else 0)
                        jNeumann[nnzNeumann] = k + 1 + (hpddm.numbering.value == b'F')
                        nnzNeumann += 1
                    if j < jEnd - 1:
                        aNeumann[nnzNeumann] = -1 / (dy * dy) + (-1 / (dx * dx) if i == iEnd - 1 else 0)
                        jNeumann[nnzNeumann] = k + int(Ny / yGrid) + (hpddm.numbering.value == b'F')
                        nnzNeumann += 1
                    k += 1
                    iNeumann[k] = nnzNeumann + (hpddm.numbering.value == b'F')
            MatNeumann = hpddm.matrixCSRCreate(ndof, ndof, nnzNeumann, aNeumann, iNeumann, jNeumann, False)
        else:
            aNeumann = numpy.empty_like(a)
            numpy.copyto(aNeumann, a)
            nnzNeumann = 0
            for j in xrange(jStart, jEnd):
                for i in xrange(iStart, iEnd):
                    if j > jStart:
                        if i == iStart:
                            aNeumann[nnzNeumann] -= 1 / (dx * dx)
                        nnzNeumann += 1
                    if i > iStart:
                        if j == jStart:
                            aNeumann[nnzNeumann] -= 1 / (dy * dy)
                        nnzNeumann += 1
                    nnzNeumann += 1
                    if i < iEnd - 1:
                        if j == jEnd - 1:
                            aNeumann[nnzNeumann] -= 1 / (dy * dy)
                        nnzNeumann += 1
                    if j < jEnd - 1:
                        if i == iEnd - 1:
                            aNeumann[nnzNeumann] -= 1 / (dx * dx)
                        nnzNeumann += 1
            MatNeumann = hpddm.matrixCSRCreate(ndof, ndof, nnzNeumann, aNeumann, ia, ja, False)
    else:
        MatNeumann = ctypes.POINTER(hpddm.MatrixCSR)()
    Mat = hpddm.matrixCSRCreate(ndof, ndof, nnz, a, ia, ja, sym)
    if sizeWorld > 1:
        for k in xrange(sizeWorld):
            if k == rankWorld:
                print(str(rankWorld) + ':')
                print(str(x) + 'x' + str(y) + ' -- [' + str(iStart) + ' ; ' + str(iEnd) + '] x [' + str(jStart) + ' ; ' + str(jEnd) + '] -- ' + str(ndof) + ', ' + str(nnz))
            MPI.COMM_WORLD.Barrier()
    return o, connectivity, ndof, Mat, MatNeumann, d, f, sol
