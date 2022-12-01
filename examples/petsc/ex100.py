#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2019-02-21

   Copyright (C) 2019-     Centre National de la Recherche Scientifique

   Note:      Reference PETSc implementation available at
                                                       https://bit.ly/2U5VFfL
              Contributed by
                                                              Lisandro Dalcin

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

def RunTest():

    from petsc4py import PETSc
    import example100
    import hpddm

    OptDB = PETSc.Options()
    N     = OptDB.getInt('N', 100)
    draw  = OptDB.getBool('draw', False)
    hpddm.registerKSP()
    OptDB.setValue('ksp_type', 'hpddm')
    hpddm.optionParse(hpddm.optionGet(), '-hpddm_krylov_method gcrodr -hpddm_recycle 10 -hpddm_verbosity')

    A = PETSc.Mat()
    A.create(comm = PETSc.COMM_WORLD)
    A.setSizes([N, N])
    A.setType(PETSc.Mat.Type.PYTHON)
    A.setPythonContext(example100.Laplace1D())
    A.setUp()

    x, b = A.getVecs()
    b.set(1)

    ksp = PETSc.KSP()
    ksp.create(comm = PETSc.COMM_WORLD)
    ksp.setType(PETSc.KSP.Type.PYTHON)
    ksp.setPythonContext(example100.ConjGrad())

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(example100.Jacobi())

    ksp.setOperators(A, A)
    ksp.setFromOptions()
    ksp.solve(b, x)

    r = b.duplicate()
    A.mult(x, r)
    r.aypx(-1, b)
    rnorm = r.norm()
    PETSc.Sys.Print('error norm = %g' % rnorm,
                    comm = PETSc.COMM_WORLD)

    if draw:
        viewer = PETSc.Viewer.DRAW(x.getComm())
        x.view(viewer)
        PETSc.Sys.sleep(2)

    ksp.solve(b, x)
    A.mult(x, r)
    r.aypx(-1, b)
    rnorm = r.norm()
    PETSc.Sys.Print('error norm = %g' % rnorm,
                    comm = PETSc.COMM_WORLD)

if __name__ == '__main__':
    import sys, petsc4py
    petsc4py.init(sys.argv)
    RunTest()
