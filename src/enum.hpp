/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@inf.ethz.ch>
        Date: 2012-10-04

   Copyright (C) 2011-2014 Université de Grenoble
                 2015      Eidgenössische Technische Hochschule Zürich

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
 */

#ifndef _ENUM_
#define _ENUM_

namespace HPDDM {
/* Enum: Parameter
 *
 *  Parameters for the construction of a distributed matrix.
 *
 *    NU             - Number of eigenvalues on current subdomain.
 *    P              - Number of master processes.
 *    TOPOLOGY       - Distribution of the matrix.
 *    DISTRIBUTION   - Controls whether right-hand sides and solution vectors should be distributed or not.
 *    STRATEGY       - Strategy of the direct solver for the analysis phase.
 *
 * See also: <DMatrix>. */
enum Parameter : char {
    NU, P, TOPOLOGY, DISTRIBUTION, STRATEGY
};
/* Enum: Gmres
 *
 *  Defines the type of GMRES used.
 *
 *    CLASSICAL      - GMRES with classical Gram-Schmidt process to orthogonalize against the Krylov space.
 *    MODIFIED       - GMRES with modified Gram-Schmidt process to orthogonalize against the Krylov space.
 *    PIPELINED      - Pipelined GMRES.
 *    FUSED          - Fused pipelined GMRES.
 *
 * See also: <Iterative method::GMRES>. */
enum Gmres : char {
    CLASSICAL, MODIFIED, PIPELINED, FUSED
};
/* Enum: FetiPrcndtnr
 *
 *  Defines the FETI preconditioner used in the projection.
 *
 * NONE         - No preconditioner.
 * SUPERLUMPED  - Approximation of the local Schur complement by the diagonal of <Schur::bb>.
 * LUMPED       - Approximation of the local Schur complement by <Schur::bb>.
 * DIRICHLET    - Local Schur complement.
 *
 * See also: <Feti>. */
enum class FetiPrcndtnr : char {
    NONE, SUPERLUMPED, LUMPED, DIRICHLET
};
} // HPDDM
#endif // _ENUM_
