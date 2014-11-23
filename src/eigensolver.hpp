/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
        Date: 2012-12-15

   Copyright (C) 2011-2014 Université de Grenoble

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

#ifndef _EIGENSOLVER_
#define _EIGENSOLVER_

namespace HPDDM {
/* Class: Eigensolver
 *
 *  A base class used to interface eigenvalue problem solvers such as <Arpack> or <Lapack>. */
class Eigensolver {
    protected:
        /* Variable: tol
         *  Relative tolerance of the eigenvalue problem solver. */
        double                     _tol;
        /* Variable: threshold
         *  Threshold criterion. */
        double               _threshold;
        /* Variable: n
         *  Number of rows of the eigenvalue problem. */
        int                          _n;
        /* Variable: nu
         *  Number of desired eigenvalues. */
        int                         _nu;
    public:
        Eigensolver(int n, int& nu)                               : _tol(1.0e-6), _threshold(0.0), _n(n), _nu(std::max(1, std::min(nu, n / 4))) { nu = _nu; }
        Eigensolver(double threshold, int n, int& nu)             : _tol(1.0e-6), _threshold(threshold), _n(n), _nu(std::max(1, std::min(nu, n / 4))) { nu = _nu; }
        Eigensolver(double tol, double threshold, int n, int& nu) : _tol(tol), _threshold(threshold), _n(n), _nu(std::max(1, std::min(nu, n / 4))) { nu = _nu; }
        /* Function: selectNu
         *
         *  Computes a uniform threshold criterion.
         *
         * Parameters:
         *    eigenvalues   - Input array used to store eigenvalues in ascending order.
         *    communicator  - MPI communicator (usually <Subdomain::communicator>) on which the criterion <Eigensolver::nu> has to be uniformized. */
        template<class K>
        inline void selectNu(const K* const eigenvalues, const MPI_Comm& communicator) {
            unsigned short nevThreshold = std::min(static_cast<int>(std::distance(eigenvalues, std::upper_bound(eigenvalues, eigenvalues + _nu, _threshold, [](const K& lhs, const K& rhs) { return std::real(lhs) < std::real(rhs); }))), _nu);
            MPI_Allreduce(MPI_IN_PLACE, &nevThreshold, 1, MPI_UNSIGNED_SHORT, MPI_MAX, communicator);
            _nu = nevThreshold;
        }
        /* Function: getTol
         *  Returns the value of <Eigensolver::tol>. */
        inline double getTol() const { return _tol; }
        /* Function: getNu
         *  Returns the value of <Eigensolver::nu>. */
        inline int getNu() const { return _nu; }
};
} // HPDDM
#endif // _EIGENSOLVER_
