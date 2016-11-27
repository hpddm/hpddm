/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2012-12-15

   Copyright (C) 2011-2014 Université de Grenoble
                 2015      Eidgenössische Technische Hochschule Zürich
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
 */

#ifndef _HPDDM_EIGENSOLVER_
#define _HPDDM_EIGENSOLVER_

namespace HPDDM {
/* Class: Eigensolver
 *
 *  A base class used to interface eigenvalue problem solvers such as <Arpack> or <Lapack>.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
class Eigensolver {
    protected:
        /* Variable: tol
         *  Relative tolerance of the eigenvalue problem solver. */
        underlying_type<K>       _tol;
        /* Variable: threshold
         *  Threshold criterion. */
        underlying_type<K> _threshold;
        /* Variable: n
         *  Number of rows of the eigenvalue problem. */
        int                        _n;
    public:
        /* Variable: nu
         *  Number of desired eigenvalues. */
        int                       _nu;
        Eigensolver(int n)                                                               : _tol(), _threshold(), _n(n), _nu() { }
        Eigensolver(int n, int nu)                                                       : _tol(Option::get()->val("eigensolver_tol", 1.0e-6)), _threshold(), _n(n), _nu(std::max(1, std::min(nu, n))) { }
        Eigensolver(underlying_type<K> threshold, int n, int nu)                         : _tol(threshold > 0.0 ? HPDDM_EPS : Option::get()->val("eigensolver_tol", 1.0e-6)), _threshold(threshold), _n(n), _nu(std::max(1, std::min(nu, n))) { }
        Eigensolver(underlying_type<K> tol, underlying_type<K> threshold, int n, int nu) : _tol(threshold > 0.0 ? HPDDM_EPS : tol), _threshold(threshold), _n(n), _nu(std::max(1, std::min(nu, n))) { }
        void dump(const K* const eigenvalues, const K* const* const eigenvectors, const MPI_Comm& communicator) const {
            int rankWorld;
            MPI_Comm_rank(communicator, &rankWorld);
            const Option& opt = *Option::get();
            std::string filename = opt.prefix("dump_local_eigenvectors", true);
            if(filename.size() == 0)
                filename = opt.prefix("dump_local_eigenvectors_" + to_string(rankWorld), true);
            if(filename.size() != 0) {
                int sizeWorld;
                MPI_Comm_size(communicator, &sizeWorld);
                std::ofstream output { filename + "_" + to_string(rankWorld) + "_" + to_string(sizeWorld) + ".txt" };
                output << std::scientific;
                for(unsigned short col = 0; col < _nu; col += 5) {
                    for(unsigned short i = col; i < std::min(col + 5, _nu); ++i)
                        output << std::setw(13) << i + 1 << "\t";
                    output << "\n";
                    for(unsigned short i = col; i < std::min(col + 5, _nu); ++i)
                        output << std::setw(13) << eigenvalues[i] << "\t";
                    output << "\n\n";
                    for(unsigned int j = 0; j < _n; ++j) {
                        for(unsigned short i = col; i < std::min(col + 5, _nu); ++i)
                            output << std::setw(13) << eigenvectors[i][j] << "\t";
                        output << "\n";
                    }
                    output << "\n\n";
                }
            }
        }
        /* Function: selectNu
         *
         *  Computes a uniform threshold criterion.
         *
         * Parameters:
         *    eigenvalues   - Input array used to store eigenvalues in ascending order.
         *    communicator  - MPI communicator (usually <Subdomain::communicator>) on which the criterion <Eigensolver::nu> has to be uniformized. */
        template<class T, bool min = false>
        void selectNu(const T* const eigenvalues, const MPI_Comm& communicator, unsigned short m = 0) {
            static_assert(std::is_same<T, K>::value || std::is_same<T, underlying_type<K>>::value, "Wrong types");
            unsigned short nev = _nu ? std::min(static_cast<int>(std::distance(eigenvalues, std::upper_bound(eigenvalues, eigenvalues + _nu, _threshold, [](const T& lhs, const T& rhs) { return std::real(lhs) < std::real(rhs); }))), _nu) : std::numeric_limits<unsigned short>::max();
            if(Option::get()->val<char>("geneo_force_uniformity", 0)) {
                if(!min)
                    MPI_Allreduce(MPI_IN_PLACE, &nev, 1, MPI_UNSIGNED_SHORT, MPI_MIN, communicator);
                else {
                    unsigned short r[2] = { nev, static_cast<unsigned short>(std::numeric_limits<unsigned short>::max() - m) };
                    MPI_Allreduce(MPI_IN_PLACE, r, 2, MPI_UNSIGNED_SHORT, MPI_MIN, communicator);
                    nev = std::max(r[0], static_cast<unsigned short>(std::numeric_limits<unsigned short>::max() - r[1]));
                }
            }
            _nu = std::min(_nu, static_cast<int>(nev));
        }
        /* Function: getTol
         *  Returns the value of <Eigensolver::tol>. */
        underlying_type<K> getTol() const { return _tol; }
};
} // HPDDM
#endif // _HPDDM_EIGENSOLVER_
