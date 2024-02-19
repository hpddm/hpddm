 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2016-01-06

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
 */

#ifndef HPDDM_ORTHODIR_HPP_
#define HPDDM_ORTHODIR_HPP_

#include "HPDDM_iterative.hpp"

namespace HPDDM {
template<bool excluded, class Operator, class K>
inline int IterativeMethod::ORTHODIR(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
    const Option& opt = *Option::get();
    const int n = excluded ? 0 : A.getDof();
    const unsigned short it = opt["max_it"];
    underlying_type<K> tol = opt["tol"];
    const char verbosity = opt.val<char>("verbosity");
    if(std::abs(tol) < std::numeric_limits<underlying_type<K>>::epsilon()) {
        if(verbosity > 0)
            std::cout << "WARNING -- the tolerance of the iterative method was set to " << tol << " which is lower than the machine epsilon for type " << demangle(typeid(underlying_type<K>).name()) << ", forcing the tolerance to " << 2 * std::numeric_limits<underlying_type<K>>::epsilon() << std::endl;
        tol = 2 * std::numeric_limits<underlying_type<K>>::epsilon();
    }
    const unsigned short m = std::min(static_cast<unsigned short>(std::numeric_limits<short>::max()), std::min(static_cast<unsigned short>(opt["gmres_restart"]), it));
    const char variant = (opt["variant"] == 0 ? 'L' : opt["variant"] == 1 ? 'R' : 'F');
    
    K** const v = new K*[(m + 1) * 2];
    K** const av = v + m;
    K* const s = new K[mu * ((m + 1) * (m + 1) + n * (4 + 2 * m) + (!Wrapper<K>::is_complex ? m + 1 : (m + 2) / 2))];
    K* const Ax = s + mu * (m + 1);
    K* const r = Ax + mu * n;
    *v = r + mu * n;
    for(unsigned short i = 1; i < m + 1; ++i)
        v[i] = *v + i * mu * n;
    *av = *v + (m + 1) * mu * n;
    for(unsigned short i = 1; i < m + 1; ++i)
        av[i] = *av + i * mu * n;
    K* const alpha = *av + (m + 1) * mu * n;
    
    underlying_type<K>* const norm = reinterpret_cast<underlying_type<K>*>(alpha + (m + 1) * mu);
    underlying_type<K>* const sn = norm + mu;
    // bool alloc = A.setBuffer();
    short* const hasConverged = new short[mu];
    std::fill_n(hasConverged, mu, -m);
    
    A.template start<excluded>(b, x, mu);
    
    A.template apply<excluded>(b, *v, mu, Ax);
    
    for(unsigned short nu = 0; nu < mu; ++nu)
        norm[nu] = std::real(Blas<K>::dot(&n, *v + nu * n, &i__1, *v + nu * n, &i__1));
    
    if(!excluded)
        for(unsigned short nu = 0; nu < mu; ++nu)
            for(unsigned int i = 0; i < n; ++i)
                if(std::abs(b[nu * n + i]) > HPDDM_PEN * HPDDM_EPS)
                    ;// depenalize(b[nu * n + i], x[nu * n + i]);
    
    unsigned short j = 1;
    
    while(j <= it) {
        
        if(!excluded)
            A.GMV(x,  Ax , mu);
        Blas<K>::axpby(mu * n, 1.0, b, 1, -1.0, Ax , 1);
        A.template apply<excluded>(Ax, *v, mu);
        
        std::copy_n(*v, mu * n, r);
        
        if(!excluded)
            A.GMV(*v,  Ax , mu);
        A.template apply<excluded>(Ax, *av, mu);
        
        for(unsigned short nu = 0; nu < mu; ++nu)
            sn[nu] = std::real(Blas<K>::dot(&n, *av + nu * n, &i__1, *av + nu * n, &i__1));
        if(j == 1) {
            MPI_Allreduce(MPI_IN_PLACE, norm, 2 * mu, Wrapper<underlying_type<K>>::mpi_type(), MPI_SUM, comm);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                norm[nu] = std::sqrt(norm[nu]);
                if(norm[nu] < HPDDM_EPS)
                    norm[nu] = 1.0;
                if(tol > 0.0 && sn[nu] / norm[nu] < tol) {
                    if(norm[nu] > 1.0 / HPDDM_EPS)
                        norm[nu] = 1.0;
                    else
                        hasConverged[nu] = 0;
                }
                else if(sn[nu] < -tol)
                    hasConverged[nu] = 0;
            }
        }
        else
            MPI_Allreduce(MPI_IN_PLACE, sn, mu, Wrapper<underlying_type<K>>::mpi_type(), MPI_SUM, comm);
        for(unsigned short nu = 0; nu < mu; ++nu) {
            if(hasConverged[nu] > 0)
                hasConverged[nu] = 0;
            s[nu] = std::sqrt(sn[nu]);
            std::for_each(*v + nu * n, *v + (nu + 1) * n, [&](K& y) { y /= s[nu]; });
            std::for_each(*av + nu * n, *av + (nu + 1) * n, [&](K& y) { y /= s[nu]; });
        }
        
        int i = 0;
        while(i < m && j <= it) {
            
            std::copy_n(av[i], mu * n, v[i + 1]);
            
            if(!excluded)
                A.GMV(v[i + 1], Ax, mu);
            A.template apply<excluded>(Ax, av[i + 1], mu);
            
            if(opt["gs"] == 1)
                for(unsigned short k = 0; k < i + 1; ++k) {
                    for(unsigned short nu = 0; nu < mu; ++nu)
                        alpha[nu] = Blas<K>::dot(&n, av[k] + nu * n, &i__1, av[i + 1] + nu * n, &i__1);
                    MPI_Allreduce(MPI_IN_PLACE, alpha, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                    std::for_each(alpha, alpha + mu, [&](K& y) { y = -y; });
                    
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        Blas<K>::axpy(&n, alpha + nu, v[k] + nu * n, &i__1, v[i + 1] + nu * n, &i__1);
                        Blas<K>::axpy(&n, alpha + nu, av[k] + nu * n, &i__1, av[i + 1] + nu * n, &i__1);
                    }
                }
            else {
                int tmp[2] { i + 1, mu * n };
                for(unsigned short nu = 0; nu < mu; ++nu)
                    Blas<K>::gemv(&(Wrapper<K>::transc), &n, tmp, &(Wrapper<K>::d__1), *av + nu * n, tmp + 1, av[i + 1] + nu * n, &i__1, &(Wrapper<K>::d__0), alpha + nu, &mu);
                MPI_Allreduce(MPI_IN_PLACE, alpha, (i + 1) * mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
                if(opt["gs"] == 0)
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        Blas<K>::gemv("N", &n, tmp, &(Wrapper<K>::d__2), *v + nu * n, tmp + 1, alpha + nu, &mu, &(Wrapper<K>::d__1), v[i + 1] + nu * n, &i__1);
                        Blas<K>::gemv("N", &n, tmp, &(Wrapper<K>::d__2), *av + nu * n, tmp + 1, alpha + nu, &mu, &(Wrapper<K>::d__1), av[i + 1] + nu * n, &i__1);
                    }
                else
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        Blas<K>::axpby(n, -alpha[i * mu + nu], v[i] + nu * n, 1, 1.0, v[i + 1] + nu * n, 1);
                        Blas<K>::axpby(n, -alpha[i * mu + nu], av[i] + nu * n, 1, 1.0, av[i + 1] + nu * n, 1);
                    }
            }
            
            for(unsigned short nu = 0; nu < mu; ++nu)
                sn[i * mu + nu] = std::real(Blas<K>::dot(&n, av[i + 1] + nu * n, &i__1, av[i + 1] + nu * n, &i__1));
            MPI_Allreduce(MPI_IN_PLACE, sn + i * mu, mu, Wrapper<underlying_type<K>>::mpi_type(), MPI_SUM, comm);
            
            for(unsigned short nu = 0; nu < mu; ++nu) {
                s[nu] = std::sqrt(sn[i * mu + nu]);
                std::for_each(v[i + 1] + nu * n, v[i + 1] + (nu + 1) * n, [&](K& y) { y /= s[nu]; });
                std::for_each(av[i + 1] + nu * n, av[i + 1] + (nu + 1) * n, [&](K& y) { y /= s[nu]; });
            }
            
            for(unsigned short nu = 0; nu < mu; ++nu)
                alpha[nu] = Blas<K>::dot(&n, av[i] + nu * n, &i__1, r + nu * n, &i__1);
            MPI_Allreduce(MPI_IN_PLACE, alpha, mu, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            
            for(unsigned short nu = 0; nu < mu; ++nu)
                Blas<K>::axpy(&n, alpha + nu, v[i] + nu * n, &i__1, x + nu * n, &i__1);
            
            std::for_each(alpha, alpha + mu, [&](K& y) { y = -y; });
            
            for(unsigned short nu = 0; nu < mu; ++nu)
                Blas<K>::axpy(&n, alpha + nu, av[i] + nu * n, &i__1, r + nu * n, &i__1);
            
            
            for(unsigned short nu = 0; nu < mu; ++nu) {
                
                s[(i + 1) * mu + nu] = std::real(Blas<K>::dot(&n, r + nu * n, &i__1, r + nu * n, &i__1));
            }
            MPI_Allreduce(MPI_IN_PLACE, s + (i + 1) * mu, mu, Wrapper<underlying_type<K>>::mpi_type(), MPI_SUM, comm);
            for(unsigned short nu = 0; nu < mu; ++nu) {
                s[(i + 1) * mu + nu] = std::sqrt(s[(i + 1) * mu + nu]);
                if(hasConverged[nu] == -m && ((tol > 0 && std::abs(s[(i + 1) * mu + nu]) / norm[nu] <= tol) || (tol < 0 && std::abs(s[(i + 1) * mu + nu]) <= -tol)))
                    hasConverged[nu] = i + 1;
            }
            if(verbosity > 0) {
                int tmp[2] { 0, 0 };
                underlying_type<K> beta = std::abs(s[(i + 1) * mu]);
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    if(hasConverged[nu] != -m)
                        ++tmp[0];
                    else if(std::abs(s[(i + 1) * mu + nu]) > beta) {
                        beta = std::abs(s[(i + 1) * mu + nu]);
                        tmp[1] = nu;
                    }
                }
                if(tol > 0)
                    std::cout << "ORTHODIR: " << std::setw(3) << j << " " << std::scientific << beta << " " <<  norm[tmp[1]] << " " <<  beta / norm[tmp[1]] << " < " << tol;
                else
                    std::cout << "ORTHODIR: " << std::setw(3) << j << " " << std::scientific << beta << " < " << -tol;
                if(mu > 1) {
                    std::cout << " (rhs #" << tmp[1] + 1;
                    if(tmp[0] > 0)
                        std::cout << ", " << tmp[0] << " converged rhs";
                    std::cout << ")";
                }
                std::cout << std::endl;
            }
            if(std::find(hasConverged, hasConverged + mu, -m) == hasConverged + mu)
                break;
            else
                ++i, ++j;
        }
        if(j != it + 1 && i == m) {
            std::fill_n(s + mu, mu * m, K());
            if(verbosity > 0)
                std::cout << "ORTHODIR restart(" << m << ")" << std::endl;
        }
        else
            break;
    }
    if(verbosity > 0) {
        if(j != it + 1)
            std::cout << "ORTHODIR converges after " << j << " iteration" << (j > 1 ? "s" : "") << std::endl;
        else
            std::cout << "ORTHODIR does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
    }
    
    delete [] hasConverged;
    // A.clearBuffer(alloc);
    delete [] s;
    return std::min(j, it);
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::BORTHODIR(const Operator&, const K* const, K* const, const int&, const MPI_Comm&) {
    return 0;
}
} // HPDDM
#endif // HPDDM_ORTHODIR_HPP_
