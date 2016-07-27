 /*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-12-21

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
 */

#ifndef _HPDDM_CG_
#define _HPDDM_CG_

#include "iterative.hpp"

namespace HPDDM {
template<bool excluded, class Operator, class K>
inline int IterativeMethod::CG(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
    const Option& opt = *Option::get();
    if(opt.any_of("schwarz_method", { 0, 1, 4 }) || opt.any_of("schwarz_coarse_correction", { 0 }))
        return GMRES(A, b, x, mu, comm);
    const int n = excluded ? 0 : A.getDof();
    const int dim = n * mu;
    const unsigned short it = std::min(opt.val<short>("max_it", 100), std::numeric_limits<short>::max());
    underlying_type<K> tol = opt.val("tol", 1.0e-6);
    const char verbosity = opt.val<char>("verbosity", 0);
    std::cout << std::scientific;
    epsilon(tol, verbosity);
    underlying_type<K>* res;
    K* trash;
    allocate(res, trash, n, opt.val<char>("variant", 0) == 2 ? 2 : 1, it, mu);
    bool allocate = A.setBuffer();
    short* const hasConverged = new short[mu];
    std::fill_n(hasConverged, mu, -it);

    underlying_type<K>* const dir = res + mu;
    K* const z = trash + dim;
    K* const r = z + dim;
    K* const p = r + dim;
    const underlying_type<K>* const d = A.getScaling();

    A.template start<excluded>(b, x, mu);
    if(!excluded)
        A.GMV(x, z, mu);
    std::copy_n(b, dim, r);
    Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), z, &i__1, r, &i__1);

    A.template apply<excluded>(r, p, mu, z);

    Wrapper<K>::diag(n, d, p, trash, mu);
    for(unsigned short nu = 0; nu < mu; ++nu)
        dir[nu] = std::real(Blas<K>::dot(&n, trash + n * nu, &i__1, p + n * nu, &i__1));
    MPI_Allreduce(MPI_IN_PLACE, dir, mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
    std::transform(dir, dir + mu, res, [](const underlying_type<K>& d) { return std::sqrt(d); });

    int i = 0;
    while(i <= it && std::find(hasConverged, hasConverged + mu, -it) != hasConverged + mu) {
        for(unsigned short nu = 0; nu < mu; ++nu)
            dir[nu] = std::real(Blas<K>::dot(&n, r + n * nu, &i__1, trash + n * nu, &i__1));
        if(opt.val<char>("variant", 0) == 2 && i) {
            for(unsigned short k = 0; k < i; ++k)
                for(unsigned short nu = 0; nu < mu; ++nu)
                    dir[mu + k * mu + nu] = -std::real(Blas<K>::dot(&n, trash + n * nu, &i__1, p + (1 + it + k) * dim + n * nu, &i__1)) / dir[mu + (it + k) * mu + nu];
            MPI_Allreduce(MPI_IN_PLACE, dir + mu, i * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
            if(!excluded && n) {
                std::copy_n(z, dim, p);
                for(unsigned short nu = 0; nu < mu; ++nu) {
                    for(unsigned short k = 0; k < i; ++k)
                        trash[k] = dir[mu + k * mu + nu];
                    Blas<K>::gemv("N", &n, &i, &(Wrapper<K>::d__1), p + dim + n * nu, &dim, trash, &i__1, &(Wrapper<K>::d__1), p + nu * n, &i__1);
                }
            }
        }
        if(!excluded)
            A.GMV(p, z, mu);
        if(i) {
            Wrapper<K>::diag(n, d, z, trash, mu);
            for(unsigned short k = 0; k < i; ++k)
                for(unsigned short nu = 0; nu < mu; ++nu)
                    dir[mu + k * mu + nu] = -std::real(Blas<K>::dot(&n, trash + n * nu, &i__1, p + (1 + k) * dim + n * nu, &i__1)) / dir[mu + (it + k) * mu + nu];
            MPI_Allreduce(MPI_IN_PLACE, dir + mu, i * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
            if(!excluded) {
                if(n)
                    for(unsigned short nu = 0; nu < mu; ++nu) {
                        for(unsigned short k = 0; k < i; ++k)
                            trash[k] = dir[mu + k * mu + nu];
                        Blas<K>::gemv("N", &n, &i, &(Wrapper<K>::d__1), p + dim + n * nu, &dim, trash, &i__1, &(Wrapper<K>::d__1), p + nu * n, &i__1);
                    }
                A.GMV(p, z, mu);
            }
        }
        Wrapper<K>::diag(n, d, p, trash, mu);
        for(unsigned short nu = 0; nu < mu; ++nu)
            dir[mu + nu] = std::real(Blas<K>::dot(&n, z + n * nu, &i__1, trash + n * nu, &i__1));
        MPI_Allreduce(MPI_IN_PLACE, dir, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
        ++i;
        std::copy_n(dir + mu, mu, dir + (it + i) * mu);
        std::copy_n(p, dim, p + i * dim);
        if(opt.val<char>("variant", 0) == 2)
            std::copy_n(z, dim, p + (it + i) * dim);
        for(unsigned short nu = 0; nu < mu; ++nu) {
            if(hasConverged[nu] == -it) {
                trash[nu] = dir[nu] / dir[mu + nu];
                Blas<K>::axpy(&n, trash + nu, p + n * nu, &i__1, x + n * nu, &i__1);
                trash[nu] = -trash[nu];
                Blas<K>::axpy(&n, trash + nu, z + n * nu, &i__1, r + n * nu, &i__1);
            }
        }
        A.template apply<excluded>(r, z, mu, trash);
        Wrapper<K>::diag(n, d, z, trash, mu);
        for(unsigned short nu = 0; nu < mu; ++nu) {
            dir[mu + nu] = std::real(Blas<K>::dot(&n, r + n * nu, &i__1, trash + n * nu, &i__1)) / dir[nu];
            dir[nu] = std::real(Blas<K>::dot(&n, z + n * nu, &i__1, trash + n * nu, &i__1));
        }
        MPI_Allreduce(MPI_IN_PLACE, dir, 2 * mu, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
        if(opt.val<char>("variant", 0) != 2)
            for(unsigned short nu = 0; nu < mu; ++nu)
                Blas<K>::axpby(n, 1.0, z + n * nu, 1, dir[mu + nu], p + n * nu, 1);
        std::for_each(dir, dir + mu, [](underlying_type<K>& d) { d = std::sqrt(d); });
        for(unsigned short nu = 0; nu < mu; ++nu) {
            if(hasConverged[nu] == -it && ((tol > 0.0 && dir[nu] / res[nu] <= tol) || (tol < 0.0 && dir[nu] <= -tol)))
                hasConverged[nu] = i;
        }
        if(verbosity > 2)
            outputResidual<2>(i, tol, mu, res, dir, hasConverged, it);
    }
    if(verbosity) {
        if(i != it + 1)
            std::cout << "CG converges after " << i << " iteration" << (i > 1 ? "s" : "") << std::endl;
        else
            std::cout << "CG does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
    }
    delete [] res;
    if(Wrapper<K>::is_complex)
        delete [] trash;
    delete [] hasConverged;
    A.clearBuffer(allocate);
    std::cout.unsetf(std::ios_base::scientific);
    return std::min(static_cast<unsigned short>(i), it);
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::BCG(const Operator& A, const K* const b, K* const x, const int& mu, const MPI_Comm& comm) {
    const Option& opt = *Option::get();
    if(opt.any_of("schwarz_method", { 0, 1, 4 }) || opt.any_of("schwarz_coarse_correction", { 0 }))
        return GMRES(A, b, x, mu, comm);
    if(opt.val<char>("variant", 0) == 2)
        return CG(A, b, x, mu, comm);
    const int n = excluded ? 0 : A.getDof();
    const int dim = n * mu;
    const unsigned short it = std::min(opt.val<short>("max_it", 100), std::numeric_limits<short>::max());
    underlying_type<K> tol = opt.val("tol", 1.0e-6);
    const char verbosity = opt.val<char>("verbosity", 0);
    std::cout << std::scientific;
    epsilon(tol, verbosity);
    bool allocate = A.setBuffer();
    K* const trash = new K[4 * (dim + mu * mu)];
    K* const p = trash + dim;
    K* const z = p + dim;
    K* const r = z + dim;
    K* const rho = r + dim;
    K* const rhs = rho + 2 * mu * mu;
    K* const gamma = rhs + mu * mu;
    const underlying_type<K>* const d = A.getScaling();

    A.template start<excluded>(b, x, mu);
    if(!excluded)
        A.GMV(x, z, mu);
    std::copy_n(b, dim, r);
    Blas<K>::axpy(&dim, &(Wrapper<K>::d__2), z, &i__1, r, &i__1);

    A.template apply<excluded>(r, p, mu, z);
    Wrapper<K>::diag(n, d, p, trash, mu);
    if(!excluded && n) {
        Blas<K>::gemmt("U", &(Wrapper<K>::transc), "N", &mu, &n, &(Wrapper<K>::d__1), r, &n, trash, &n, &(Wrapper<K>::d__0), rho, &mu);
        for(unsigned short nu = 1; nu < mu; ++nu)
            std::copy_n(rho + nu * mu, nu + 1, rho + (nu * (nu + 1)) / 2);
    }
    else
        std::fill_n(rho, (mu * (mu + 1)) / 2, K());
    MPI_Allreduce(MPI_IN_PLACE, rho, (mu * (mu + 1)) / 2, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
    for(unsigned short nu = mu; nu > 0; --nu)
        std::copy_backward(rho + (nu * (nu - 1)) / 2, rho + (nu * (nu + 1)) / 2, rho + nu * mu - (mu - nu));
    for(unsigned short i = 0; i < mu; ++i)
        for(unsigned short j = 0; j < i; ++j)
            rho[i + j * mu] = Wrapper<K>::conj(rho[j + i * mu]);
    std::copy_n(rho, mu * mu, rho + mu * mu);

    const char id = opt.val<char>("qr", 0);
    int info = QR<excluded>(id, n, mu, 1, p, gamma, mu, comm, static_cast<K*>(nullptr), true, d, trash);
    if(info) {
        delete [] trash;
        A.clearBuffer(allocate);
        return CG<excluded>(A, b, x, mu, comm);
    }
    underlying_type<K>* const norm = new underlying_type<K>[mu];
    for(unsigned short nu = 0; nu < mu; ++nu)
        norm[nu] = std::sqrt(std::real(gamma[(mu + 1) * nu]));

    unsigned short i = 1;
    while(i <= it) {
        if(!excluded) {
            A.GMV(p, z, mu);
            Blas<K>::trsm("L", "U", &(Wrapper<K>::transc), "N", &mu, &mu, &(Wrapper<K>::d__1), gamma, &mu, rho + mu * mu, &mu);
        }
        Wrapper<K>::diag(n, d, z, trash, mu);
        if(!excluded && n) {
            Blas<K>::gemmt("U", &(Wrapper<K>::transc), "N", &mu, &n, &(Wrapper<K>::d__1), p, &n, trash, &n, &(Wrapper<K>::d__0), rhs, &mu);
            for(unsigned short nu = 1; nu < mu; ++nu)
                std::copy_n(rhs + nu * mu, nu + 1, rhs + (nu * (nu + 1)) / 2);
        }
        else
            std::fill_n(rhs, (mu * (mu + 1)) / 2, K());
        MPI_Allreduce(MPI_IN_PLACE, rhs, (mu * (mu + 1)) / 2, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
        Lapack<K>::ppsv("U", &mu, &mu, rhs, rho + mu * mu, &mu, &info);
        if(info) {
            delete [] norm;
            delete [] trash;
            A.clearBuffer(allocate);
            return CG<excluded>(A, b, x, mu, comm);
        }
        if(!excluded && n) {
            Blas<K>::gemm("N", "N", &n, &mu, &mu, &(Wrapper<K>::d__1), p, &n, rho + mu * mu, &mu, &(Wrapper<K>::d__1), x, &n);
            Blas<K>::gemm("N", "N", &n, &mu, &mu, &(Wrapper<K>::d__2), z, &n, rho + mu * mu, &mu, &(Wrapper<K>::d__1), r, &n);
        }
        A.template apply<excluded>(r, z, mu, trash);
        Wrapper<K>::diag(n, d, z, trash, mu);
        if(!excluded && n) {
            Blas<K>::gemmt("U", &(Wrapper<K>::transc), "N", &mu, &n, &(Wrapper<K>::d__1), r, &n, trash, &n, &(Wrapper<K>::d__0), rhs, &mu);
            for(unsigned short nu = 1; nu < mu; ++nu)
                std::copy_n(rhs + nu * mu, nu + 1, rhs + (nu * (nu + 1)) / 2);
            for(unsigned short nu = 0; nu < mu; ++nu)
                rho[(2 * mu - 1) * mu + nu] = std::real(Blas<K>::dot(&n, z + n * nu, &i__1, trash + n * nu, &i__1));
        }
        else
            std::fill_n(rho + (2 * mu - 1) * mu, mu + (mu * (mu + 1)) / 2, K());
        MPI_Allreduce(MPI_IN_PLACE, rhs - mu, mu + (mu * (mu + 1)) / 2, Wrapper<K>::mpi_underlying_type(), MPI_SUM, comm);
        info = 0;
        for(unsigned short nu = 0; nu < mu; ++nu) {
            underlying_type<K> beta = std::sqrt(std::real(rho[(2 * mu - 1) * mu + nu]));
            if(((tol > 0.0 && beta / norm[nu] <= tol) || (tol < 0.0 && beta <= -tol)))
                ++info;
        }
        if(verbosity > 2) {
            K* max = std::max_element(rho + (2 * mu - 1) * mu, rhs, [](const K& lhs, const K& rhs) { return std::sqrt(std::real(lhs)) < std::sqrt(std::real(rhs)); });
            if(tol > 0.0)
                std::cout << "BCG: " << std::setw(3) << i << " " << std::sqrt(std::real(*max)) << " " <<  norm[std::distance(rho + (2 * mu - 1) * mu, max)] << " " <<  std::sqrt(std::real(*max)) / norm[std::distance(rho + (2 * mu - 1) * mu, max)] << " < " << tol;
            else
                std::cout << "BCG: " << std::setw(3) << i << " " << std::sqrt(std::real(*max)) << " < " << -tol;
            std::cout << " (rhs #" << std::distance(rho + (2 * mu - 1) * mu, max) + 1 << ")" << std::endl;;
        }
        if(info == mu)
            break;
        else
            ++i;
        for(unsigned short nu = mu; nu > 0; --nu)
            std::copy_backward(rhs + (nu * (nu - 1)) / 2, rhs + (nu * (nu + 1)) / 2, rhs + nu * mu - (mu - nu));
        for(unsigned short i = 0; i < mu; ++i)
            for(unsigned short j = 0; j < i; ++j)
                rhs[i + j * mu] = Wrapper<K>::conj(rhs[j + i * mu]);
        std::copy_n(rhs, mu * mu, rho + mu * mu);
        Lapack<K>::posv("U", &mu, &mu, rho, &mu, rhs, &mu, &info);
        if(info) {
            delete [] norm;
            delete [] trash;
            A.clearBuffer(allocate);
            return CG<excluded>(A, b, x, mu, comm);
        }
        if(!excluded && n) {
            Blas<K>::trmm("L", "U", "N", "N", &mu, &mu, &(Wrapper<K>::d__1), gamma, &mu, rhs, &mu);
            std::copy(p, r, trash);
            Blas<K>::gemm("N", "N", &n, &mu, &mu, &(Wrapper<K>::d__1), trash, &n, rhs, &mu, &(Wrapper<K>::d__1), p, &n);
        }
        info = QR<excluded>(id, n, mu, 1, p, gamma, mu, comm, static_cast<K*>(nullptr), true, d, trash);
        if(info) {
            delete [] norm;
            delete [] trash;
            A.clearBuffer(allocate);
            return CG<excluded>(A, b, x, mu, comm);
        }
        std::copy_n(rho + mu * mu, mu * mu, rho);
    }
    if(verbosity) {
        if(i != it + 1)
            std::cout << "BCG converges after " << i << " iteration" << (i > 1 ? "s" : "") << std::endl;
        else
            std::cout << "BCG does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
    }
    delete [] norm;
    delete [] trash;
    A.clearBuffer(allocate);
    std::cout.unsetf(std::ios_base::scientific);
    return std::min(i, it);
}
template<bool excluded, class Operator, class K>
inline int IterativeMethod::PCG(const Operator& A, const K* const f, K* const x, const MPI_Comm& comm) {
    typedef typename std::conditional<std::is_pointer<typename std::remove_reference<decltype(*A.getScaling())>::type>::value, K**, K*>::type ptr_type;
    const Option& opt = *Option::get();
    const int n = std::is_same<ptr_type, K*>::value ? A.getDof() : A.getMult();
    const unsigned short it = opt.val<unsigned short>("max_it", 100);
    underlying_type<K> tol = opt.val("tol", 1.0e-6);
    const char verbosity = opt.val<char>("verbosity", 0);
    std::cout << std::scientific;
    epsilon(tol, verbosity);
    const int offset = std::is_same<ptr_type, K*>::value ? A.getEliminated() : 0;
    ptr_type storage[std::is_same<ptr_type, K*>::value ? 1 : 2];
    // storage[0] = r
    // storage[1] = lambda
    A.allocateArray(storage);
    bool allocate = A.setBuffer();
    auto m = A.getScaling();
    if(std::is_same<ptr_type, K*>::value)
        A.template start<excluded>(f, x + offset, nullptr, storage[0]);
    else
        A.template start<excluded>(f, x, storage[1], storage[0]);

    std::vector<ptr_type> z;
    z.reserve(it);
    ptr_type zCurr;
    A.allocateSingle(zCurr);
    z.emplace_back(zCurr);
    if(!excluded)
        A.precond(storage[0], zCurr);                                                              //     z_0 = M r_0

    underlying_type<K> resInit;
    A.template computeDot<excluded>(&resInit, zCurr, zCurr, comm);
    resInit = std::sqrt(resInit);

    std::vector<ptr_type> p;
    p.reserve(it);
    ptr_type pCurr;
    A.allocateSingle(pCurr);
    p.emplace_back(pCurr);

    K* alpha = new K[excluded ? std::max(static_cast<unsigned short>(2), it) : 2 * it];
    underlying_type<K> resRel = std::numeric_limits<underlying_type<K>>::max();
    unsigned short i = 1;
    while(i <= it) {
        if(!excluded) {
            A.template project<excluded, 'N'>(zCurr, pCurr);                                       //     p_i = P z_i
            for(unsigned short k = 0; k < i - 1; ++k)
                alpha[it + k] = dot(&n, z[k], &i__1, pCurr, &i__1);
            MPI_Allreduce(MPI_IN_PLACE, alpha + it, i - 1, Wrapper<K>::mpi_type(), MPI_SUM, comm); // alpha_k = < z_k, p_i >
            for(unsigned short k = 0; k < i - 1; ++k) {
                alpha[it + k] /= -alpha[k];
                axpy(&n, alpha + it + k, p[k], &i__1, pCurr, &i__1);                               //     p_i = p_i - sum < z_k, p_i > / < z_k, p_k > p_k
            }
            A.apply(pCurr, zCurr);                                                                 //     z_i = F p_i

            A.allocateSingle(zCurr);
            if(std::is_same<ptr_type, K*>::value) {
                diag(n, m, pCurr, zCurr);
                alpha[i - 1] = dot(&n, z.back(), &i__1, zCurr, &i__1);
                alpha[i]     = dot(&n, storage[0], &i__1, zCurr, &i__1);
            }
            else {
                alpha[i - 1] = dot(&n, z.back(), &i__1, pCurr, &i__1);
                alpha[i]     = dot(&n, storage[0], &i__1, pCurr, &i__1);
            }
            MPI_Allreduce(MPI_IN_PLACE, alpha + i - 1, 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            alpha[it] = alpha[i] / alpha[i - 1];
            if(std::is_same<ptr_type, K*>::value)
                axpy(&n, alpha + it, pCurr, &i__1, x + offset, &i__1);
            else
                axpy(&n, alpha + it, pCurr, &i__1, storage[1], &i__1);                             // l_i + 1 = l_i + < r_i, p_i > / < z_i, p_i > p_i
            alpha[it] = -alpha[it];
            axpy(&n, alpha + it, z.back(), &i__1, storage[0], &i__1);                              // r_i + 1 = r_i - < r_i, p_i > / < z_i, p_i > z_i
            A.template project<excluded, 'T'>(storage[0]);                                         // r_i + 1 = P^T r_i + 1

            z.emplace_back(zCurr);
            A.precond(storage[0], zCurr);                                                          // z_i + 1 = M r_i
        }
        else {
            A.template project<excluded, 'N'>(zCurr, pCurr);
            std::fill_n(alpha, i - 1, K());
            MPI_Allreduce(MPI_IN_PLACE, alpha, i - 1, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            std::fill_n(alpha, 2, K());
            MPI_Allreduce(MPI_IN_PLACE, alpha, 2, Wrapper<K>::mpi_type(), MPI_SUM, comm);
            A.template project<excluded, 'T'>(storage[0]);
        }
        A.template computeDot<excluded>(&resRel, zCurr, zCurr, comm);
        resRel = std::sqrt(resRel);
        if(verbosity > 2)
            std::cout << "PCG: " << std::setw(3) << i << " " << resRel << " " << resInit << " " << resRel / resInit << " < " << tol << std::endl;
        if(resRel / resInit <= tol)
            break;
        else
            ++i;
        if(!excluded) {
            A.allocateSingle(pCurr);
            p.emplace_back(pCurr);
            diag(n, m, z[i - 2]);
        }
    }
    if(verbosity) {
        if(i != it + 1)
            std::cout << "PCG converges after " << i << " iteration" << (i > 1 ? "s" : "") << std::endl;
        else
            std::cout << "PCG does not converges after " << it << " iteration" << (it > 1 ? "s" : "") << std::endl;
    }
    if(std::is_same<ptr_type, K*>::value)
        A.template computeSolution<excluded>(f, x);
    else
        A.template computeSolution<excluded>(storage[1], x);
    delete [] alpha;
    for(auto zCurr : z)
        clean(zCurr);
    for(auto pCurr : p)
        clean(pCurr);
    clean(storage[0]);
    A.clearBuffer(allocate);
    std::cout.unsetf(std::ios_base::scientific);
    return std::min(i, it);
}
} // HPDDM
#endif // _HPDDM_CG_
