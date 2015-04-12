/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <jolivet@ann.jussieu.fr>
              Ralf Deiterding <ralf.deiterding@dlr.de>
        Date: 2013-05-22

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

#include <HPDDM.hpp>
#include <list>

#define xx(i) (xdim[0] + dx * (i + 0.5))
#define yy(j) (ydim[0] + dy * (j + 0.5))

#ifdef FORCE_SINGLE
#ifdef FORCE_COMPLEX
typedef std::complex<float> K;
#ifndef GENERAL_CO
#define GENERAL_CO
#endif
#else
typedef float K;
#endif
#else
#ifdef FORCE_COMPLEX
typedef std::complex<double> K;
#ifndef GENERAL_CO
#define GENERAL_CO
#endif
#else
typedef double K;
#endif
#endif

#ifdef GENERAL_CO
const char symmetryCoarseOperator = 'G';
#else
const char symmetryCoarseOperator = 'S';
#endif

const HPDDM::Wrapper<K>::ul_type pi = 3.141592653589793238463;

int main(int argc, char **argv) {
#if !((OMPI_MAJOR_VERSION > 1 || (OMPI_MAJOR_VERSION == 1 && OMPI_MINOR_VERSION >= 7)) || MPICH_NUMVERSION >= 30000000)
    MPI_Init(&argc, &argv);
#else
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);
#endif
    if(argc < 7 || argc > 10) {
        int rankWorld;
        MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
        if(rankWorld == 0) {
            std::cout << "Parameters expected: prec Nx Ny overlap eps symmetry" << std::endl;
            std::cout << "           prec = 0  => one-level RAS" << std::endl;
            std::cout << "           prec = -1 => two-level Nicolaides" << std::endl;
            std::cout << "           prec = M  => two-level GenEO" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
    std::vector<std::string> arguments(argv + 1, argv + argc);
    int prec = HPDDM::sto<int>(arguments[0]);
    int Nx = HPDDM::sto<int>(arguments[1]);
    int Ny = HPDDM::sto<int>(arguments[2]);
    int overlap = HPDDM::sto<int>(arguments[3]);
    HPDDM::Wrapper<K>::ul_type eps = HPDDM::sto<HPDDM::Wrapper<K>::ul_type>(arguments[4]);
    int sym = HPDDM::sto<int>(arguments[5]);
    /*# Init #*/
    int rankWorld;
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int xGrid = int(sqrt(sizeWorld));
    while(sizeWorld % xGrid != 0)
        --xGrid;
    int yGrid = sizeWorld / xGrid;

    int y = rankWorld / xGrid;
    int x = rankWorld - xGrid * y;

    int iStart = std::max(x * Nx / xGrid - overlap, 0);
    int iEnd   = std::min((x + 1) * Nx / xGrid + overlap, Nx);
    int jStart = std::max(y * Ny / yGrid - overlap, 0);
    int jEnd   = std::min((y + 1) * Ny / yGrid + overlap, Ny);
    int ndof   = (iEnd - iStart) * (jEnd - jStart);
    int nnz = ndof * 3 - (iEnd - iStart) - (jEnd - jStart);
    /*# InitEnd #*/
    if(!sym)
        nnz = 2 * nnz - ndof;
    HPDDM::Wrapper<K>::ul_type xdim[2] = { 0.0, 10.0 };
    HPDDM::Wrapper<K>::ul_type ydim[2] = { 0.0, 10.0 };
    int Nf = 3;
    HPDDM::Wrapper<K>::ul_type xsc[3] = { 6.5, 2.0, 7.0 };
    HPDDM::Wrapper<K>::ul_type ysc[3] = { 8.0, 7.0, 3.0 };
    HPDDM::Wrapper<K>::ul_type rsc[3] = { 0.3, 0.3, 0.4 };
    HPDDM::Wrapper<K>::ul_type asc[3] = { 0.3, 0.2, -0.1 };
    HPDDM::Wrapper<K>::ul_type dx = (xdim[1] - xdim[0]) / static_cast<HPDDM::Wrapper<K>::ul_type>(Nx);
    HPDDM::Wrapper<K>::ul_type dy = (ydim[1] - ydim[0]) / static_cast<HPDDM::Wrapper<K>::ul_type>(Ny);
    K* f = new K[ndof];
    K* sol = new K[ndof]();
    for(int j = jStart, k = 0; j < jEnd; ++j) {
        for(int i = iStart; i < iEnd; ++i, ++k) {
            HPDDM::Wrapper<K>::ul_type frs = 1.0;
            for(int n = 0; n < Nf; ++n) {
                HPDDM::Wrapper<K>::ul_type xdist = (xx(i) - xsc[n]), ydist = (yy(j) - ysc[n]);
                if(sqrt(xdist * xdist + ydist * ydist) <= rsc[n])
                    frs -= asc[n] * cos(0.5 * pi * xdist / rsc[n]) * cos(0.5 * pi * ydist / rsc[n]);
                f[k] = frs;
            }
        }
    }
    /*# Structures #*/
    HPDDM::Wrapper<K>::ul_type* d = new HPDDM::Wrapper<K>::ul_type[ndof];
    std::fill(d, d + ndof, 1.0);
    std::vector<std::vector<int>> mapping;
    mapping.reserve(8);
    std::list<int> o; // at most eight neighbors in 2D
    if(jStart != 0) { // this subdomain doesn't touch the bottom side of %*\color{DarkGreen}{$\Omega$}*)
        if(iStart != 0) { // this subd. doesn't touch the left side of %*\color{DarkGreen}{$\Omega$}*)
            o.push_back(rankWorld - xGrid - 1); // subd. on the lower left corner is a neighbor
            mapping.push_back(std::vector<int>());
            mapping.back().reserve(4 * overlap * overlap);
            for(int j = 0; j < 2 * overlap; ++j)
                for(int i = iStart; i < iStart + 2 * overlap; ++i)
                    mapping.back().push_back(i - iStart + (iEnd - iStart) * j);
            for(int j = 0; j < overlap; ++j) {
                for(int i = 0; i < overlap - j; ++i)
                    d[i + j + j * (iEnd - iStart)] = j / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
                for(int i = 0; i < j; ++i)
                    d[i + j * (iEnd - iStart)] = i / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
            }
        }
        else // this subd. touches the left side of %*\color{DarkGreen}{$\Omega$}*)
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    d[i + j * (iEnd - iStart)] = j / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
        o.push_back(rankWorld - xGrid); // subd. below is a neighbor
        mapping.push_back(std::vector<int>());
        mapping.back().reserve(2 * overlap * (iEnd - iStart));
        for(int j = 0; j < 2 * overlap; ++j)
            for(int i = iStart; i < iEnd; ++i)
                mapping.back().push_back(i - iStart + (iEnd - iStart) * j);
        for(int j = 0; j < overlap; ++j)
            for(int i = iStart + overlap; i < iEnd - overlap; ++i)
                d[i - iStart + (iEnd - iStart) * j] = j / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
        if(iEnd != Nx) { // this subd. doesn't touch the right side of %*\color{DarkGreen}{$\Omega$}*)
            o.push_back(rankWorld - xGrid + 1); // subd. on the lower right corner is a neighbor
            mapping.push_back(std::vector<int>());
            mapping.back().reserve(4 * overlap * overlap);
            for(int i = 0; i < 2 * overlap; ++i)
                for(int j = 0; j < 2 * overlap; ++j)
                    mapping.back().push_back((iEnd - iStart) * (i + 1) - 2 * overlap + j);
            for(int j = 0; j < overlap; ++j) {
                for(int i = 0; i < overlap - j; ++i)
                    d[(iEnd - iStart) * (j + 1) - overlap + i] = j / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
                for(int i = 0; i < j; ++i)
                    d[(iEnd - iStart) * (j + 1) - i - 1] = i / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
            }
        }
        else
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    d[(iEnd - iStart) * (j + 1) - overlap + i] = j / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
    }
    /*# StructuresEnd #*/
    if(iStart != 0) {
        o.push_back(rankWorld - 1);
        mapping.push_back(std::vector<int>());
        mapping.back().reserve(2 * overlap * (jEnd - jStart));
        for(int i = jStart; i < jEnd; ++i)
            for(int j = 0; j < 2 * overlap; ++j)
                mapping.back().push_back(j + (i - jStart) * (iEnd - iStart));
        for(int i = jStart + (jStart != 0) * overlap; i < jEnd - (jEnd != Ny) * overlap; ++i)
            for(int j = 0; j < overlap; ++j)
                d[j + (i - jStart) * (iEnd - iStart)] = j / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
    }
    if(iEnd != Nx) {
        o.push_back(rankWorld + 1);
        mapping.push_back(std::vector<int>());
        mapping.back().reserve(2 * overlap * (jEnd - jStart));
        for(int i = jStart; i < jEnd; ++i)
            for(int j = 0; j < 2 * overlap; ++j)
                mapping.back().push_back((iEnd - iStart) * (i + 1 - jStart) - 2 * overlap + j);
        for(int i = jStart + (jStart != 0) * overlap; i < jEnd - (jEnd != Ny) * overlap; ++i)
            for(int j = 0; j < overlap; ++j)
                d[(iEnd - iStart) * (i + 1 - jStart) - j - 1] = j / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
    }
    if(jEnd != Ny) {
        if(iStart != 0) {
            o.push_back(rankWorld + xGrid - 1);
            mapping.push_back(std::vector<int>());
            mapping.back().reserve(4 * overlap * overlap);
            for(int j = 0; j < 2 * overlap; ++j)
                for(int i = iStart; i < iStart + 2 * overlap; ++i)
                    mapping.back().push_back(ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j);
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap - j; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * j] = i / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
            for(int j = 0; j < overlap; ++j)
                for(int i = overlap - j; i < overlap; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * j] = (overlap - 1 - j) / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
        }
        else {
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    d[ndof - overlap * (iEnd - iStart) + (iEnd - iStart) * j + i] = (overlap - j - 1) / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
        }
        o.push_back(rankWorld + xGrid);
        mapping.push_back(std::vector<int>());
        mapping.back().reserve(2 * overlap * (iEnd - iStart));
        for(int j = 0; j < 2 * overlap; ++j)
            for(int i = iStart; i < iEnd; ++i)
                mapping.back().push_back(ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j);
        for(int j = 0; j < overlap; ++j)
            for(int i = iStart + overlap; i < iEnd - overlap; ++i)
                d[ndof - overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j] = (overlap - 1 - j) / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
        if(iEnd != Nx) {
            o.push_back(rankWorld + xGrid + 1);
            mapping.push_back(std::vector<int>());
            mapping.back().reserve(4 * overlap * overlap);
            for(int j = 0; j < 2 * overlap; ++j)
                for(int i = iStart; i < iStart + 2 * overlap; ++i)
                    mapping.back().push_back(ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j + (iEnd - iStart - 2 * overlap));
            for(int j = 0; j < overlap; ++j)
                for(int i = j; i < overlap; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - 1 - i) / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < j; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - 1 - j) / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
        }
        else {
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - j - 1) / static_cast<HPDDM::Wrapper<K>::ul_type>(overlap);
        }
    }

    int* ia;
    int* ja;
    K* a;
    if(sym) {
        /*# Matrix #*/
        ia = new int[ndof + 1];
        ja = new int[nnz];
        a = new K[nnz];
        ia[0] = 0;
        ia[ndof] = nnz;
        for(int j = jStart, k = 0, nnz = 0; j < jEnd; ++j) {
            for(int i = iStart; i < iEnd; ++i) {
                if(j > jStart) { // this d.o.f. is not on the bottom side of the subd.
                    a[nnz] = -1 / (dy * dy);
                    ja[nnz++] = k - (Ny / yGrid);
                }
                if(i > iStart) { // this d.o.f. is not on the left side of the subd.
                    a[nnz] = -1 / (dx * dx);
                    ja[nnz++] = k - 1;
                }
                a[nnz]  = 2 / (dx * dx) + 2 / (dy * dy);
                ja[nnz++] = k;
                ia[++k] = nnz;
            }
        }
        /*# MatrixEnd #*/
    }
    else {
        ia = new int[ndof + 1];
        ja = new int[nnz];
        a = new K[nnz];
        ia[0] = 0;
        ia[ndof] = nnz;
        for(int j = jStart, k = 0, nnz = 0; j < jEnd; ++j) {
            for(int i = iStart; i < iEnd; ++i) {
                if(j > jStart) {
                    a[nnz] = -1 / (dy * dy);
                    ja[nnz++] = k - (Ny / yGrid);
                }
                if(i > iStart) {
                    a[nnz] = -1 / (dx * dx);
                    ja[nnz++] = k - 1;
                }
                a[nnz]  = 2 / (dx * dx) + 2 / (dy * dy);
                ja[nnz++] = k;
                if(i < iEnd - 1) {
                    a[nnz] = -1 / (dx * dx);
                    ja[nnz++] = k + 1;
                }
                if(j < jEnd - 1) {
                    a[nnz] = -1 / (dy * dy);
                    ja[nnz++] = k + (Ny / yGrid);
                }
                ia[++k] = nnz;
            }
        }
    }
    HPDDM::MatrixCSR<K>* N = nullptr;
    int* in = nullptr;
    int* jn = nullptr;
    K* neumann = nullptr;
    if(prec > 0) {
        if(sym) {
            int nnzNeumann = 2 * nnz - ndof;
            in = new int[ndof + 1];
            jn = new int[nnzNeumann];
            in[0] = 0;
            in[ndof] = nnzNeumann;
            neumann = new K[nnzNeumann];
            N = new HPDDM::MatrixCSR<K>(ndof, ndof, nnzNeumann, neumann, in, jn, 0);
            for(int j = jStart, k = 0, nnzNeumann = 0; j < jEnd; ++j) {
                for(int i = iStart; i < iEnd; ++i) {
                    if(j > jStart) {
                        neumann[nnzNeumann] = -1 / (dy * dy) + (i == iStart ? -1 / (dx * dx) : 0);
                        jn[nnzNeumann++] = k - (Ny / yGrid);
                    }
                    if(i > iStart) {
                        neumann[nnzNeumann] = -1 / (dx * dx) + (j == jStart ? -1 / (dy * dy) : 0);
                        jn[nnzNeumann++] = k - 1;
                    }
                    neumann[nnzNeumann]  = 2 / (dx * dx) + 2 / (dy * dy);
                    jn[nnzNeumann++] = k;
                    if(i < iEnd - 1) {
                        neumann[nnzNeumann] = -1 / (dx * dx) + (j == jEnd - 1 ? -1 / (dy * dy) : 0);
                        jn[nnzNeumann++] = k + 1;
                    }
                    if(j < jEnd - 1) {
                        neumann[nnzNeumann] = -1 / (dy * dy) + (i == iEnd - 1 ? -1 / (dx * dx) : 0);
                        jn[nnzNeumann++] = k + (Ny / yGrid);
                    }
                    in[++k] = nnzNeumann;
                }
            }
        }
        else {
            in = ia;
            jn = ja;
            neumann = new K[nnz];
            N = new HPDDM::MatrixCSR<K>(ndof, ndof, nnz, neumann, in, jn, 0);
            std::copy(a, a + nnz, neumann);
            for(int j = jStart, k = 0, nnz = 0; j < jEnd; ++j)
                for(int i = iStart; i < iEnd; ++i) {
                    if(j > jStart) {
                        if(i == iStart)
                            neumann[nnz] -= 1 / (dx * dx);
                        ++nnz;
                    }
                    if(i > iStart) {
                        if(i == iStart)
                            neumann[nnz] -= 1 / (dy * dy);
                        ++nnz;
                    }
                    ++nnz;
                    if(i < iEnd - 1) {
                        if(j == jEnd - 1)
                            neumann[nnz] -= 1 / (dy * dy);
                        ++nnz;
                    }
                    if(j < jEnd - 1) {
                        if(i == iEnd - 1)
                            neumann[nnz] -= 1 / (dx * dx);
                        ++nnz;
                    }
                }
        }
    }
    HPDDM::MatrixCSR<K>* Mat = new HPDDM::MatrixCSR<K>(ndof, ndof, nnz, a, ia, ja, sym, true);
    double timing;
    /*# Deflation #*/
    K** deflation = new K*[1];
    *deflation = new K[ndof];
    std::fill(*deflation, *deflation + ndof, 1.0);
    /*# DeflationEnd #*/
    if(sizeWorld > 1) {
        /*# Creation #*/
        HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, symmetryCoarseOperator, K> A;
        /*# CreationEnd #*/
        /*# Initialization #*/
        A.Subdomain::initialize(Mat, o, mapping);
        decltype(mapping)().swap(mapping);
        A.multiplicityScaling(d);
        A.initialize(d);
        /*# InitializationEnd #*/
        for(int k = 0; k < sizeWorld; ++k) {
            if(k == rankWorld) {
                std::cout << rankWorld << ":" << std::endl;
                std::cout << x << "x" << y << " -- [" << iStart << " ; " << iEnd << "] x [" << jStart << " ; " << jEnd << "] -- " << ndof << ", " << nnz << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        timing = MPI_Wtime();
        if(prec != 0) {
            A.setType(0);
            /*# Factorization #*/
            std::vector<unsigned short> parm(5);
            parm[HPDDM::P] = 1;
            parm[HPDDM::TOPOLOGY] = 0;
            if(std::find(arguments.begin() + 6, arguments.end(), "-distributed_sol") != arguments.end()) {
                if(std::find(arguments.begin() + 6, arguments.end(), "-distributed_rhs") != arguments.end())
                    parm[HPDDM::DISTRIBUTION] = HPDDM::DMatrix::DISTRIBUTED_SOL_AND_RHS;
                else
                    parm[HPDDM::DISTRIBUTION] = HPDDM::DMatrix::DISTRIBUTED_SOL;
            }
            else
                parm[HPDDM::DISTRIBUTION] = HPDDM::DMatrix::NON_DISTRIBUTED;
            parm[HPDDM::STRATEGY] = 3;
            if(prec > 0) {
                parm[HPDDM::NU] = prec;
                if(std::find(arguments.begin() + 6, arguments.end(), "-nonuniform") != arguments.end())
                    parm[HPDDM::NU] += std::max(-prec + 1, (-1)^rankWorld * rankWorld);
                HPDDM::Wrapper<K>::ul_type threshold = 0.0;
                A.solveGEVP<HPDDM::Arpack>(N, parm[HPDDM::NU], threshold);
            }
            else {
                parm[HPDDM::NU] = 1;
                A.setVectors(deflation);
            }
            A.super::initialize(parm[HPDDM::NU]);
            A.buildTwo(MPI_COMM_WORLD, parm);
            A.callNumfact();
            /*# FactorizationEnd #*/
            /*# Solution #*/
            unsigned short it = 100;
            unsigned short restart = 30;
            HPDDM::IterativeMethod::GMRES(A, sol, f, restart, it, eps, A.getCommunicator(), rankWorld == 0 ? 1 : 0);
            /*# SolutionEnd #*/
        }
        else {
            A.setType(1);
            A.callNumfact();
            unsigned short it = 100;
            HPDDM::IterativeMethod::CG(A, sol, f, it, eps, A.getCommunicator(), rankWorld == 0 ? 1 : 0);
        }
        timing = MPI_Wtime() - timing;
        HPDDM::Wrapper<K>::ul_type storage[2];
        A.computeError(sol, f, storage);
        if(rankWorld == 0)
            std::cout << std::scientific << " --- error = " << storage[1] << " / " << storage[0] << std::endl;
        delete [] d;
    }
    else {
        SUBDOMAIN<K> S;
        timing = MPI_Wtime();
        S.numfact(Mat);
        S.solve(f, sol);
        timing = MPI_Wtime() - timing;
        HPDDM::Wrapper<K>::ul_type nrmb = HPDDM::Wrapper<K>::nrm2(&ndof, f, &(HPDDM::i__1));
        K* tmp = new K[ndof];
        HPDDM::Wrapper<K>::csrmv<'C'>(sym, &ndof, a, ia, ja, sol, tmp);
        HPDDM::Wrapper<K>::axpy(&ndof, &(HPDDM::Wrapper<K>::d__2), f, &(HPDDM::i__1), tmp, &(HPDDM::i__1));
        HPDDM::Wrapper<K>::ul_type nrmAx = HPDDM::Wrapper<K>::nrm2(&ndof, tmp, &(HPDDM::i__1));
        std::cout << std::scientific << " --- error = " << nrmAx << " / " << nrmb << std::endl;
        delete [] tmp;
        delete Mat;
    }

    if(prec >= 0) {
        if(deflation)
            delete [] *deflation;
        delete [] deflation;
    }
    if(prec > 0) {
        delete N;
        delete [] neumann;
        if(sym) {
            delete [] in;
            delete [] jn;
        }
    }
    delete [] sol;
    delete [] f;
    MPI_Finalize();
    return 0;
}
