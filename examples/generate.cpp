/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
              Ralf Deiterding <r.deiterding@soton.ac.uk>
        Date: 2013-05-22

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

#define HPDDM_MINIMAL
#include "schwarz.hpp"

#define xx(i) (xdim[0] + dx * (i + 0.5))
#define yy(j) (ydim[0] + dy * (j + 0.5))

template<class K, typename std::enable_if<!HPDDM::Wrapper<K>::is_complex>::type* = nullptr>
void assign(std::mt19937& gen, std::uniform_real_distribution<K>& dis, K& x) {
    x = dis(gen);
}
template<class K, typename std::enable_if<HPDDM::Wrapper<K>::is_complex>::type* = nullptr>
void assign(std::mt19937& gen, std::uniform_real_distribution<HPDDM::underlying_type<K>>& dis, K& x) {
    x = K(dis(gen), dis(gen));
}

void generate(int rankWorld, int sizeWorld, std::list<int>& o, std::vector<std::vector<int>>& mapping, int& ndof, HPDDM::MatrixCSR<K>*& Mat, HPDDM::MatrixCSR<K>*& MatNeumann, HPDDM::underlying_type<K>*& d, K*& f, K*& sol) {
    HPDDM::Option& opt = *HPDDM::Option::get();
    const int Nx = opt.app()["Nx"];
    const int Ny = opt.app()["Ny"];
    const int overlap = opt.app()["overlap"];
    const int mu = opt.app()["generate_random_rhs"];
    const bool sym = opt.app().find("symmetric_csr") != opt.app().cend() && (opt.app()["symmetric_csr"] == 1);
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
    ndof       = (iEnd - iStart) * (jEnd - jStart);
    int nnz = ndof * 3 - (iEnd - iStart) - (jEnd - jStart);
    /*# InitEnd #*/
    if(!sym)
        nnz = 2 * nnz - ndof;
    f = new K[std::max(1, mu) * ndof];
    sol = new K[std::max(1, mu) * ndof]();
    HPDDM::underlying_type<K> xdim[2] = { 0.0, 10.0 };
    HPDDM::underlying_type<K> ydim[2] = { 0.0, 10.0 };
    HPDDM::underlying_type<K> dx = (xdim[1] - xdim[0]) / static_cast<HPDDM::underlying_type<K>>(Nx);
    HPDDM::underlying_type<K> dy = (ydim[1] - ydim[0]) / static_cast<HPDDM::underlying_type<K>>(Ny);
    if(mu == 0) {
        int Nf = 3;
        HPDDM::underlying_type<K> xsc[3] = { 6.5, 2.0, 7.0 };
        HPDDM::underlying_type<K> ysc[3] = { 8.0, 7.0, 3.0 };
        HPDDM::underlying_type<K> rsc[3] = { 0.3, 0.3, 0.4 };
        HPDDM::underlying_type<K> asc[3] = { 0.3, 0.2, -0.1 };
        for(int j = jStart, k = 0; j < jEnd; ++j) {
            for(int i = iStart; i < iEnd; ++i, ++k) {
                HPDDM::underlying_type<K> frs = 1.0;
                for(int n = 0; n < Nf; ++n) {
                    HPDDM::underlying_type<K> xdist = (xx(i) - xsc[n]), ydist = (yy(j) - ysc[n]);
                    if(sqrt(xdist * xdist + ydist * ydist) <= rsc[n])
                        frs -= asc[n] * cos(0.5 * pi * xdist / rsc[n]) * cos(0.5 * pi * ydist / rsc[n]);
                    f[k] = frs;
                }
            }
        }
    }
    else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<HPDDM::underlying_type<K>> dis(0.0, 1.0);
        std::for_each(f, f + mu * ndof, [&](K& x) { assign(gen, dis, x); });
    }
    /*# Structures #*/
    d = new HPDDM::underlying_type<K>[ndof];
    std::fill(d, d + ndof, 1.0);
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
                    d[i + j + j * (iEnd - iStart)] = j / static_cast<HPDDM::underlying_type<K>>(overlap);
                for(int i = 0; i < j; ++i)
                    d[i + j * (iEnd - iStart)] = i / static_cast<HPDDM::underlying_type<K>>(overlap);
            }
        }
        else // this subd. touches the left side of %*\color{DarkGreen}{$\Omega$}*)
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    d[i + j * (iEnd - iStart)] = j / static_cast<HPDDM::underlying_type<K>>(overlap);
        o.push_back(rankWorld - xGrid); // subd. below is a neighbor
        mapping.push_back(std::vector<int>());
        mapping.back().reserve(2 * overlap * (iEnd - iStart));
        for(int j = 0; j < 2 * overlap; ++j)
            for(int i = iStart; i < iEnd; ++i)
                mapping.back().push_back(i - iStart + (iEnd - iStart) * j);
        for(int j = 0; j < overlap; ++j)
            for(int i = iStart + overlap; i < iEnd - overlap; ++i)
                d[i - iStart + (iEnd - iStart) * j] = j / static_cast<HPDDM::underlying_type<K>>(overlap);
        if(iEnd != Nx) { // this subd. doesn't touch the right side of %*\color{DarkGreen}{$\Omega$}*)
            o.push_back(rankWorld - xGrid + 1); // subd. on the lower right corner is a neighbor
            mapping.push_back(std::vector<int>());
            mapping.back().reserve(4 * overlap * overlap);
            for(int i = 0; i < 2 * overlap; ++i)
                for(int j = 0; j < 2 * overlap; ++j)
                    mapping.back().push_back((iEnd - iStart) * (i + 1) - 2 * overlap + j);
            for(int j = 0; j < overlap; ++j) {
                for(int i = 0; i < overlap - j; ++i)
                    d[(iEnd - iStart) * (j + 1) - overlap + i] = j / static_cast<HPDDM::underlying_type<K>>(overlap);
                for(int i = 0; i < j; ++i)
                    d[(iEnd - iStart) * (j + 1) - i - 1] = i / static_cast<HPDDM::underlying_type<K>>(overlap);
            }
        }
        else
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    d[(iEnd - iStart) * (j + 1) - overlap + i] = j / static_cast<HPDDM::underlying_type<K>>(overlap);
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
                d[j + (i - jStart) * (iEnd - iStart)] = j / static_cast<HPDDM::underlying_type<K>>(overlap);
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
                d[(iEnd - iStart) * (i + 1 - jStart) - j - 1] = j / static_cast<HPDDM::underlying_type<K>>(overlap);
    }
    if(jEnd != Ny) {
        if(iStart != 0) {
            o.push_back(rankWorld + xGrid - 1);
            mapping.push_back(std::vector<int>());
            mapping.back().reserve(4 * overlap * overlap);
            for(int j = 0; j < 2 * overlap; ++j)
                for(int i = iStart; i < iStart + 2 * overlap; ++i)
                    mapping.back().push_back(ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j);
            for(int j = 0; j < overlap; ++j) {
                for(int i = 0; i < overlap - j; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * j] = i / static_cast<HPDDM::underlying_type<K>>(overlap);
                for(int i = overlap - j; i < overlap; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * j] = (overlap - 1 - j) / static_cast<HPDDM::underlying_type<K>>(overlap);
            }
        }
        else {
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    d[ndof - overlap * (iEnd - iStart) + (iEnd - iStart) * j + i] = (overlap - j - 1) / static_cast<HPDDM::underlying_type<K>>(overlap);
        }
        o.push_back(rankWorld + xGrid);
        mapping.push_back(std::vector<int>());
        mapping.back().reserve(2 * overlap * (iEnd - iStart));
        for(int j = 0; j < 2 * overlap; ++j)
            for(int i = iStart; i < iEnd; ++i)
                mapping.back().push_back(ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j);
        for(int j = 0; j < overlap; ++j)
            for(int i = iStart + overlap; i < iEnd - overlap; ++i)
                d[ndof - overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j] = (overlap - 1 - j) / static_cast<HPDDM::underlying_type<K>>(overlap);
        if(iEnd != Nx) {
            o.push_back(rankWorld + xGrid + 1);
            mapping.push_back(std::vector<int>());
            mapping.back().reserve(4 * overlap * overlap);
            for(int j = 0; j < 2 * overlap; ++j)
                for(int i = iStart; i < iStart + 2 * overlap; ++i)
                    mapping.back().push_back(ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j + (iEnd - iStart - 2 * overlap));
            for(int j = 0; j < overlap; ++j) {
                for(int i = j; i < overlap; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - 1 - i) / static_cast<HPDDM::underlying_type<K>>(overlap);
                for(int i = 0; i < j; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - 1 - j) / static_cast<HPDDM::underlying_type<K>>(overlap);
            }
        }
        else {
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    d[ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - j - 1) / static_cast<HPDDM::underlying_type<K>>(overlap);
        }
    }

    int* in = nullptr, *jn = nullptr;
    K* neumann = nullptr;
    constexpr char N = HPDDM_NUMBERING;
    int* ia = new int[ndof + 1];
    int* ja = new int[nnz];
    K* a = new K[nnz];
    ia[0] = (N == 'F');
    ia[ndof] = nnz + (N == 'F');
    if(sym) {
        /*# Matrix #*/
        for(int j = jStart, k = 0, nnz = 0; j < jEnd; ++j) {
            for(int i = iStart; i < iEnd; ++i) {
                if(j > jStart) { // this d.o.f. is not on the bottom side of the subd.
                    a[nnz] = -1 / (dy * dy);
                    ja[nnz++] = k - (Nx / xGrid) + (N == 'F');
                }
                if(i > iStart) { // this d.o.f. is not on the left side of the subd.
                    a[nnz] = -1 / (dx * dx);
                    ja[nnz++] = k - (N == 'C');
                }
                a[nnz]  = 2 / (dx * dx) + 2 / (dy * dy);
                ja[nnz++] = k + (N == 'F');
                ia[++k] = nnz + (N == 'F');
            }
        }
        /*# MatrixEnd #*/
    }
    else {
        for(int j = jStart, k = 0, nnz = 0; j < jEnd; ++j) {
            for(int i = iStart; i < iEnd; ++i) {
                if(j > jStart) {
                    a[nnz] = -1 / (dy * dy);
                    ja[nnz++] = k - (Nx / xGrid) + (N == 'F');
                }
                if(i > iStart) {
                    a[nnz] = -1 / (dx * dx);
                    ja[nnz++] = k - (N == 'C');
                }
                a[nnz]  = 2 / (dx * dx) + 2 / (dy * dy);
                ja[nnz++] = k + (N == 'F');
                if(i < iEnd - 1) {
                    a[nnz] = -1 / (dx * dx);
                    ja[nnz++] = k + 1 + (N == 'F');
                }
                if(j < jEnd - 1) {
                    a[nnz] = -1 / (dy * dy);
                    ja[nnz++] = k + (Nx / xGrid) + (N == 'F');
                }
                ia[++k] = nnz + (N == 'F');
            }
        }
    }
    if(sizeWorld > 1 && opt.set(opt.prefix("prefix") + "schwarz_coarse_correction") && opt[opt.prefix("prefix") + "geneo_nu"] > 0) {
        if(sym) {
            int nnzNeumann = 2 * nnz - ndof;
            in = new int[ndof + 1];
            jn = new int[nnzNeumann];
            in[0] = (N == 'F');
            in[ndof] = nnzNeumann + (N == 'F');
            neumann = new K[nnzNeumann];
            MatNeumann = new HPDDM::MatrixCSR<K>(ndof, ndof, nnzNeumann, neumann, in, jn, false, true);
            for(int j = jStart, k = 0, nnzNeumann = 0; j < jEnd; ++j) {
                for(int i = iStart; i < iEnd; ++i) {
                    if(j > jStart) {
                        neumann[nnzNeumann] = -1 / (dy * dy) + (i == iStart ? -1 / (dx * dx) : 0);
                        jn[nnzNeumann++] = k - (Nx / xGrid) + (N == 'F');
                    }
                    if(i > iStart) {
                        neumann[nnzNeumann] = -1 / (dx * dx) + (j == jStart ? -1 / (dy * dy) : 0);
                        jn[nnzNeumann++] = k - (N == 'C');
                    }
                    neumann[nnzNeumann]  = 2 / (dx * dx) + 2 / (dy * dy);
                    jn[nnzNeumann++] = k + (N == 'F');
                    if(i < iEnd - 1) {
                        neumann[nnzNeumann] = -1 / (dx * dx) + (j == jEnd - 1 ? -1 / (dy * dy) : 0);
                        jn[nnzNeumann++] = k + 1 + (N == 'F');
                    }
                    if(j < jEnd - 1) {
                        neumann[nnzNeumann] = -1 / (dy * dy) + (i == iEnd - 1 ? -1 / (dx * dx) : 0);
                        jn[nnzNeumann++] = k + (Nx / xGrid) + (N == 'F');
                    }
                    in[++k] = nnzNeumann + (N == 'F');
                }
            }
        }
        else {
            in = new int[ndof + 1];
            std::copy_n(ia, ndof + 1, in);
            jn = new int[nnz];
            std::copy_n(ja, nnz, jn);
            neumann = new K[nnz];
            std::copy_n(a, nnz, neumann);
            MatNeumann = new HPDDM::MatrixCSR<K>(ndof, ndof, nnz, neumann, in, jn, false, true);
            for(int j = jStart, nnz = 0; j < jEnd; ++j)
                for(int i = iStart; i < iEnd; ++i) {
                    if(j > jStart) {
                        if(i == iStart)
                            neumann[nnz] -= 1 / (dx * dx);
                        ++nnz;
                    }
                    if(i > iStart) {
                        if(j == jStart)
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
    Mat = new HPDDM::MatrixCSR<K>(ndof, ndof, nnz, a, ia, ja, sym, true);
    if(sizeWorld > 1) {
        for(int k = 0; k < sizeWorld; ++k) {
            if(k == rankWorld) {
                std::cout << rankWorld << ":" << std::endl;
                std::cout << x << "x" << y << " -- [" << iStart << " ; " << iEnd << "] x [" << jStart << " ; " << jEnd << "] -- " << ndof << ", " << nnz << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}
