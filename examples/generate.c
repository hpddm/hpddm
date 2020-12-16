/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
              Ralf Deiterding <r.deiterding@soton.ac.uk>
        Date: 2015-10-29

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

#include <time.h>
#include "schwarz.h"

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

#define xx(i) (xdim[0] + dx * (i + 0.5))
#define yy(j) (ydim[0] + dy * (j + 0.5))

void generate(int rankWorld, int sizeWorld, int* neighbors, int* o, int* sizes, int** connectivity, int* ndof, HpddmMatrixCSR** Mat, HpddmMatrixCSR** MatNeumann, underlying_type** d, K** f, K** sol) {
    const HpddmOption* const opt = HpddmOptionGet();
    const int Nx = HpddmOptionApp(opt, "Nx");
    const int Ny = HpddmOptionApp(opt, "Ny");
    const int overlap = HpddmOptionApp(opt, "overlap");
    const unsigned short mu = HpddmOptionApp(opt, "generate_random_rhs");
    const bool sym = HpddmOptionApp(opt, "symmetric_csr");
    int xGrid = (int)sqrt(sizeWorld);
    while(sizeWorld % xGrid != 0)
        --xGrid;
    int yGrid = sizeWorld / xGrid;

    int y = rankWorld / xGrid;
    int x = rankWorld - xGrid * y;
    int iStart = MAX(x * Nx / xGrid - overlap, 0);
    int iEnd   = MIN((x + 1) * Nx / xGrid + overlap, Nx);
    int jStart = MAX(y * Ny / yGrid - overlap, 0);
    int jEnd   = MIN((y + 1) * Ny / yGrid + overlap, Ny);
    *ndof      = (iEnd - iStart) * (jEnd - jStart);
    int nnz = *ndof * 3 - (iEnd - iStart) - (jEnd - jStart);
    /*# InitEnd #*/
    if(!sym)
        nnz = 2 * nnz - *ndof;
    *f = malloc(sizeof(K) * MAX(1, mu) * *ndof);
    *sol = calloc(MAX(1, mu) * *ndof, sizeof(K));
    underlying_type xdim[2] = { 0.0, 10.0 };
    underlying_type ydim[2] = { 0.0, 10.0 };
    underlying_type dx = (xdim[1] - xdim[0]) / (underlying_type)(Nx);
    underlying_type dy = (ydim[1] - ydim[0]) / (underlying_type)(Ny);
    if(mu == 0) {
        int Nf = 3;
        underlying_type xsc[3] = { 6.5, 2.0, 7.0 };
        underlying_type ysc[3] = { 8.0, 7.0, 3.0 };
        underlying_type rsc[3] = { 0.3, 0.3, 0.4 };
        underlying_type asc[3] = { 0.3, 0.2, -0.1 };
        for(int j = jStart, k = 0; j < jEnd; ++j) {
            for(int i = iStart; i < iEnd; ++i, ++k) {
                underlying_type frs = 1.0;
                for(int n = 0; n < Nf; ++n) {
                    underlying_type xdist = (xx(i) - xsc[n]), ydist = (yy(j) - ysc[n]);
                    if(sqrt(xdist * xdist + ydist * ydist) <= rsc[n])
                        frs -= asc[n] * cos(0.5 * M_PI * xdist / rsc[n]) * cos(0.5 * M_PI * ydist / rsc[n]);
                    (*f)[k] = frs;
                }
            }
        }
    }
    else {
        int rank, size, length;
        char name[MPI_MAX_PROCESSOR_NAME];
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Get_processor_name(name, &length);
        srand((unsigned)time(NULL)+ rank * size + length);
        underlying_type* pt = (underlying_type*)*f;
        for(unsigned i = 0; i < mu * (sizeof(K) / sizeof(underlying_type)) * *ndof; ++i)
            pt[i] = (rand() % 10000) / 10000.0;
    }
    /*# Structures #*/
    *d = malloc(sizeof(underlying_type) * *ndof);
    for(int i = 0; i < *ndof; ++i)
        (*d)[i] = 1.0;
    if(jStart != 0) {
        if(iStart != 0) {
            o[*neighbors] = rankWorld - xGrid - 1;
            sizes[*neighbors] = 4 * overlap * overlap;
            connectivity[*neighbors] = malloc(sizeof(int) * 4 * overlap * overlap);
            for(int j = 0, k = 0; j < 2 * overlap; ++j)
                for(int i = iStart; i < iStart + 2 * overlap; ++i)
                    connectivity[*neighbors][k++] = i - iStart + (iEnd - iStart) * j;
            for(int j = 0; j < overlap; ++j) {
                for(int i = 0; i < overlap - j; ++i)
                    (*d)[i + j + j * (iEnd - iStart)] = j / (underlying_type)(overlap);
                for(int i = 0; i < j; ++i)
                    (*d)[i + j * (iEnd - iStart)] = i / (underlying_type)(overlap);
            }
            (*neighbors)++;
        }
        else
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    (*d)[i + j * (iEnd - iStart)] = j / (underlying_type)(overlap);
        o[*neighbors] = rankWorld - xGrid;
        sizes[*neighbors] = 2 * overlap * (iEnd - iStart);
        connectivity[*neighbors] = malloc(sizeof(int) * 2 * overlap * (iEnd - iStart));
        for(int j = 0, k = 0; j < 2 * overlap; ++j)
            for(int i = iStart; i < iEnd; ++i)
                connectivity[*neighbors][k++] = i - iStart + (iEnd - iStart) * j;
        for(int j = 0; j < overlap; ++j)
            for(int i = iStart + overlap; i < iEnd - overlap; ++i)
                (*d)[i - iStart + (iEnd - iStart) * j] = j / (underlying_type)(overlap);
        (*neighbors)++;
        if(iEnd != Nx) {
            o[*neighbors] = rankWorld - xGrid + 1;
            sizes[*neighbors] = 4 * overlap * overlap;
            connectivity[*neighbors] = malloc(sizeof(int) * 4 * overlap * overlap);
            for(int i = 0, k = 0; i < 2 * overlap; ++i)
                for(int j = 0; j < 2 * overlap; ++j)
                    connectivity[*neighbors][k++] = (iEnd - iStart) * (i + 1) - 2 * overlap + j;
            for(int j = 0; j < overlap; ++j) {
                for(int i = 0; i < overlap - j; ++i)
                    (*d)[(iEnd - iStart) * (j + 1) - overlap + i] = j / (underlying_type)(overlap);
                for(int i = 0; i < j; ++i)
                    (*d)[(iEnd - iStart) * (j + 1) - i - 1] = i / (underlying_type)(overlap);
            }
            (*neighbors)++;
        }
        else
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    (*d)[(iEnd - iStart) * (j + 1) - overlap + i] = j / (underlying_type)(overlap);
    }
    /*# StructuresEnd #*/
    if(iStart != 0) {
        o[*neighbors] = rankWorld - 1;
        sizes[*neighbors] = 2 * overlap * (jEnd - jStart);
        connectivity[*neighbors] = malloc(sizeof(int) * 2 * overlap * (jEnd - jStart));
        for(int i = jStart, k = 0; i < jEnd; ++i)
            for(int j = 0; j < 2 * overlap; ++j)
                connectivity[*neighbors][k++] = j + (i - jStart) * (iEnd - iStart);
        for(int i = jStart + (jStart != 0) * overlap; i < jEnd - (jEnd != Ny) * overlap; ++i)
            for(int j = 0; j < overlap; ++j)
                (*d)[j + (i - jStart) * (iEnd - iStart)] = j / (underlying_type)(overlap);
        (*neighbors)++;
    }
    if(iEnd != Nx) {
        o[*neighbors] = rankWorld + 1;
        sizes[*neighbors] = 2 * overlap * (jEnd - jStart);
        connectivity[*neighbors] = malloc(sizeof(int) * 2 * overlap * (jEnd - jStart));
        for(int i = jStart, k = 0; i < jEnd; ++i)
            for(int j = 0; j < 2 * overlap; ++j)
                connectivity[*neighbors][k++] = (iEnd - iStart) * (i + 1 - jStart) - 2 * overlap + j;
        for(int i = jStart + (jStart != 0) * overlap; i < jEnd - (jEnd != Ny) * overlap; ++i)
            for(int j = 0; j < overlap; ++j)
                (*d)[(iEnd - iStart) * (i + 1 - jStart) - j - 1] = j / (underlying_type)(overlap);
        (*neighbors)++;
    }
    if(jEnd != Ny) {
        if(iStart != 0) {
            o[*neighbors] = rankWorld + xGrid - 1;
            sizes[*neighbors] = 4 * overlap * overlap;
            connectivity[*neighbors] = malloc(sizeof(int) * 4 * overlap * overlap);
            for(int j = 0, k = 0; j < 2 * overlap; ++j)
                for(int i = iStart; i < iStart + 2 * overlap; ++i)
                    connectivity[*neighbors][k++] = *ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j;
            for(int j = 0; j < overlap; ++j) {
                for(int i = 0; i < overlap - j; ++i)
                    (*d)[*ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * j] = i / (underlying_type)(overlap);
                for(int i = overlap - j; i < overlap; ++i)
                    (*d)[*ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * j] = (overlap - 1 - j) / (underlying_type)(overlap);
            }
            (*neighbors)++;
        }
        else {
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    (*d)[*ndof - overlap * (iEnd - iStart) + (iEnd - iStart) * j + i] = (overlap - j - 1) / (underlying_type)(overlap);
        }
        o[*neighbors] = rankWorld + xGrid;
        sizes[*neighbors] = 2 * overlap * (iEnd - iStart);
        connectivity[*neighbors] = malloc(sizeof(int) * 2 * overlap * (iEnd - iStart));
        for(int j = 0, k = 0; j < 2 * overlap; ++j)
            for(int i = iStart; i < iEnd; ++i)
                connectivity[*neighbors][k++] = *ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j;
        for(int j = 0; j < overlap; ++j)
            for(int i = iStart + overlap; i < iEnd - overlap; ++i)
                (*d)[*ndof - overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j] = (overlap - 1 - j) / (underlying_type)(overlap);
        (*neighbors)++;
        if(iEnd != Nx) {
            o[*neighbors] = rankWorld + xGrid + 1;
            sizes[*neighbors] = 4 * overlap * overlap;
            connectivity[*neighbors] = malloc(sizeof(int) * 4 * overlap * overlap);
            for(int j = 0, k = 0; j < 2 * overlap; ++j)
                for(int i = iStart; i < iStart + 2 * overlap; ++i)
                    connectivity[*neighbors][k++] = *ndof - 2 * overlap * (iEnd - iStart) + i - iStart + (iEnd - iStart) * j + (iEnd - iStart - 2 * overlap);
            for(int j = 0; j < overlap; ++j) {
                for(int i = j; i < overlap; ++i)
                    (*d)[*ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - 1 - i) / (underlying_type)(overlap);
                for(int i = 0; i < j; ++i)
                    (*d)[*ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - 1 - j) / (underlying_type)(overlap);
            }
            (*neighbors)++;
        }
        else {
            for(int j = 0; j < overlap; ++j)
                for(int i = 0; i < overlap; ++i)
                    (*d)[*ndof - overlap * (iEnd - iStart) + i + (iEnd - iStart) * (j + 1) - overlap] = (overlap - j - 1) / (underlying_type)(overlap);
        }
    }

    int* in = NULL, *jn = NULL;
    K* neumann = NULL;
    char N = HPDDM_NUMBERING;
    int* ia = malloc(sizeof(int) * (*ndof + 1));
    int* ja = malloc(sizeof(int) * nnz);
    K* a = malloc(sizeof(K) * nnz);
    ia[0] = (N == 'F');
    ia[*ndof] = nnz + (N == 'F');
    if(sym) {
        /*# Matrix #*/
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
    if(sizeWorld > 1 && HpddmOptionSet(opt, "schwarz_coarse_correction") && HpddmOptionVal(opt, "geneo_nu") > 0) {
        if(sym) {
            int nnzNeumann = 2 * nnz - *ndof;
            in = malloc(sizeof(int) * (*ndof + 1));
            jn = malloc(sizeof(int) * nnzNeumann);
            in[0] = (N == 'F');
            in[*ndof] = nnzNeumann + (N == 'F');
            neumann = malloc(sizeof(K) * nnzNeumann);
            *MatNeumann = HpddmMatrixCSRCreate(*ndof, *ndof, nnzNeumann, neumann, in, jn, false, true);
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
            in = malloc(sizeof(int) * (*ndof + 1));
            memcpy(in, ia, sizeof(int) * (*ndof + 1));
            jn = malloc(sizeof(int) * nnz);
            memcpy(jn, ja, sizeof(int) * nnz);
            neumann = malloc(sizeof(K) * nnz);
            memcpy(neumann, a, sizeof(K) * nnz);
            *MatNeumann = HpddmMatrixCSRCreate(*ndof, *ndof, nnz, neumann, in, jn, false, true);
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
    *Mat = HpddmMatrixCSRCreate(*ndof, *ndof, nnz, a, ia, ja, sym, true);
    if(sizeWorld > 1) {
        for(int k = 0; k < sizeWorld; ++k) {
            if(k == rankWorld) {
                printf("%d:\n", rankWorld);
                printf("%dx%d -- [%d ; %d] x [%d ; %d] -- %d, %d\n", x, y, iStart, iEnd, jStart, jEnd, *ndof, nnz);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}
