/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2020-05-07

   Copyright (C) 2020-     Centre National de la Recherche Scientifique

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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <HPDDM.h>

struct HpddmCustomOperator {
    MPI_Comm comm;
    int n;
};

int mv(const HpddmCustomOperator* const op, const double* in, double* out, int mu) {
    int rank;
    MPI_Comm_rank(op->comm, &rank);
    for(int j = 0; j < mu; ++j)
        for(int i = 0; i < op->n; ++i) {
            out[op->n * j + i] = (op->n * rank + i + 2.0) * in[op->n * j + i];
            if(i > 0)
                out[op->n * j + i] -= 0.5 * in[op->n * j + i - 1];
            if(i < op->n - 1)
                out[op->n * j + i] -= 0.5 * in[op->n * j + i + 1];
        }
    return 0;
}
int apply(const HpddmCustomOperator* const op, const double* in, double* out, int mu) {
    int rank;
    MPI_Comm_rank(op->comm, &rank);
    for(int j = 0; j < mu; ++j)
        for(int i = 0; i < op->n; ++i)
            out[op->n * j + i] = in[op->n * j + i] / (op->n * rank + i + 2.0);
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const HpddmOption* const opt = HpddmOptionGet();
    HpddmOptionParse(opt, argc, argv, 0);
    HpddmOptionParseString(opt, "-hpddm_verbosity 4");
    {
        char* val = "mu=<2>";
        char* desc = "Number of generated random right-hand sides.";
        HpddmOptionParseInt(opt, argc, argv, val, desc);
        val = "n=<100>";
        desc = "Size of the local matrices.";
        HpddmOptionParseInt(opt, argc, argv, val, desc);
    }
    int mu = HpddmOptionApp(opt, "mu");
    if(mu < 1)
        mu = 2;
    HpddmCustomOperator* op = (HpddmCustomOperator*)malloc(sizeof(HpddmCustomOperator));
    op->n = HpddmOptionApp(opt, "n");
    if(op->n < 1)
        op->n = 1;
    MPI_Comm_dup(MPI_COMM_WORLD, &op->comm);
    int rank, size, length;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_rank(op->comm, &rank);
    if(rank != 0)
        HpddmOptionRemove(opt, "verbosity");
    MPI_Comm_size(op->comm, &size);
    MPI_Get_processor_name(name, &length);
    srand((unsigned)time(NULL)+ rank * size + length);
    double* in = malloc(sizeof(double) * mu * op->n);
    for(int i = 0; i < mu * op->n; ++i)
        in[i] = (rand() % 10000) / 100.0;
    double* out = (double*)calloc(mu * op->n, sizeof(double));
    HpddmCustomOperatorSolve(op, op->n, &mv, &apply, in, out, mu, &op->comm);
    MPI_Comm_free(&op->comm);
    free(out);
    free(in);
    free(op);
    MPI_Finalize();
    return 0;
}
