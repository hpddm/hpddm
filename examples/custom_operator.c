/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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
#include <stdint.h>

#include <HPDDM.h>

struct HpddmCustomOperator {
  MPI_Comm comm;
  int      n;
};

int mv(const HpddmCustomOperator *const op, const double *in, double *out, int mu)
{
  int rank;
  MPI_Comm_rank(op->comm, &rank);
  for (int j = 0; j < mu; ++j)
    for (int i = 0; i < op->n; ++i) {
      out[op->n * j + i] = (op->n * rank + i + 2.0) * in[op->n * j + i];
      if (i > 0) out[op->n * j + i] -= 0.5 * in[op->n * j + i - 1];
      if (i < op->n - 1) out[op->n * j + i] -= 0.5 * in[op->n * j + i + 1];
    }
  return 0;
}
int apply(const HpddmCustomOperator *const op, const double *in, double *out, int mu)
{
  int rank;
  MPI_Comm_rank(op->comm, &rank);
  for (int j = 0; j < mu; ++j)
    for (int i = 0; i < op->n; ++i) out[op->n * j + i] = in[op->n * j + i] / (op->n * rank + i + 2.0);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  const HpddmOption *const opt = HpddmOptionGet();
  HpddmOptionParse(opt, argc, argv, 0);
  HpddmOptionParseString(opt, "-hpddm_verbosity 4");
  {
    char *val  = "mu=<2>";
    char *desc = "Number of right-hand sides.";
    HpddmOptionParseInt(opt, argc, argv, val, desc);
    val  = "n=<100>";
    desc = "Size of the local matrices.";
    HpddmOptionParseInt(opt, argc, argv, val, desc);
  }
  int mu = HpddmOptionApp(opt, "mu");
  if (mu < 1) mu = 2;
  HpddmCustomOperator *op = (HpddmCustomOperator *)malloc(sizeof(HpddmCustomOperator));
  op->n                   = HpddmOptionApp(opt, "n");
  if (op->n < 0) op->n = 1;
  const int n = op->n;
  MPI_Comm_dup(MPI_COMM_WORLD, &op->comm);
  int rank;
  MPI_Comm_rank(op->comm, &rank);
  if (rank != 0) HpddmOptionRemove(opt, "verbosity");
  size_t   size = n ? (size_t)mu * n : 1;
  double  *in   = malloc(sizeof(double) * size);
  uint32_t seed = 17 + rank;
  for (int i = 0; i < mu * n; ++i) {
    seed  = 1664525U * seed + 1013904223U;
    in[i] = (seed % 10000) / 100.0;
  }
  double *out = (double *)calloc(size, sizeof(double));
  HpddmCustomOperatorSolve(op, op->n, &mv, &apply, in, out, mu, &op->comm);
  double                   *res = malloc(sizeof(double) * size), norms[2] = {0.0, 0.0};
  const HpddmCustomOperator residual = {op->comm, n};
  mv(&residual, out, res, mu);
  for (int i = 0; i < mu * n; ++i) {
    norms[0] += (res[i] - in[i]) * (res[i] - in[i]);
    norms[1] += in[i] * in[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM, op->comm);
  if (rank == 0) printf(" --- squared residual = %e / %e\n", norms[0], norms[1]);
  int status = !(norms[0] <= 1.0e-10 * norms[1]);
  free(res);
  MPI_Comm_free(&op->comm);
  free(out);
  free(in);
  free(op);
  MPI_Finalize();
  return status;
}
