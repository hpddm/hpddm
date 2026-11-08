#define HPDDM_MIXED_PRECISION 1
#include <petsc/private/petschpddm.h>
#include <petsc/private/deviceimpl.h>
#include <petscdevice_cuda.h>
#include <petscsf.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

#include <petsc/private/cupmobject.hpp>

/* Arnoldi, Givens rotations, and the least-squares solve stay on the device.
   Only residual norms are read back for PETSc monitors and convergence tests. */
static __global__ void KSPHPDDMGMRESArnoldiKernel(PetscInt mu, PetscInt m, PetscInt i, const unsigned char *locked, PetscScalar *coefficients, PetscScalar *H, PetscScalar *rhs, PetscReal *sines, PetscReal *callback)
{
  const PetscInt nu = static_cast<PetscInt>(blockIdx.x * blockDim.x + threadIdx.x);

  if (nu < mu) {
    PetscScalar *const block    = H + static_cast<size_t>(nu) * m * (m + 1);
    PetscScalar *const column   = block + static_cast<size_t>(i) * (m + 1);
    PetscScalar *const residual = rhs + static_cast<size_t>(nu) * (m + 1);
    PetscScalar        cosine;
    PetscReal *const   sine = sines + static_cast<size_t>(nu) * m;
    PetscReal          norm, delta;
    bool               inactive;

    if (locked[nu]) {
      callback[nu] = 0.0;
      return;
    }
    for (PetscInt row = 0; row <= i; ++row) column[row] = coefficients[static_cast<size_t>(row) * mu + nu];
    norm                                               = PetscSqrtReal(PetscRealPart(coefficients[static_cast<size_t>(i + 1) * mu + nu]));
    column[i + 1]                                      = norm;
    coefficients[static_cast<size_t>(i + 1) * mu + nu] = norm == 0.0 ? 1.0 : 1.0 / norm;
    for (PetscInt k = 0; k < i; ++k) {
      const PetscScalar previous = block[static_cast<size_t>(k) * (m + 1) + k + 1];
      const PetscScalar gamma    = PetscConj(previous) * column[k] + sine[k] * column[k + 1];

      column[k + 1] = -sine[k] * column[k] + previous * column[k + 1];
      column[k]     = gamma;
    }
    delta           = PetscHypotReal(PetscAbsScalar(column[i]), PetscAbsScalar(column[i + 1]));
    inactive        = delta == 0.0 && residual[i] == PetscScalar(0.0);
    sine[i]         = inactive ? 0.0 : PetscRealPart(column[i + 1]) / delta;
    cosine          = inactive ? PetscScalar(1.0) : column[i] / delta;
    column[i]       = delta;
    column[i + 1]   = cosine;
    residual[i + 1] = -sine[i] * residual[i];
    residual[i] *= PetscConj(cosine);
    callback[nu] = PetscAbsScalar(residual[i + 1]);
  }
}

template <class K>
class KSPHPDDMGMRESCUDA : Petsc::device::cupm::impl::CUPMObject<Petsc::device::cupm::DeviceType::CUDA> {
  KSP                        ksp_       = nullptr;
  PetscDeviceContext         context_   = nullptr;
  cupmBlasHandle_t           handle_    = nullptr;
  cupmStream_t               stream_    = nullptr;
  MPI_Comm                   comm_      = MPI_COMM_NULL;
  PetscMPIInt                size_      = 1;
  PetscInt                   restart_   = 0;
  PetscSF                    reduction_ = nullptr;
  K                         *reduced_   = nullptr;
  std::vector<MPI_Datatype>  reductionUnits_;
  K                         *vectors_ = nullptr, *coefficients_ = nullptr, *hessenberg_ = nullptr, *rhs_ = nullptr, *norms_ = nullptr;
  PetscReal                 *sines_ = nullptr, *residuals_ = nullptr;
  unsigned char             *locked_ = nullptr;
  std::vector<K>             hostScalars_;
  std::vector<PetscReal>     hostResiduals_;
  std::vector<unsigned char> hostLocked_;

  static const cupmScalar_t *cast(const K *p) { return reinterpret_cast<const cupmScalar_t *>(p); }
  static cupmScalar_t       *cast(K *p) { return reinterpret_cast<cupmScalar_t *>(p); }
  static PetscScalar        *scalar(K *p) { return reinterpret_cast<PetscScalar *>(p); }
  /* alignof(std::complex<PetscReal>) is only alignof(PetscReal), but the kernels access PetscScalar (thrust::complex). */
  static PetscScalar **deviceScalar(K **p) { return reinterpret_cast<PetscScalar **>(p); }

  PetscErrorCode destroyReduction()
  {
    PetscFunctionBegin;
    PetscCall(PetscSFDestroy(&reduction_));
    for (auto &unit : reductionUnits_)
      if (unit != MPI_DATATYPE_NULL && unit != MPIU_SCALAR) PetscCallMPI(MPI_Type_free(&unit));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode allreduce(K *p, PetscInt count)
  {
    PetscFunctionBegin;
    if (size_ == 1) PetscFunctionReturn(PETSC_SUCCESS);
    if (use_gpu_aware_mpi) {
      if (!reduced_) {
        reductionUnits_.resize(hostScalars_.size() + 1, MPI_DATATYPE_NULL);
        PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, hostScalars_.size(), PETSC_DECIDE, deviceScalar(&reduced_)));
        /* This one-block graph is independent of the RHS count and restart length. */
        PetscCall(PetscObjectQuery((PetscObject)ksp_, "_HPDDM_CUDA_GMRES_reduction", (PetscObject *)&reduction_));
        if (reduction_) PetscCall(PetscObjectReference((PetscObject)reduction_));
        else {
          PetscSFNode remote = {0, 0};

          PetscCall(PetscSFCreate(comm_, &reduction_));
          PetscCall(PetscSFSetType(reduction_, PETSCSFBASIC));
          PetscCall(PetscSFSetGraph(reduction_, 1, 1, nullptr, PETSC_COPY_VALUES, &remote, PETSC_COPY_VALUES));
          PetscCall(PetscSFSetUp(reduction_));
          PetscCall(PetscObjectCompose((PetscObject)ksp_, "_HPDDM_CUDA_GMRES_reduction", (PetscObject)reduction_));
        }
      }
      auto &unit = reductionUnits_[count];

      if (count == 1) unit = MPIU_SCALAR;
      else if (unit == MPI_DATATYPE_NULL) {
        PetscMPIInt mpiCount;

        PetscCall(PetscMPIIntCast(count, &mpiCount));
        PetscCallMPI(MPI_Type_contiguous(mpiCount, MPIU_SCALAR, &unit));
        PetscCallMPI(MPI_Type_commit(&unit));
      }
      /* CUDA-aware MPI reductions may still stage through the host. Basic SF
         transports device buffers and performs the summation with CUDA kernels. */
      PetscCallCUDA(cudaMemsetAsync(reduced_, 0, count * sizeof(K), stream_));
      PetscCall(PetscSFReduceWithMemTypeBegin(reduction_, unit, PETSC_MEMTYPE_DEVICE, p, PETSC_MEMTYPE_DEVICE, reduced_, MPIU_SUM));
      PetscCall(PetscSFReduceEnd(reduction_, unit, p, reduced_, MPIU_SUM));
      PetscCall(PetscSFBcastWithMemTypeBegin(reduction_, unit, PETSC_MEMTYPE_DEVICE, reduced_, PETSC_MEMTYPE_DEVICE, p, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(reduction_, unit, reduced_, p, MPI_REPLACE));
    } else {
      PetscCallCUDA(cudaMemcpyAsync(hostScalars_.data(), p, count * sizeof(K), cudaMemcpyDeviceToHost, stream_));
      PetscCallCUDA(cudaStreamSynchronize(stream_));
      PetscCall(PetscLogGpuToCpu(count * sizeof(K)));
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, hostScalars_.data(), count, MPIU_SCALAR, MPIU_SUM, comm_));
      PetscCallCUDA(cudaMemcpyAsync(p, hostScalars_.data(), count * sizeof(K), cudaMemcpyHostToDevice, stream_));
      PetscCall(PetscLogCpuToGpu(count * sizeof(K)));
      /* The staging buffer is reused by the next reduction. */
      PetscCallCUDA(cudaStreamSynchronize(stream_));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode dotSquared(int n, const K *x, K *dots, int mu)
  {
    CUPMBlasPointerModeGuard guard(handle_, CUPMBLAS_POINTER_MODE_DEVICE);

    PetscFunctionBegin;
    if (n) {
      for (int nu = 0; nu < mu; ++nu) PetscCallCUBLAS(cupmBlasXdot(handle_, n, cast(x + nu * n), 1, cast(x + nu * n), 1, cast(dots + nu)));
      PetscCall(PetscLogGpuFlops((2.0 * n - 1) * mu));
    } else PetscCallCUDA(cudaMemsetAsync(dots, 0, mu * sizeof(K), stream_));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode orthogonalization(char id, int n, int k, int mu, const K *basis, K *v)
  {
    CUPMBlasPointerModeGuard guard(handle_, CUPMBLAS_POINTER_MODE_HOST);
    const K                  one(1), zero(0), minusOne(-1);

    PetscFunctionBegin;
    if (id == HPDDM_ORTHOGONALIZATION_MGS) {
      for (int i = 0; i < k; ++i) {
        if (n) {
          for (int nu = 0; nu < mu; ++nu) PetscCallCUBLAS(cupmBlasXgemv(handle_, CUPMBLAS_OP_C, n, 1, cast(&minusOne), cast(basis + (i * mu + nu) * n), n, cast(v + nu * n), 1, cast(&zero), cast(coefficients_ + i * mu + nu), 1));
        } else PetscCallCUDA(cudaMemsetAsync(coefficients_ + i * mu, 0, mu * sizeof(K), stream_));
        PetscCall(allreduce(coefficients_ + i * mu, mu));
        if (n) {
          CUPMBlasPointerModeGuard deviceGuard(handle_, CUPMBLAS_POINTER_MODE_DEVICE);

          for (int nu = 0; nu < mu; ++nu) PetscCallCUBLAS(cupmBlasXaxpy(handle_, n, cast(coefficients_ + i * mu + nu), cast(basis + (i * mu + nu) * n), 1, cast(v + nu * n), 1));
        }
      }
      PetscCallCUBLAS(cupmBlasXscal(handle_, k * mu, cast(&minusOne), cast(coefficients_), 1));
    } else {
      if (n) {
        for (int nu = 0; nu < mu; ++nu) PetscCallCUBLAS(cupmBlasXgemv(handle_, CUPMBLAS_OP_C, n, k, cast(&one), cast(basis + nu * n), mu * n, cast(v + nu * n), 1, cast(&zero), cast(coefficients_ + nu), mu));
      } else PetscCallCUDA(cudaMemsetAsync(coefficients_, 0, k * mu * sizeof(K), stream_));
      PetscCall(allreduce(coefficients_, k * mu));
      if (n)
        for (int nu = 0; nu < mu; ++nu) PetscCallCUBLAS(cupmBlasXgemv(handle_, CUPMBLAS_OP_N, n, k, cast(&minusOne), cast(basis + nu * n), mu * n, cast(coefficients_ + nu), mu, cast(&one), cast(v + nu * n), 1));
    }
    PetscCall(PetscLogGpuFlops(4.0 * n * k * mu));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

public:
  ~KSPHPDDMGMRESCUDA()
  {
    PetscCallContinue(destroyReduction());
    PetscCallContinue(PetscDeviceFree(context_, reduced_));
    PetscCallContinue(PetscDeviceFree(context_, vectors_));
    PetscCallContinue(PetscDeviceFree(context_, coefficients_));
    PetscCallContinue(PetscDeviceFree(context_, hessenberg_));
    PetscCallContinue(PetscDeviceFree(context_, rhs_));
    PetscCallContinue(PetscDeviceFree(context_, norms_));
    PetscCallContinue(PetscDeviceFree(context_, sines_));
    PetscCallContinue(PetscDeviceFree(context_, residuals_));
    PetscCallContinue(PetscDeviceFree(context_, locked_));
  }

  PetscErrorCode setup(PetscInt restart, PetscInt mu, KSP ksp)
  {
    const MPI_Comm comm = PetscObjectComm((PetscObject)ksp);

    PetscFunctionBegin;
    static_assert(sizeof(K) == sizeof(PetscScalar), "Matching scalar precision required");
    PetscCall(GetHandles_(&context_, &handle_, &stream_));
    PetscCallMPI(MPI_Comm_size(comm, &size_));
    ksp_     = ksp;
    comm_    = comm;
    restart_ = restart;
    hostScalars_.resize(static_cast<size_t>(restart + 1) * mu);
    hostResiduals_.resize(mu);
    hostLocked_.assign(mu, 0);
    PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, static_cast<size_t>(restart + 1) * mu, PETSC_DECIDE, deviceScalar(&coefficients_)));
    PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, static_cast<size_t>(restart + 1) * restart * mu, PETSC_DECIDE, deviceScalar(&hessenberg_)));
    PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, static_cast<size_t>(restart + 1) * mu, PETSC_DECIDE, deviceScalar(&rhs_)));
    PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, mu, PETSC_DECIDE, deviceScalar(&norms_)));
    PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, static_cast<size_t>(restart) * mu, PETSC_DECIDE, &sines_));
    PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, mu, PETSC_DECIDE, &residuals_));
    PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, mu, PETSC_DECIDE, &locked_));
    PetscCallCUDA(cudaMemsetAsync(locked_, 0, mu * sizeof(*locked_), stream_));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode allocate(K *&p, size_t count)
  {
    PetscFunctionBegin;
    PetscCall(PetscDeviceMalloc(context_, PETSC_MEMTYPE_DEVICE, PetscMax(count, size_t(1)), PETSC_DECIDE, deviceScalar(&vectors_)));
    p = vectors_;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode release(K *&p)
  {
    PetscFunctionBegin;
    PetscCall(PetscDeviceFree(context_, vectors_));
    p = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode scale(int n, K alpha, K *x)
  {
    CUPMBlasPointerModeGuard guard(handle_, CUPMBLAS_POINTER_MODE_HOST);

    PetscFunctionBegin;
    if (n) {
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUBLAS(cupmBlasXscal(handle_, n, cast(&alpha), cast(x), 1));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(PetscLogGpuFlops(n));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode zero(int n, K *x)
  {
    PetscFunctionBegin;
    if (n) PetscCallCUDA(cudaMemsetAsync(x, 0, n * sizeof(K), stream_));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode axpby(int n, K alpha, const K *x, K beta, K *y)
  {
    CUPMBlasPointerModeGuard guard(handle_, CUPMBLAS_POINTER_MODE_HOST);

    PetscFunctionBegin;
    if (n) {
      PetscCall(PetscLogGpuTimeBegin());
      if (beta == K() && alpha == K(1)) {
        PetscCallCUDA(cudaMemcpyAsync(y, x, n * sizeof(K), cudaMemcpyDeviceToDevice, stream_));
        PetscCall(PetscLogGpuTimeEnd());
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      if (beta == K()) PetscCallCUDA(cudaMemsetAsync(y, 0, n * sizeof(K), stream_));
      else if (beta != K(1)) PetscCallCUBLAS(cupmBlasXscal(handle_, n, cast(&beta), cast(y), 1));
      PetscCallCUBLAS(cupmBlasXaxpy(handle_, n, cast(&alpha), cast(x), 1, cast(y), 1));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(PetscLogGpuFlops((beta == K(1) ? 2.0 : 3.0) * n));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode normSquared(int n, const K *x, HPDDM::underlying_type<K> *norm, int mu, const HPDDM::underlying_type<K> *)
  {
    PetscFunctionBegin;
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(dotSquared(n, x, norms_, mu));
    PetscCallCUDA(cudaMemcpyAsync(hostScalars_.data(), norms_, mu * sizeof(K), cudaMemcpyDeviceToHost, stream_));
    PetscCallCUDA(cudaStreamSynchronize(stream_));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuToCpu(mu * sizeof(K)));
    for (int nu = 0; nu < mu; ++nu) norm[nu] = HPDDM::real(hostScalars_[nu]);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <class Operator>
  PetscErrorCode initializeNorm(const Operator &A, char variant, const K *b, K *x, K *v, int n, K *work, HPDDM::underlying_type<K> *norm, int mu, bool &allocated)
  {
    PetscFunctionBegin;
    allocated = A.template start<false>(b, x, mu);
    if (variant == HPDDM_VARIANT_LEFT) {
      PetscCall(A.template apply<false>(b, v, mu, work));
      PetscCall(normSquared(n, v, norm, mu, nullptr));
    } else PetscCall(normSquared(n, b, norm, mu, nullptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode startCycle(int mu, const K *s)
  {
    PetscFunctionBegin;
    PetscCallCUDA(cudaMemcpy2DAsync(rhs_, (restart_ + 1) * sizeof(K), s, sizeof(K), sizeof(K), mu, cudaMemcpyHostToDevice, stream_));
    PetscCall(PetscLogCpuToGpu(mu * sizeof(K)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode arnoldi(char id, unsigned short m, K *const *, K *const *v, K *s, HPDDM::underlying_type<K> *, int n, int i, int mu, const HPDDM::underlying_type<K> *, K *, MPI_Comm, const short *hasConverged = nullptr)
  {
    bool changed = false;

    PetscFunctionBegin;
    PetscCall(PetscLogEventBegin(KSP_Orthogonalization, nullptr, nullptr, nullptr, nullptr));
    PetscCall(PetscLogGpuTimeBegin());
    for (int nu = 0; nu < mu; ++nu) {
      const unsigned char locked = hasConverged && hasConverged[nu] != -m;

      changed |= hostLocked_[nu] != locked;
      hostLocked_[nu] = locked;
      if (locked) PetscCall(zero(n, v[i + 1] + nu * n));
    }
    if (changed) {
      // Preserve the completed least-squares systems while other columns continue.
      PetscCallCUDA(cudaMemcpyAsync(locked_, hostLocked_.data(), mu * sizeof(*locked_), cudaMemcpyHostToDevice, stream_));
      PetscCall(PetscLogCpuToGpu(mu * sizeof(*locked_)));
    }
    PetscCall(orthogonalization(id & 3, n, i + 1, mu, *v, v[i + 1]));
    PetscCall(dotSquared(n, v[i + 1], coefficients_ + (i + 1) * mu, mu));
    PetscCall(allreduce(coefficients_ + (i + 1) * mu, mu));
    KSPHPDDMGMRESArnoldiKernel<<<PetscCeilInt(mu, 256), 256, 0, stream_>>>(mu, m, i, locked_, scalar(coefficients_), scalar(hessenberg_), scalar(rhs_), sines_, residuals_);
    PetscCallCUDA(cudaGetLastError());
    if (n && i < m - 1) {
      CUPMBlasPointerModeGuard guard(handle_, CUPMBLAS_POINTER_MODE_DEVICE);

      for (int nu = 0; nu < mu; ++nu)
        if (!hostLocked_[nu]) PetscCallCUBLAS(cupmBlasXscal(handle_, n, cast(coefficients_ + (i + 1) * mu + nu), cast(v[i + 1] + nu * n), 1));
      PetscCall(PetscLogGpuFlops(1.0 * n * mu));
    }
    PetscCallCUDA(cudaMemcpyAsync(hostResiduals_.data(), residuals_, mu * sizeof(PetscReal), cudaMemcpyDeviceToHost, stream_));
    PetscCallCUDA(cudaStreamSynchronize(stream_));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuToCpu(mu * sizeof(PetscReal)));
    for (int nu = 0; nu < mu; ++nu) s[(i + 1) * mu + nu] = hostResiduals_[nu];
    PetscCall(PetscLogEventEnd(KSP_Orthogonalization, nullptr, nullptr, nullptr, nullptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <class Operator>
  PetscErrorCode updateSolution(const Operator &A, char variant, int n, K *x, int ldh, K *const *, K *, K *const *v, const short *converged, int mu, K *work)
  {
    CUPMBlasPointerModeGuard guard(handle_, CUPMBLAS_POINTER_MODE_HOST);
    const K                  one(1), zero(0);
    K                       *correction = variant == HPDDM_VARIANT_RIGHT ? v[ldh / mu - 1] : work;

    PetscFunctionBegin;
    PetscCall(PetscLogGpuTimeBegin());
    for (int nu = 0; nu < mu; ++nu) {
      const int dim   = std::abs(converged[nu]);
      K        *coeff = rhs_ + nu * (restart_ + 1);

      if (dim) PetscCallCUBLAS(cupmBlasXtrsv(handle_, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, dim, cast(hessenberg_ + static_cast<size_t>(nu) * restart_ * (restart_ + 1)), restart_ + 1, cast(coeff), 1));
      if (n) {
        K *out = (variant == HPDDM_VARIANT_LEFT ? x : work) + nu * n;

        if (dim) PetscCallCUBLAS(cupmBlasXgemv(handle_, CUPMBLAS_OP_N, n, dim, cast(&one), cast(*v + nu * n), mu * n, cast(coeff), 1, cast(variant == HPDDM_VARIANT_LEFT ? &one : &zero), cast(out), 1));
        else if (variant != HPDDM_VARIANT_LEFT) PetscCallCUDA(cudaMemsetAsync(out, 0, n * sizeof(K), stream_));
      }
      PetscCall(PetscLogGpuFlops(2.0 * n * dim + 1.0 * dim * dim));
    }
    PetscCall(PetscLogGpuTimeEnd());
    if (variant == HPDDM_VARIANT_RIGHT) PetscCall(A.template apply<false>(work, correction, mu));
    if (variant != HPDDM_VARIANT_LEFT)
      for (int nu = 0; nu < mu; ++nu)
        if (converged[nu]) PetscCall(axpby(n, one, correction + nu * n, one, x + nu * n));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

PetscErrorCode KSPSolve_HPDDM_CUDA_Private(KSP_HPDDM *data, const PetscScalar *b, PetscScalar *x, PetscInt n, MPI_Comm comm)
{
  const PetscInt     N = data->op->getDof() * n;
  PetscDeviceContext dctx;
  cudaStream_t      *stream;
  using K = std::conditional_t<PetscDefined(USE_REAL_DOUBLE), HPDDM::downscaled_type<PetscScalar>, HPDDM::upscaled_type<PetscScalar>>;

  PetscFunctionBegin;
  if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GMRES && data->precision == PETSC_SCALAR_PRECISION) {
#if PetscDefined(USE_COMPLEX)
    using Scalar = std::complex<PetscReal>;
#else
    using Scalar = PetscScalar;
#endif
    KSPHPDDMGMRESCUDA<Scalar> operations;

    if (!data->cudaop) data->cudaop = new HPDDM::PETScCUDAGMRESOperator(data->op->ksp_, data->op->getDof());

    PetscCall(operations.setup(data->scntl[0], n, data->op->ksp_));
    PetscCall(PetscErrorCode(HPDDM::IterativeMethod::GMRES<false>(*data->cudaop, reinterpret_cast<const Scalar *>(b), reinterpret_cast<Scalar *>(x), n, comm, operations)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, (void **)&stream));
  if (data->precision != PETSC_SCALAR_PRECISION) {
    const thrust::device_ptr<const PetscScalar> db = thrust::device_pointer_cast(b);
    const thrust::device_ptr<PetscScalar>       dx = thrust::device_pointer_cast(x);
    K                                          *ptr, *host_ptr;
    thrust::device_ptr<K>                       dptr[2];

    PetscCall(PetscMalloc1(2 * N, &host_ptr));
    PetscCallCUDA(cudaMalloc((void **)&ptr, 2 * N * sizeof(K)));
    dptr[0] = thrust::device_pointer_cast(ptr);
    dptr[1] = thrust::device_pointer_cast(ptr + N);
    PetscCallCXX(thrust::copy_n(thrust::cuda::par.on(*stream), db, N, dptr[0]));
    PetscCallCXX(thrust::copy_n(thrust::cuda::par.on(*stream), dx, N, dptr[1]));
    PetscCallCUDA(cudaMemcpyAsync(host_ptr, ptr, 2 * N * sizeof(K), cudaMemcpyDeviceToHost, *stream));
    PetscCall(PetscDeviceContextSynchronize(dctx));
#if PetscDefined(USE_COMPLEX)
    /* reinterpret thrust::complex<> as std::complex<> so that HPDDM deduces a type with BLAS/LAPACK and MPI support */
    std::complex<HPDDM::underlying_type<K>> *const hb = reinterpret_cast<std::complex<HPDDM::underlying_type<K>> *>(host_ptr);
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, hb, hb + N, n, comm));
#else
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, host_ptr, host_ptr + N, n, comm));
#endif
    PetscCallCUDA(cudaMemcpyAsync(ptr + N, host_ptr + N, N * sizeof(K), cudaMemcpyHostToDevice, *stream));
    PetscCallCXX(thrust::copy_n(thrust::cuda::par.on(*stream), dptr[1], N, dx));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCallCUDA(cudaFree(ptr));
    PetscCall(PetscFree(host_ptr));
    PetscCall(PetscLogGpuToCpu(2 * N * sizeof(K)));
    PetscCall(PetscLogCpuToGpu(N * sizeof(K)));
  } else {
    PetscScalar *host_ptr;

    PetscCall(PetscMalloc1(2 * N, &host_ptr));
    PetscCallCUDA(cudaMemcpyAsync(host_ptr, b, N * sizeof(PetscScalar), cudaMemcpyDeviceToHost, *stream));
    PetscCallCUDA(cudaMemcpyAsync(host_ptr + N, x, N * sizeof(PetscScalar), cudaMemcpyDeviceToHost, *stream));
    PetscCall(PetscDeviceContextSynchronize(dctx));
#if PetscDefined(USE_COMPLEX)
    std::complex<PetscReal> *const hb = reinterpret_cast<std::complex<PetscReal> *>(host_ptr);
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, hb, hb + N, n, comm));
#else
    PetscCall(HPDDM::IterativeMethod::solve(*data->op, host_ptr, host_ptr + N, n, comm));
#endif
    PetscCallCUDA(cudaMemcpyAsync(x, host_ptr + N, N * sizeof(PetscScalar), cudaMemcpyHostToDevice, *stream));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(PetscFree(host_ptr));
    PetscCall(PetscLogGpuToCpu(2 * N * sizeof(PetscScalar)));
    PetscCall(PetscLogCpuToGpu(N * sizeof(PetscScalar)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
