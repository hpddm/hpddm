#include <petsc/private/dmimpl.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/petschpddm.h> /*I "petscpc.h" I*/
#include <petsc/private/pcimpl.h> /* this must be included after petschpddm.h so that _PCIMPL_H is not defined            */
                                  /* otherwise, it is assumed that one is compiling libhpddm_petsc => circular dependency */
#include <petsc/private/pcasmimpl.h>
#if defined(PETSC_HAVE_FORTRAN)
#include <petsc/private/fortranimpl.h>
#endif

static PetscErrorCode (*loadedSym)(HPDDM::Schwarz<PetscScalar>* const, IS, Mat, Mat, Mat, std::vector<Vec>, PC_HPDDM_Level** const) = NULL;

static PetscBool PCHPDDMPackageInitialized = PETSC_FALSE;
static PetscBool citePC = PETSC_FALSE;
static const char hpddmCitationPC[] = "@inproceedings{jolivet2013scalable,\n\tTitle = {{Scalable Domain Decomposition Preconditioners For Heterogeneous Elliptic Problems}},\n\tAuthor = {Jolivet, Pierre and Hecht, Fr\'ed\'eric and Nataf, Fr\'ed\'eric and Prud'homme, Christophe},\n\tOrganization = {ACM},\n\tYear = {2013},\n\tSeries = {SC13},\n\tBooktitle = {Proceedings of the 2013 International Conference for High Performance Computing, Networking, Storage and Analysis}\n}\n";

PetscLogEvent PC_HPDDM_Strc;
PetscLogEvent PC_HPDDM_PtAP;
PetscLogEvent PC_HPDDM_PtBP;
PetscLogEvent PC_HPDDM_Next;
PetscLogEvent PC_HPDDM_SetUp[PETSC_HPDDM_MAXLEVELS];
PetscLogEvent PC_HPDDM_Solve[PETSC_HPDDM_MAXLEVELS];

static const char *PCHPDDMCoarseCorrectionTypes[] = { "deflated", "additive", "balanced" };

static PetscErrorCode PCReset_HPDDM(PC pc)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (data->levels) {
    for (i = 0; i < PETSC_HPDDM_MAXLEVELS && data->levels[i]; ++i) {
      ierr = KSPDestroy(&data->levels[i]->ksp);CHKERRQ(ierr);
      ierr = PCDestroy(&data->levels[i]->pc);CHKERRQ(ierr);
      ierr = PetscFree(data->levels[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(data->levels);CHKERRQ(ierr);
  }

  ierr = ISDestroy(&data->is);CHKERRQ(ierr);
  ierr = MatDestroy(&data->aux);CHKERRQ(ierr);
  ierr = MatDestroy(&data->B);CHKERRQ(ierr);
  ierr = VecDestroy(&data->normal);CHKERRQ(ierr);
  data->correction = PC_HPDDM_COARSE_CORRECTION_DEFLATED;
  data->Neumann    = PETSC_FALSE;
  data->setup      = NULL;
  data->setup_ctx  = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_HPDDM(PC pc)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_HPDDM(pc);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)pc, 0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetAuxiliaryMat_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMHasNeumannMat_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetRHSMat_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetCoarseCorrectionType_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetCoarseCorrectionType_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetSTShareSubKSP_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetAuxiliaryMat_HPDDM(PC pc, IS is, Mat A, PetscErrorCode (*setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void* setup_ctx)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is) {
    ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
    if (data->is) { /* new overlap definition resets the PC */
      ierr = PCReset_HPDDM(pc);CHKERRQ(ierr);
      pc->setfromoptionscalled = 0;
    }
    ierr = ISDestroy(&data->is);CHKERRQ(ierr);
    data->is = is;
  }
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatDestroy(&data->aux);CHKERRQ(ierr);
    data->aux = A;
  }
  if (setup) {
    data->setup = setup;
    data->setup_ctx = setup_ctx;
  }
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMSetAuxiliaryMat - Sets the auxiliary matrix used by PCHPDDM for the concurrent GenEO eigenproblems at the finest level. As an example, in a finite element context with nonoverlapping subdomains plus (overlapping) ghost elements, this could be the unassembled (Neumann) local overlapping operator. As opposed to the assembled (Dirichlet) local overlapping operator obtained by summing neighborhood contributions at the interface of ghost elements.

   Input Parameters:
+     pc - preconditioner context
.     is - index set of the local auxiliary, e.g., Neumann, matrix
.     A - auxiliary sequential matrix
.     setup - function for generating the auxiliary matrix
-     setup_ctx - context for setup

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCHPDDMSetRHSMat(), MATIS
@*/
PetscErrorCode PCHPDDMSetAuxiliaryMat(PC pc, IS is, Mat A, PetscErrorCode (*setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void* setup_ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (is) PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  if (A) PetscValidHeaderSpecific(A, MAT_CLASSID, 3);
#if defined(PETSC_HAVE_FORTRAN)
  if (reinterpret_cast<void*>(setup) == reinterpret_cast<void*>(PETSC_NULL_FUNCTION_Fortran)) setup = NULL;
  if (setup_ctx == PETSC_NULL_INTEGER_Fortran) setup_ctx = NULL;
#endif
  ierr = PetscTryMethod(pc, "PCHPDDMSetAuxiliaryMat_C", (PC, IS, Mat, PetscErrorCode (*)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void*), (pc, is, A, setup, setup_ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMHasNeumannMat_HPDDM(PC pc, PetscBool has)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  data->Neumann = has;
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMHasNeumannMat - Informs PCHPDDM that the Mat passed to PCHPDDMSetAuxiliaryMat() is the local Neumann matrix. This may be used to bypass a call to MatCreateSubMatrices() and to MatConvert() for MATMPISBAIJ matrices. If a DMCreateNeumannOverlap() implementation is available in the DM attached to the Pmat, or the Amat, or the PC, the flag is internally set to PETSC_TRUE. Its default value is otherwise PETSC_FALSE.

   Input Parameters:
+     pc - preconditioner context
-     has - Boolean value

   Level: intermediate

.seealso:  PCHPDDM, PCHPDDMSetAuxiliaryMat()
@*/
PetscErrorCode PCHPDDMHasNeumannMat(PC pc, PetscBool has)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscTryMethod(pc, "PCHPDDMHasNeumannMat_C", (PC, PetscBool), (pc, has));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetRHSMat_HPDDM(PC pc, Mat B)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
  ierr = MatDestroy(&data->B);CHKERRQ(ierr);
  data->B = B;
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMSetRHSMat - Sets the right-hand side matrix used by PCHPDDM for the concurrent GenEO eigenproblems at the finest level. Must be used in conjuction with PCHPDDMSetAuxiliaryMat(N), so that Nv = lambda Bv is solved using EPSSetOperators(N, B). It is assumed that N and B are provided using the same numbering. This provides a means to try more advanced methods such as GenEO-II or H-GenEO.

   Input Parameters:
+     pc - preconditioner context
-     B - right-hand side sequential matrix

   Level: advanced

.seealso:  PCHPDDMSetAuxiliaryMat(), PCHPDDM
@*/
PetscErrorCode PCHPDDMSetRHSMat(PC pc, Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (B) {
    PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
    ierr = PetscTryMethod(pc, "PCHPDDMSetRHSMat_C", (PC, Mat), (pc, B));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_HPDDM(PetscOptionItems *PetscOptionsObject, PC pc)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PC_HPDDM_Level **levels = data->levels;
  char           prefix[256];
  int            i = 1;
  PetscMPIInt    size, previous;
  PetscInt       n;
  PetscBool      flg = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!data->levels) {
    ierr = PetscCalloc1(PETSC_HPDDM_MAXLEVELS, &levels);CHKERRQ(ierr);
    data->levels = levels;
  }
  ierr = PetscOptionsHead(PetscOptionsObject, "PCHPDDM options");CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size);CHKERRMPI(ierr);
  previous = size;
  while (i < PETSC_HPDDM_MAXLEVELS) {
    PetscInt p = 1;

    if (!data->levels[i - 1]) {
      ierr = PetscNewLog(pc, data->levels + i - 1);CHKERRQ(ierr);
    }
    data->levels[i - 1]->parent = data;
    /* if the previous level has a single process, it is not possible to coarsen further */
    if (previous == 1 || !flg) break;
    data->levels[i - 1]->nu = 0;
    data->levels[i - 1]->threshold = -1.0;
    ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_nev", i);CHKERRQ(ierr);
    ierr = PetscOptionsInt(prefix, "Local number of deflation vectors computed by SLEPc", "none", data->levels[i - 1]->nu, &data->levels[i - 1]->nu, NULL);CHKERRQ(ierr);
    ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_threshold", i);CHKERRQ(ierr);
    ierr = PetscOptionsReal(prefix, "Local threshold for selecting deflation vectors returned by SLEPc", "none", data->levels[i - 1]->threshold, &data->levels[i - 1]->threshold, NULL);CHKERRQ(ierr);
    if (i == 1) {
      ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_1_st_share_sub_ksp");CHKERRQ(ierr);
      ierr = PetscOptionsBool(prefix, "Shared KSP between SLEPc ST and the fine-level subdomain solver", "none", PETSC_FALSE, &data->share, NULL);CHKERRQ(ierr);
    }
    /* if there is no prescribed coarsening, just break out of the loop */
    if (data->levels[i - 1]->threshold <= 0.0 && data->levels[i - 1]->nu <= 0) break;
    else {
      ++i;
      ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_nev", i);CHKERRQ(ierr);
      ierr = PetscOptionsHasName(PetscOptionsObject->options, PetscOptionsObject->prefix, prefix, &flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_threshold", i);CHKERRQ(ierr);
        ierr = PetscOptionsHasName(PetscOptionsObject->options, PetscOptionsObject->prefix, prefix, &flg);CHKERRQ(ierr);
      }
      if (flg) {
        /* if there are coarsening options for the next level, then register it  */
        /* otherwise, don't to avoid having both options levels_N_p and coarse_p */
        ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_p", i);CHKERRQ(ierr);
        ierr = PetscOptionsRangeInt(prefix, "Number of processes used to assemble the coarse operator at this level", "none", p, &p, &flg, 1, PetscMax(1, previous / 2));CHKERRQ(ierr);
        previous = p;
      }
    }
  }
  data->N = i;
  n = 1;
  if (i > 1) {
    ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_coarse_p");CHKERRQ(ierr);
    ierr = PetscOptionsRangeInt(prefix, "Number of processes used to assemble the coarsest operator", "none", n, &n, NULL, 1, PetscMax(1, previous / 2));CHKERRQ(ierr);
    ierr = PetscOptionsEList("-pc_hpddm_coarse_correction", "Type of coarse correction applied each iteration", "PCHPDDMSetCoarseCorrectionType", PCHPDDMCoarseCorrectionTypes, 3, PCHPDDMCoarseCorrectionTypes[PC_HPDDM_COARSE_CORRECTION_DEFLATED], &n, &flg);CHKERRQ(ierr);
    if (flg) data->correction = PCHPDDMCoarseCorrectionType(n);
    ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_has_neumann");CHKERRQ(ierr);
    ierr = PetscOptionsBool(prefix, "Is the auxiliary Mat the local Neumann matrix?", "PCHPDDMHasNeumannMat", data->Neumann, &data->Neumann, NULL);CHKERRQ(ierr);
    data->log_separate = PETSC_FALSE;
    if (PetscDefined(USE_LOG)) {
      ierr = PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_log_separate");CHKERRQ(ierr);
      ierr = PetscOptionsBool(prefix, "Log events level by level instead of inside PCSetUp()/KSPSolve()", NULL, data->log_separate, &data->log_separate, NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  while (i < PETSC_HPDDM_MAXLEVELS && data->levels[i]) {
    ierr = PetscFree(data->levels[i++]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_HPDDM(PC pc, Vec x, Vec y)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(hpddmCitationPC, &citePC);CHKERRQ(ierr);
  if (data->levels[0]->ksp) {
    if (data->log_separate) { /* coarser-level events are directly triggered in HPDDM */
      ierr = PetscLogEventBegin(PC_HPDDM_Solve[0], data->levels[0]->ksp, 0, 0, 0);CHKERRQ(ierr);
    }
    ierr = KSPSolve(data->levels[0]->ksp, x, y);CHKERRQ(ierr);
    if (data->log_separate) {
      ierr = PetscLogEventEnd(PC_HPDDM_Solve[0], data->levels[0]->ksp, 0, 0, 0);CHKERRQ(ierr);
    }
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "No KSP attached to PCHPDDM"); // LCOV_EXCL_LINE
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_HPDDM(PC pc, Mat X, Mat Y)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(hpddmCitationPC, &citePC);CHKERRQ(ierr);
  if (data->levels[0]->ksp) {
    ierr = KSPMatSolve(data->levels[0]->ksp, X, Y);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "No KSP attached to PCHPDDM"); // LCOV_EXCL_LINE
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMGetComplexities - Computes the grid and operator complexities.

   Input Parameter:
.     pc - preconditioner context

   Output Parameters:
+     gc - grid complexity = sum_i(m_i) / m_1
-     oc - operator complexity = sum_i(nnz_i) / nnz_1

   Notes:
     PCGAMG does not follow the usual convention and names the grid complexity what is usually referred to as the operator complexity. PCHPDDM follows what is found in the literature, and in particular, what you get with PCHYPRE and -pc_hypre_boomeramg_print_statistics.

   Level: advanced

.seealso:  PCMGGetGridComplexity(), PCHPDDM
@*/
static PetscErrorCode PCHPDDMGetComplexities(PC pc, PetscReal *gc, PetscReal *oc)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  MatInfo        info;
  PetscInt       n, m;
  PetscLogDouble accumulate[2] { }, nnz1 = 1.0, m1 = 1.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (n = 0, *gc = 0, *oc = 0; n < data->N; ++n) {
    if (data->levels[n]->ksp) {
      Mat P;
      ierr = KSPGetOperators(data->levels[n]->ksp, NULL, &P);CHKERRQ(ierr);
      ierr = MatGetSize(P, &m, NULL);CHKERRQ(ierr);
      accumulate[0] += m;
      ierr = MatGetInfo(P, MAT_GLOBAL_SUM, &info);CHKERRQ(ierr);
      accumulate[1] += info.nz_used;
      if (n == 0) {
        m1 = m;
        nnz1 = info.nz_used;
      }
    }
  }
  *gc = static_cast<PetscReal>(accumulate[0]/m1);
  *oc = static_cast<PetscReal>(accumulate[1]/nnz1);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_HPDDM(PC pc, PetscViewer viewer)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscViewer    subviewer;
  PetscSubcomm   subcomm;
  PetscReal      oc, gc;
  PetscInt       i, tabs;
  PetscMPIInt    size, color, rank;
  PetscBool      ascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &ascii);CHKERRQ(ierr);
  if (ascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "level%s: %D\n", data->N > 1 ? "s" : "", data->N);CHKERRQ(ierr);
    ierr = PCHPDDMGetComplexities(pc, &gc, &oc);CHKERRQ(ierr);
    if (data->N > 1) {
      ierr = PetscViewerASCIIPrintf(viewer, "Neumann matrix attached? %s\n", PetscBools[data->Neumann]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "coarse correction: %s\n", PCHPDDMCoarseCorrectionTypes[data->correction]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "on process #0, value%s (+ threshold%s if available) for selecting deflation vectors:", data->N > 2 ? "s" : "", data->N > 2 ? "s" : "");CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetTab(viewer, &tabs);CHKERRQ(ierr);
      ierr = PetscViewerASCIISetTab(viewer, 0);CHKERRQ(ierr);
      for (i = 1; i < data->N; ++i) {
        ierr = PetscViewerASCIIPrintf(viewer, " %D", data->levels[i - 1]->nu);CHKERRQ(ierr);
        if (data->levels[i - 1]->threshold > -0.1) {
          ierr = PetscViewerASCIIPrintf(viewer, " (%g)", (double)data->levels[i - 1]->threshold);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISetTab(viewer, tabs);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "grid and operator complexities: %g %g\n", (double)gc, (double)oc);CHKERRQ(ierr);
    if (data->levels[0]->ksp) {
      ierr = KSPView(data->levels[0]->ksp, viewer);CHKERRQ(ierr);
      if (data->levels[0]->pc) {
        ierr = PCView(data->levels[0]->pc, viewer);CHKERRQ(ierr);
      }
      for (i = 1; i < data->N; ++i) {
        if (data->levels[i]->ksp) color = 1;
        else color = 0;
        ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size);CHKERRMPI(ierr);
        ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank);CHKERRMPI(ierr);
        ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)pc), &subcomm);CHKERRQ(ierr);
        ierr = PetscSubcommSetNumber(subcomm, PetscMin(size, 2));CHKERRQ(ierr);
        ierr = PetscSubcommSetTypeGeneral(subcomm, color, rank);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = PetscViewerGetSubViewer(viewer, PetscSubcommChild(subcomm), &subviewer);CHKERRQ(ierr);
        if (color == 1) {
          ierr = KSPView(data->levels[i]->ksp, subviewer);CHKERRQ(ierr);
          if (data->levels[i]->pc) {
            ierr = PCView(data->levels[i]->pc, subviewer);CHKERRQ(ierr);
          }
          ierr = PetscViewerFlush(subviewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSubViewer(viewer, PetscSubcommChild(subcomm), &subviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        ierr = PetscSubcommDestroy(&subcomm);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPreSolve_HPDDM(PC pc, KSP ksp, Vec, Vec)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscBool      flg;
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp) {
    ierr = PetscObjectTypeCompare((PetscObject)ksp, KSPLSQR, &flg);CHKERRQ(ierr);
    if (flg && !data->normal) {
      ierr = KSPGetOperators(ksp, &A, NULL);CHKERRQ(ierr);
      ierr = MatCreateVecs(A, NULL, &data->normal);CHKERRQ(ierr); /* temporary Vec used in PCHPDDMShellApply() for coarse grid corrections */
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMShellSetUp(PC pc)
{
  PC_HPDDM_Level *ctx;
  Mat            A, P;
  Vec            x;
  const char     *pcpre;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&ctx);CHKERRQ(ierr);
  ierr = KSPGetOptionsPrefix(ctx->ksp, &pcpre);CHKERRQ(ierr);
  ierr = KSPGetOperators(ctx->ksp, &A, &P);CHKERRQ(ierr);
  /* smoother */
  ierr = PCSetOptionsPrefix(ctx->pc, pcpre);CHKERRQ(ierr);
  ierr = PCSetOperators(ctx->pc, A, P);CHKERRQ(ierr);
  if (!ctx->v[0]) {
    ierr = VecDuplicateVecs(ctx->D, 1, &ctx->v[0]);CHKERRQ(ierr);
    if (!std::is_same<PetscScalar, PetscReal>::value) {
      ierr = VecDestroy(&ctx->D);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(A, &x, NULL);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(x, 2, &ctx->v[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template<class Type, typename std::enable_if<std::is_same<Type, Vec>::value>::type* = nullptr>
PETSC_STATIC_INLINE PetscErrorCode PCHPDDMDeflate_Private(PC pc, Type x, Type y)
{
  PC_HPDDM_Level *ctx;
  PetscScalar    *out;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&ctx);CHKERRQ(ierr);
  /* going from PETSc to HPDDM numbering */
  ierr = VecScatterBegin(ctx->scatter, x, ctx->v[0][0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatter, x, ctx->v[0][0], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(ctx->v[0][0], &out);CHKERRQ(ierr);
  ctx->P->deflation<false>(NULL, out, 1); /* y = Q x */
  ierr = VecRestoreArrayWrite(ctx->v[0][0], &out);CHKERRQ(ierr);
  /* going from HPDDM to PETSc numbering */
  ierr = VecScatterBegin(ctx->scatter, ctx->v[0][0], y, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatter, ctx->v[0][0], y, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template<class Type, typename std::enable_if<std::is_same<Type, Mat>::value>::type* = nullptr>
PETSC_STATIC_INLINE PetscErrorCode PCHPDDMDeflate_Private(PC pc, Type X, Type Y)
{
  PC_HPDDM_Level *ctx;
  Vec            vX, vY, vC;
  PetscScalar    *out;
  PetscInt       i, m, N, prev = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&ctx);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ctx->v[0][0], &m);CHKERRQ(ierr);
  ierr = MatGetSize(X, NULL, &N);CHKERRQ(ierr);
  if (ctx->V) {
    ierr = MatGetSize(ctx->V, NULL, &prev);CHKERRQ(ierr);
  }
  if (N != prev) {
    ierr = MatDestroy(&ctx->V);CHKERRQ(ierr);
    ierr = MatCreateDense(PetscObjectComm((PetscObject)pc), m, PETSC_DECIDE, PETSC_DECIDE, N, NULL, &ctx->V);CHKERRQ(ierr);
  }
  /* going from PETSc to HPDDM numbering */
  for (i = 0; i < N; ++i) {
    ierr = MatDenseGetColumnVecRead(X, i, &vX);CHKERRQ(ierr);
    ierr = MatDenseGetColumnVecWrite(ctx->V, i, &vC);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->scatter, vX, vC, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scatter, vX, vC, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecWrite(ctx->V, i, &vC);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecRead(X, i, &vX);CHKERRQ(ierr);
  }
  ierr = MatDenseGetArrayWrite(ctx->V, &out);CHKERRQ(ierr);
  if (N != prev) {
    ctx->P->start(N);
    prev = N;
  }
  ctx->P->deflation<false>(NULL, out, N); /* Y = Q X */
  ierr = MatDenseRestoreArrayWrite(ctx->V, &out);CHKERRQ(ierr);
  /* going from HPDDM to PETSc numbering */
  for (i = 0; i < N; ++i) {
    ierr = MatDenseGetColumnVecRead(ctx->V, i, &vC);CHKERRQ(ierr);
    ierr = MatDenseGetColumnVecWrite(Y, i, &vY);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->scatter, vC, vY, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->scatter, vC, vY, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecWrite(Y, i, &vY);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecRead(ctx->V, i, &vC);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMShellApply - Applies a (2) deflated, (1) additive, or (3) balanced coarse correction. In what follows, E = Z Pmat Z^T and Q = Z^T E^-1 Z.

.vb
   (1) y =                Pmat^-1              x + Q x,
   (2) y =                Pmat^-1 (I - Amat Q) x + Q x (default),
   (3) y = (I - Q Amat^T) Pmat^-1 (I - Amat Q) x + Q x.
.ve

   Input Parameters:
+     pc - preconditioner context
-     x - input vector

   Output Parameter:
.     y - output vector

   Application Interface Routine: PCApply()

   Notes:
     The options of Pmat^1 = pc(Pmat) are prefixed by -pc_hpddm_levels_1_pc_. Z is a tall-and-skiny matrix assembled by HPDDM. The number of processes on which (Z Pmat Z^T) is aggregated is set via -pc_hpddm_coarse_p.
     The options of (Z Pmat Z^T)^-1 = ksp(Z Pmat Z^T) are prefixed by -pc_hpddm_coarse_ (KSPPREONLY and PCCHOLESKY by default), unless a multilevel correction is turned on, in which case, this function is called recursively at each level except the coarsest one.
     (1) and (2) visit the "next" level (in terms of coarsening) once per application, while (3) visits it twice, so it is asymptotically twice costlier. (2) is not symmetric even if both Amat and Pmat are symmetric.

   Level: advanced

.seealso:  PCHPDDM, PCHPDDMCoarseCorrectionType
@*/
static PetscErrorCode PCHPDDMShellApply(PC pc, Vec x, Vec y)
{
  PC_HPDDM_Level *ctx;
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&ctx);CHKERRQ(ierr);
  if (ctx->P) {
    ierr = KSPGetOperators(ctx->ksp, &A, NULL);CHKERRQ(ierr);
    ierr = PCHPDDMDeflate_Private(pc, x, y);CHKERRQ(ierr);                    /* y = Q x                          */
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED || ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      if (!ctx->parent->normal || ctx != ctx->parent->levels[0]) {
        ierr = MatMult(A, y, ctx->v[1][0]);CHKERRQ(ierr);                     /* y = A Q x                        */
      } else { /* KSPLSQR and finest level */
        ierr = MatMult(A, y, ctx->parent->normal);CHKERRQ(ierr);              /* y = A Q x                        */
        ierr = MatMultTranspose(A, ctx->parent->normal, ctx->v[1][0]);CHKERRQ(ierr); /* y = A^T A Q x             */
      }
      ierr = VecWAXPY(ctx->v[1][1], -1.0, ctx->v[1][0], x);CHKERRQ(ierr);     /* y = (I - A Q) x                  */
      ierr = PCApply(ctx->pc, ctx->v[1][1], ctx->v[1][0]);CHKERRQ(ierr);      /* y = M^-1 (I - A Q) x             */
      if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
        if (!ctx->parent->normal || ctx != ctx->parent->levels[0]) {
          ierr = MatMultTranspose(A, ctx->v[1][0], ctx->v[1][1]);CHKERRQ(ierr); /* z = A^T M^-1 (I - A Q) x       */
        } else {
          ierr = MatMult(A, ctx->v[1][0], ctx->parent->normal);CHKERRQ(ierr);
          ierr = MatMultTranspose(A, ctx->parent->normal, ctx->v[1][1]);CHKERRQ(ierr); /* z = A^T A M^-1 (I - A^T A Q) x */
        }
        ierr = PCHPDDMDeflate_Private(pc, ctx->v[1][1], ctx->v[1][1]);CHKERRQ(ierr);
        ierr = VecAXPY(ctx->v[1][0], -1.0, ctx->v[1][1]);CHKERRQ(ierr);       /* y = (I - Q A^T) M^-1 (I - A Q) x */
      }
      ierr = VecAXPY(y, 1.0, ctx->v[1][0]);CHKERRQ(ierr);                     /* y = y + Q x                      */
    } else if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_ADDITIVE) {
      ierr = PCApply(ctx->pc, x, ctx->v[1][0]);CHKERRQ(ierr);
      ierr = VecAXPY(y, 1.0, ctx->v[1][0]);CHKERRQ(ierr);                     /* y = M^-1 x + Q x                 */
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with an unknown PCHPDDMCoarseCorrectionType %d", ctx->parent->correction); // LCOV_EXCL_LINE
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with no HPDDM object"); // LCOV_EXCL_LINE
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMShellMatApply - Variant of PCHPDDMShellApply() for blocks of vectors

   Input Parameters:
+     pc - preconditioner context
-     X - block of input vectors

   Output Parameter:
.     Y - block of output vectors

   Application Interface Routine: PCApply()

   Level: advanced

.seealso:  PCHPDDM, PCHPDDMShellMatApply(), PCHPDDMCoarseCorrectionType
@*/
static PetscErrorCode PCHPDDMShellMatApply(PC pc, Mat X, Mat Y)
{
  PC_HPDDM_Level *ctx;
  Mat            A, C, D;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&ctx);CHKERRQ(ierr);
  if (ctx->P) {
    ierr = KSPGetOperators(ctx->ksp, &A, NULL);CHKERRQ(ierr);
    ierr = PCHPDDMDeflate_Private(pc, X, Y);CHKERRQ(ierr);
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED || ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      ierr = MatMatMult(A, Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);CHKERRQ(ierr);
      ierr = MatDuplicate(C, MAT_COPY_VALUES, &D);CHKERRQ(ierr);
      ierr = MatAYPX(D, -1.0, X, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = PCMatApply(ctx->pc, D, C);CHKERRQ(ierr);
      if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
#if 0 // TODO FIXME: there is a bug in MatTransposeMatMult(): results are inconsitent with a column-by-column product
        ierr = MatTransposeMatMult(A, C, MAT_REUSE_MATRIX, PETSC_DEFAULT, &D);CHKERRQ(ierr);
#else
        PetscInt N;
        ierr = MatGetSize(D, NULL, &N);
        for (PetscInt i = 0; i < N; ++i) {
          Vec cD, cC;
          ierr = MatDenseGetColumnVecRead(C, i, &cC);CHKERRQ(ierr);
          ierr = MatDenseGetColumnVecWrite(D, i, &cD);CHKERRQ(ierr);
          ierr = MatMultTranspose(A, cC, cD);CHKERRQ(ierr);
          ierr = MatDenseRestoreColumnVecWrite(D, i, &cD);CHKERRQ(ierr);
          ierr = MatDenseRestoreColumnVecRead(C, i, &cC);CHKERRQ(ierr);
        }
#endif
      }
      ierr = MatAXPY(Y, 1.0, C, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&D);CHKERRQ(ierr);
    } else if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_ADDITIVE) {
      ierr = MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &C);CHKERRQ(ierr);
      ierr = PCMatApply(ctx->pc, X, C);CHKERRQ(ierr);
      ierr = MatAXPY(Y, 1.0, C, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with an unknown PCHPDDMCoarseCorrectionType %d", ctx->parent->correction); // LCOV_EXCL_LINE
    ierr = MatDestroy(&C);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with no HPDDM object"); // LCOV_EXCL_LINE
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMShellDestroy(PC pc)
{
  PC_HPDDM_Level *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc, (void**)&ctx);CHKERRQ(ierr);
  ierr = HPDDM::Schwarz<PetscScalar>::destroy(ctx, PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecDestroyVecs(1, &ctx->v[0]);CHKERRQ(ierr);
  ierr = VecDestroyVecs(2, &ctx->v[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->V);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx->scatter);CHKERRQ(ierr);
  ierr = PCDestroy(&ctx->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSolve_Private(const PC_HPDDM_Level *ctx, PetscScalar *rhs, const unsigned short& mu)
{
  Mat            B, X;
  PetscInt       n, N, j = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetOperators(ctx->ksp, &B, NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B, &n, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(B, &N, NULL);CHKERRQ(ierr);
  if (ctx->parent->log_separate) {
    j = std::distance(ctx->parent->levels, std::find(ctx->parent->levels, ctx->parent->levels + ctx->parent->N, ctx));
    ierr = PetscLogEventBegin(PC_HPDDM_Solve[j], ctx->ksp, 0, 0, 0);CHKERRQ(ierr);
  }
  if (mu == 1) {
    if (!ctx->ksp->vec_rhs) {
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)ctx->ksp), 1, n, N, NULL, &ctx->ksp->vec_rhs);CHKERRQ(ierr);
      ierr = VecCreateMPI(PetscObjectComm((PetscObject)ctx->ksp), n, N, &ctx->ksp->vec_sol);CHKERRQ(ierr);
    }
    ierr = VecPlaceArray(ctx->ksp->vec_rhs, rhs);CHKERRQ(ierr);
    ierr = KSPSolve(ctx->ksp, NULL, NULL);CHKERRQ(ierr);
    ierr = VecCopy(ctx->ksp->vec_sol, ctx->ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecResetArray(ctx->ksp->vec_rhs);CHKERRQ(ierr);
  } else {
    ierr = MatCreateDense(PetscObjectComm((PetscObject)ctx->ksp), n, PETSC_DECIDE, N, mu, rhs, &B);CHKERRQ(ierr);
    ierr = MatCreateDense(PetscObjectComm((PetscObject)ctx->ksp), n, PETSC_DECIDE, N, mu, NULL, &X);CHKERRQ(ierr);
    ierr = KSPMatSolve(ctx->ksp, B, X);CHKERRQ(ierr);
    ierr = MatCopy(X, B, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
  if (ctx->parent->log_separate) {
    ierr = PetscLogEventEnd(PC_HPDDM_Solve[j], ctx->ksp, 0, 0, 0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetUpNeumannOverlap_Private(PC pc)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (data->setup) {
    Mat       P;
    Vec       x, xt = NULL;
    PetscReal t = 0.0, s = 0.0;

    ierr = PCGetOperators(pc, NULL, &P);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)P, "__SNES_latest_X", (PetscObject*)&x);CHKERRQ(ierr);
    PetscStackPush("PCHPDDM Neumann callback");
    ierr = (*data->setup)(data->aux, t, x, xt, s, data->is, data->setup_ctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_HPDDM(PC pc)
{
  PC_HPDDM                 *data = (PC_HPDDM*)pc->data;
  PC                       inner;
  KSP                      *ksp;
  Mat                      *sub, A, P, N, C = NULL, uaux = NULL, weighted, subA[2];
  Vec                      xin, v;
  std::vector<Vec>         initial;
  IS                       is[1], loc, uis = data->is;
  ISLocalToGlobalMapping   l2g;
  char                     prefix[256];
  const char               *pcpre;
  const PetscScalar *const *ev;
  PetscInt                 n, requested = data->N, reused = 0;
  PetscBool                subdomains = PETSC_FALSE, flag = PETSC_FALSE, ismatis;
  DM                       dm;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (!data->levels || !data->levels[0]) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not a single level allocated"); // LCOV_EXCL_LINE
  ierr = PCGetOptionsPrefix(pc, &pcpre);CHKERRQ(ierr);
  ierr = PCGetOperators(pc, &A, &P);CHKERRQ(ierr);
  if (!data->levels[0]->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)pc), &data->levels[0]->ksp);CHKERRQ(ierr);
    ierr = PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_%s_", pcpre ? pcpre : "", data->N > 1 ? "levels_1" : "coarse");CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(data->levels[0]->ksp, prefix);CHKERRQ(ierr);
    ierr = KSPSetType(data->levels[0]->ksp, KSPPREONLY);CHKERRQ(ierr);
  } else if (data->levels[0]->ksp->pc && data->levels[0]->ksp->pc->setupcalled == 1 && data->levels[0]->ksp->pc->reusepreconditioner) {
    /* if the fine-level PCSHELL exists, its setup has succeeded, and one wants to reuse it, */
    /* then just propagate the appropriate flag to the coarser levels                        */
    for (n = 0; n < PETSC_HPDDM_MAXLEVELS && data->levels[n]; ++n) {
      /* the following KSP and PC may be NULL for some processes, hence the check            */
      if (data->levels[n]->ksp) {
        ierr = KSPSetReusePreconditioner(data->levels[n]->ksp, PETSC_TRUE);CHKERRQ(ierr);
      }
      if (data->levels[n]->pc) {
        ierr = PCSetReusePreconditioner(data->levels[n]->pc, PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    /* early bail out because there is nothing to do */
    PetscFunctionReturn(0);
  } else {
    /* reset coarser levels */
    for (n = 1; n < PETSC_HPDDM_MAXLEVELS && data->levels[n]; ++n) {
      if (data->levels[n]->ksp && data->levels[n]->ksp->pc && data->levels[n]->ksp->pc->setupcalled == 1 && data->levels[n]->ksp->pc->reusepreconditioner && n < data->N) {
        reused = data->N - n;
        break;
      }
      ierr = KSPDestroy(&data->levels[n]->ksp);CHKERRQ(ierr);
      ierr = PCDestroy(&data->levels[n]->pc);CHKERRQ(ierr);
    }
    /* check if some coarser levels are being reused */
    ierr = MPIU_Allreduce(MPI_IN_PLACE, &reused, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    const int *addr = data->levels[0]->P ? data->levels[0]->P->getAddrLocal() : &HPDDM::i__0;

    if (addr != &HPDDM::i__0 && reused != data->N - 1) {
      /* reuse previously computed eigenvectors */
      ev = data->levels[0]->P->getVectors();
      if (ev) {
        initial.reserve(*addr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, data->levels[0]->P->getDof(), ev[0], &xin);CHKERRQ(ierr);
        for (n = 0; n < *addr; ++n) {
          ierr = VecDuplicate(xin, &v);CHKERRQ(ierr);
          ierr = VecPlaceArray(xin, ev[n]);CHKERRQ(ierr);
          ierr = VecCopy(xin, v);CHKERRQ(ierr);
          initial.emplace_back(v);
          ierr = VecResetArray(xin);CHKERRQ(ierr);
        }
        ierr = VecDestroy(&xin);CHKERRQ(ierr);
      }
    }
  }
  data->N -= reused;
  ierr = KSPSetOperators(data->levels[0]->ksp, A, P);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)P, MATIS, &ismatis);CHKERRQ(ierr);
  if (!data->is && !ismatis) {
    PetscErrorCode (*create)(DM, IS*, Mat*, PetscErrorCode (**)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void**) = NULL;
    PetscErrorCode (*usetup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*) = NULL;
    void           *uctx = NULL;

    /* first see if we can get the data from the DM */
    ierr = MatGetDM(P, &dm);CHKERRQ(ierr);
    if (!dm) {
      ierr = MatGetDM(A, &dm);CHKERRQ(ierr);
    }
    if (!dm) {
      ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
    }
    if (dm) { /* this is the hook for DMPLEX and DMDA for which the auxiliary Mat is the local Neumann matrix */
      ierr = PetscObjectQueryFunction((PetscObject)dm, "DMCreateNeumannOverlap_C", &create);CHKERRQ(ierr);
      if (create) {
        ierr = (*create)(dm, &uis, &uaux, &usetup, &uctx);CHKERRQ(ierr);
        data->Neumann = PETSC_TRUE;
      }
    }
    if (!create) {
      if (!uis) {
        ierr = PetscObjectQuery((PetscObject)pc, "_PCHPDDM_Neumann_IS", (PetscObject*)&uis);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)uis);CHKERRQ(ierr);
      }
      if (!uaux) {
        ierr = PetscObjectQuery((PetscObject)pc, "_PCHPDDM_Neumann_Mat", (PetscObject*)&uaux);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)uaux);CHKERRQ(ierr);
      }
      /* look inside the Pmat instead of the PC, needed for MatSchurComplementComputeExplicitOperator() */
      if (!uis) {
        ierr = PetscObjectQuery((PetscObject)P, "_PCHPDDM_Neumann_IS", (PetscObject*)&uis);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)uis);CHKERRQ(ierr);
      }
      if (!uaux) {
        ierr = PetscObjectQuery((PetscObject)P, "_PCHPDDM_Neumann_Mat", (PetscObject*)&uaux);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)uaux);CHKERRQ(ierr);
      }
    }
    ierr = PCHPDDMSetAuxiliaryMat(pc, uis, uaux, usetup, uctx);CHKERRQ(ierr);
    ierr = MatDestroy(&uaux);CHKERRQ(ierr);
    ierr = ISDestroy(&uis);CHKERRQ(ierr);
  }

  if (!ismatis) {
    ierr = PCHPDDMSetUpNeumannOverlap_Private(pc);CHKERRQ(ierr);
  }

  if (data->is || (ismatis && data->N > 1)) {
    if (ismatis) {
      std::initializer_list<std::string> list = { MATSEQBAIJ, MATSEQSBAIJ };
      ierr = MatISGetLocalMat(P, &N);CHKERRQ(ierr);
      std::initializer_list<std::string>::const_iterator it = std::find(list.begin(), list.end(), ((PetscObject)N)->type_name);
      ierr = MatISRestoreLocalMat(P, &N);CHKERRQ(ierr);
      switch (std::distance(list.begin(), it)) {
      case 0:
        ierr = MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &C);CHKERRQ(ierr);
        break;
      case 1:
        /* MatCreateSubMatrices() does not work with MATSBAIJ and unsorted ISes, so convert to MPIBAIJ */
        ierr = MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &C);CHKERRQ(ierr);
        ierr = MatSetOption(C, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
        break;
      default:
        ierr = MatConvert(P, MATMPIAIJ, MAT_INITIAL_MATRIX, &C);CHKERRQ(ierr);
      }
      ierr = MatGetLocalToGlobalMapping(P, &l2g, NULL);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)P);CHKERRQ(ierr);
      ierr = KSPSetOperators(data->levels[0]->ksp, A, C);CHKERRQ(ierr);
      std::swap(C, P);
      ierr = ISLocalToGlobalMappingGetSize(l2g, &n);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &loc);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApplyIS(l2g, loc, &is[0]);CHKERRQ(ierr);
      ierr = ISDestroy(&loc);CHKERRQ(ierr);
      /* the auxiliary Mat is _not_ the local Neumann matrix                                */
      /* it is the local Neumann matrix augmented (with zeros) through MatIncreaseOverlap() */
      data->Neumann = PETSC_FALSE;
    } else {
      is[0] = data->is;
      ierr = PetscOptionsGetBool(NULL, pcpre, "-pc_hpddm_define_subdomains", &subdomains, NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetBool(NULL, pcpre, "-pc_hpddm_has_neumann", &data->Neumann, NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(PetscObjectComm((PetscObject)data->is), P->rmap->n, P->rmap->rstart, 1, &loc);CHKERRQ(ierr);
    }
    if (data->N > 1 && (data->aux || ismatis)) {
      if (!loadedSym) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "HPDDM library not loaded, cannot use more than one level"); // LCOV_EXCL_LINE
      ierr = MatSetOption(P, MAT_SUBMAT_SINGLEIS, PETSC_TRUE);CHKERRQ(ierr);
      if (ismatis) {
        /* needed by HPDDM (currently) so that the partition of unity is 0 on subdomain interfaces */
        ierr = MatIncreaseOverlap(P, 1, is, 1);CHKERRQ(ierr);
        ierr = ISDestroy(&data->is);CHKERRQ(ierr);
        data->is = is[0];
      } else {
        if (PetscDefined(USE_DEBUG)) {
          PetscBool equal;
          IS        intersect;

          ierr = ISIntersect(data->is, loc, &intersect);CHKERRQ(ierr);
          ierr = ISEqualUnsorted(loc, intersect, &equal);CHKERRQ(ierr);
          ierr = ISDestroy(&intersect);CHKERRQ(ierr);
          if (!equal) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "IS of the auxiliary Mat does not include all local rows of A"); // LCOV_EXCL_LINE
        }
        if (!data->Neumann) {
          ierr = PetscObjectTypeCompare((PetscObject)P, MATMPISBAIJ, &flag);CHKERRQ(ierr);
          if (flag) {
            /* maybe better to ISSort(is[0]), MatCreateSubMatrices(), and then MatPermute() */
            /* but there is no MatPermute_SeqSBAIJ(), so as before, just use MATMPIBAIJ     */
            ierr = MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &uaux);CHKERRQ(ierr);
            flag = PETSC_FALSE;
          }
        }
      }
      if (!uaux) {
        if (data->Neumann) sub = &data->aux;
        else {
          ierr = MatCreateSubMatrices(P, 1, is, is, MAT_INITIAL_MATRIX, &sub);CHKERRQ(ierr);
        }
      } else {
        ierr = MatCreateSubMatrices(uaux, 1, is, is, MAT_INITIAL_MATRIX, &sub);CHKERRQ(ierr);
        ierr = MatDestroy(&uaux);CHKERRQ(ierr);
        ierr = MatConvert(sub[0], MATSEQSBAIJ, MAT_INPLACE_MATRIX, sub);CHKERRQ(ierr);
      }
      /* Vec holding the partition of unity */
      if (!data->levels[0]->D) {
        ierr = ISGetLocalSize(data->is, &n);CHKERRQ(ierr);
        ierr = VecCreateMPI(PETSC_COMM_SELF, n, PETSC_DETERMINE, &data->levels[0]->D);CHKERRQ(ierr);
      }
      /* it is possible to share the PC only given specific conditions, otherwise there is not warranty that the matrices have the same nonzero pattern */
      if (!ismatis && sub == &data->aux && !data->B && subdomains && data->share) {
        IS             perm;
        PetscInt       size;
        const PetscInt *ptr;
        PetscInt       *concatenate;
        ierr = ISGetLocalSize(*is, &size);CHKERRQ(ierr);
        ierr = ISGetIndices(*is, &ptr);CHKERRQ(ierr);
        std::map<PetscInt, PetscInt> order;
        /* MatCreateSubMatrices(), called by PCASM, follows the global numbering of Pmat */
        for (n = 0; n < size; ++n)
            order.insert(std::make_pair(ptr[n], n));
        ierr = ISRestoreIndices(*is, &ptr);CHKERRQ(ierr);
        ierr = PetscMalloc1(size, &concatenate);CHKERRQ(ierr);
        for (const std::pair<const PetscInt, PetscInt>& i : order)
          *concatenate++ = i.second;
        concatenate -= size;
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)data->is), size, concatenate, PETSC_OWN_POINTER, &perm);CHKERRQ(ierr);
        ierr = ISSetPermutation(perm);CHKERRQ(ierr);
        /* permute user-provided Mat so that it matches with MatCreateSubMatrices() numbering */
        ierr = MatPermute(data->aux, perm, perm, &C);CHKERRQ(ierr);
        ierr = ISDestroy(&perm);CHKERRQ(ierr);
        ierr = PetscMalloc1(size, &concatenate);CHKERRQ(ierr);
        for (const std::pair<const PetscInt, PetscInt>& i : order)
          *concatenate++ = i.first;
        concatenate -= size;
        /* permute user-provided IS so that it matches with MatCreateSubMatrices() numbering */
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)data->is), size, concatenate, PETSC_OWN_POINTER, &uis);CHKERRQ(ierr);
        if (!data->levels[0]->pc) {
          ierr = PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_levels_1_", pcpre ? pcpre : "");CHKERRQ(ierr);
          ierr = PCCreate(PetscObjectComm((PetscObject)pc), &data->levels[0]->pc);CHKERRQ(ierr);
          ierr = PCSetOptionsPrefix(data->levels[0]->pc, prefix);CHKERRQ(ierr);
          ierr = PCSetOperators(data->levels[0]->pc, A, P);CHKERRQ(ierr);
        }
        ierr = PCSetType(data->levels[0]->pc, PCASM);CHKERRQ(ierr);
        if (!data->levels[0]->pc->setupcalled) {
          ierr = PCASMSetLocalSubdomains(data->levels[0]->pc, 1, is, &loc);CHKERRQ(ierr);
        }
        ierr = PCSetFromOptions(data->levels[0]->pc);CHKERRQ(ierr);
        ierr = PCSetUp(data->levels[0]->pc);CHKERRQ(ierr);
        size = -1;
        ierr = PetscTryMethod(data->levels[0]->pc, "PCASMGetSubKSP_C", (PC, PetscInt*, PetscInt*, KSP**), (data->levels[0]->pc, &size, NULL, &ksp));CHKERRQ(ierr);
        if (size != 1) {
          ierr = PCDestroy(&data->levels[0]->pc);CHKERRQ(ierr);
          ierr = MatDestroy(&C);CHKERRQ(ierr);
          ierr = ISDestroy(&uis);CHKERRQ(ierr);
          data->share = PETSC_FALSE;
          if (size == -1) {
            ierr = PetscInfo(pc, "Cannot share PC between ST and subdomain solver since PCASMGetSubKSP() not found in fine-level PC\n");CHKERRQ(ierr);
          } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of subdomain solver %D != 1", size); // LCOV_EXCL_LINE
        } else {
          Mat        D;
          const char *matpre;
          ierr = KSPGetOperators(ksp[0], subA, subA + 1);CHKERRQ(ierr);
          ierr = MatDuplicate(subA[1], MAT_SHARE_NONZERO_PATTERN, &D);CHKERRQ(ierr);
          ierr = MatGetOptionsPrefix(subA[1], &matpre);CHKERRQ(ierr);
          ierr = MatSetOptionsPrefix(D, matpre);CHKERRQ(ierr);
          ierr = MatAXPY(D, 1.0, C, SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
          ierr = MatPropagateSymmetryOptions(C, D);CHKERRQ(ierr);
          ierr = MatDestroy(&C);CHKERRQ(ierr);
          C = D;
          /* swap pointers so that variables stay consistent throughout PCSetUp() */
          std::swap(C, data->aux);
          std::swap(uis, data->is);
        }
      } else if (data->share) {
        ierr = PetscInfo(pc, "Cannot share PC between ST and subdomain solver\n");CHKERRQ(ierr);
        data->share = PETSC_FALSE;
      }
      if (!data->levels[0]->scatter) {
        ierr = MatCreateVecs(P, &xin, NULL);CHKERRQ(ierr);
        if (ismatis) {
          ierr = MatDestroy(&P);CHKERRQ(ierr);
        }
        ierr = VecScatterCreate(xin, data->is, data->levels[0]->D, NULL, &data->levels[0]->scatter);CHKERRQ(ierr);
        ierr = VecDestroy(&xin);CHKERRQ(ierr);
      }
      if (data->levels[0]->P) {
        /* if the pattern is the same and PCSetUp() has previously succeeded, reuse HPDDM buffers and connectivity */
        ierr = HPDDM::Schwarz<PetscScalar>::destroy(data->levels[0], pc->setupcalled < 1 || pc->flag == DIFFERENT_NONZERO_PATTERN ? PETSC_TRUE : PETSC_FALSE);CHKERRQ(ierr);
      }
      if (!data->levels[0]->P) data->levels[0]->P = new HPDDM::Schwarz<PetscScalar>();
      if (data->log_separate) {
        ierr = PetscLogEventBegin(PC_HPDDM_SetUp[0], data->levels[0]->ksp, 0, 0, 0);CHKERRQ(ierr);
      } else {
        ierr = PetscLogEventBegin(PC_HPDDM_Strc, data->levels[0]->ksp, 0, 0, 0);CHKERRQ(ierr);
      }
      /* HPDDM internal data structure */
      ierr = data->levels[0]->P->structure(loc, data->is, sub[0], ismatis ? C : data->aux, data->levels);CHKERRQ(ierr);
      if (!data->log_separate) {
        ierr = PetscLogEventEnd(PC_HPDDM_Strc, data->levels[0]->ksp, 0, 0, 0);CHKERRQ(ierr);
      }
      /* matrix pencil of the generalized eigenvalue problem on the overlap (GenEO) */
      if (!data->B) {
        ierr = MatDuplicate(sub[0], MAT_COPY_VALUES, &weighted);CHKERRQ(ierr);
        ierr = MatDiagonalScale(weighted, data->levels[0]->D, data->levels[0]->D);CHKERRQ(ierr);
        /* neither MatDuplicate() nor MatDiagonaleScale() handles the symmetry options, so propagate the options explicitly */
        /* only useful for -mat_type baij -pc_hpddm_levels_1_st_pc_type cholesky (no problem with MATAIJ or MATSBAIJ)       */
        ierr = MatPropagateSymmetryOptions(sub[0], weighted);CHKERRQ(ierr);
      } else weighted = data->B;
      /* SLEPc is used inside the loaded symbol */
      ierr = (*loadedSym)(data->levels[0]->P, data->is, ismatis ? C : data->aux, weighted, data->B, initial, data->levels);CHKERRQ(ierr);
      if (data->share) {
        Mat st[2];
        ((PC_ASM*)data->levels[0]->pc->data)->same_local_solves = PETSC_TRUE;
        ierr = KSPGetOperators(ksp[0], st, st + 1);CHKERRQ(ierr);
        ierr = MatCopy(subA[0], st[0], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        if (subA[1] != subA[0] || st[1] != st[0]) {
          ierr = MatCopy(subA[1], st[1], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        }
      }
      if (data->log_separate) {
        ierr = PetscLogEventEnd(PC_HPDDM_SetUp[0], data->levels[0]->ksp, 0, 0, 0);CHKERRQ(ierr);
      }
      if (ismatis) {
        ierr = MatISGetLocalMat(C, &N);CHKERRQ(ierr);
      } else N = data->aux;
      P = sub[0];
      /* going through the grid hierarchy */
      for (n = 1; n < data->N; ++n) {
        if (data->log_separate) {
          ierr = PetscLogEventBegin(PC_HPDDM_SetUp[n], data->levels[n]->ksp, 0, 0, 0);CHKERRQ(ierr);
        }
        /* method composed in the loaded symbol since there, SLEPc is used as well */
        ierr = PetscUseMethod(data->levels[0]->ksp, "PCHPDDMSetUp_Private_C", (Mat*, Mat*, PetscInt, PetscInt* const, PC_HPDDM_Level** const), (&P, &N, n, &data->N, data->levels));CHKERRQ(ierr);
        if (data->log_separate) {
          ierr = PetscLogEventEnd(PC_HPDDM_SetUp[n], data->levels[n]->ksp, 0, 0, 0);CHKERRQ(ierr);
        }
      }
      /* reset to NULL to avoid any faulty use */
      ierr = PetscObjectComposeFunction((PetscObject)data->levels[0]->ksp, "PCHPDDMSetUp_Private_C", NULL);CHKERRQ(ierr);
      if (ismatis) {
        /* matching PetscObjectReference() above */
        ierr = PetscObjectDereference((PetscObject)C);CHKERRQ(ierr);
      }
      for (n = 0; n < data->N - 1; ++n)
        if (data->levels[n]->P) {
          /* HPDDM internal work buffers */
          data->levels[n]->P->setBuffer();
          data->levels[n]->P->super::start();
        }
      if (!data->Neumann) {
        ierr = MatDestroySubMatrices(1, &sub);CHKERRQ(ierr);
      }
      if (ismatis) data->is = NULL;
      for (n = 0; n < data->N - 1 + (reused > 0); ++n) {
        if (data->levels[n]->P) {
          PC spc;

          /* force the PC to be PCSHELL to do the coarse grid corrections */
          ierr = KSPSetSkipPCSetFromOptions(data->levels[n]->ksp, PETSC_TRUE);CHKERRQ(ierr);
          ierr = KSPGetPC(data->levels[n]->ksp, &spc);CHKERRQ(ierr);
          ierr = PCSetType(spc, PCSHELL);CHKERRQ(ierr);
          ierr = PCShellSetContext(spc, data->levels[n]);CHKERRQ(ierr);
          ierr = PCShellSetSetUp(spc, PCHPDDMShellSetUp);CHKERRQ(ierr);
          ierr = PCShellSetApply(spc, PCHPDDMShellApply);CHKERRQ(ierr);
          ierr = PCShellSetMatApply(spc, PCHPDDMShellMatApply);CHKERRQ(ierr);
          ierr = PCShellSetDestroy(spc, PCHPDDMShellDestroy);CHKERRQ(ierr);
          if (!data->levels[n]->pc) {
            ierr = PCCreate(PetscObjectComm((PetscObject)data->levels[n]->ksp), &data->levels[n]->pc);CHKERRQ(ierr);
          }
          if (n < reused) {
            ierr = PCSetReusePreconditioner(spc, PETSC_TRUE);CHKERRQ(ierr);
            ierr = PCSetReusePreconditioner(data->levels[n]->pc, PETSC_TRUE);CHKERRQ(ierr);
          }
          ierr = PCSetUp(spc);CHKERRQ(ierr);
        }
      }
    } else flag = reused ? PETSC_FALSE : PETSC_TRUE;
    if (!ismatis && subdomains) {
      if (flag) {
        ierr = KSPGetPC(data->levels[0]->ksp, &inner);CHKERRQ(ierr);
      } else inner = data->levels[0]->pc;
      if (inner) {
        ierr = PCSetType(inner, PCASM);CHKERRQ(ierr);
        if (!inner->setupcalled) {
          ierr = PCASMSetLocalSubdomains(inner, 1, is, &loc);CHKERRQ(ierr);
        }
      }
    }
    ierr = ISDestroy(&loc);CHKERRQ(ierr);
  } else data->N = 1 + reused; /* enforce this value to 1 + reused if there is no way to build another level */
  if (requested != data->N + reused) {
    ierr = PetscInfo5(pc, "%D levels requested, only %D built + %D reused. Options for level(s) > %D, including -%spc_hpddm_coarse_ will not be taken into account\n", requested, data->N, reused, data->N, pcpre ? pcpre : "");CHKERRQ(ierr);
    ierr = PetscInfo2(pc, "It is best to tune parameters, e.g., a higher value for -%spc_hpddm_levels_%D_eps_threshold so that at least one local deflation vector will be selected\n", pcpre ? pcpre : "", data->N);CHKERRQ(ierr);
    /* cannot use PCHPDDMShellDestroy() because PCSHELL not set for unassembled levels */
    for (n = data->N - 1; n < requested - 1; ++n) {
      if (data->levels[n]->P) {
        ierr = HPDDM::Schwarz<PetscScalar>::destroy(data->levels[n], PETSC_TRUE);CHKERRQ(ierr);
        ierr = VecDestroyVecs(1, &data->levels[n]->v[0]);CHKERRQ(ierr);
        ierr = VecDestroyVecs(2, &data->levels[n]->v[1]);CHKERRQ(ierr);
        ierr = MatDestroy(&data->levels[n]->V);CHKERRQ(ierr);
        ierr = VecDestroy(&data->levels[n]->D);CHKERRQ(ierr);
        ierr = VecScatterDestroy(&data->levels[n]->scatter);CHKERRQ(ierr);
      }
    }
    if (reused) {
      for (n = reused; n < PETSC_HPDDM_MAXLEVELS && data->levels[n]; ++n) {
        ierr = KSPDestroy(&data->levels[n]->ksp);CHKERRQ(ierr);
        ierr = PCDestroy(&data->levels[n]->pc);CHKERRQ(ierr);
      }
    }
    if (PetscDefined(USE_DEBUG)) SETERRQ7(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "%D levels requested, only %D built + %D reused. Options for level(s) > %D, including -%spc_hpddm_coarse_ will not be taken into account. It is best to tune parameters, e.g., a higher value for -%spc_hpddm_levels_%D_eps_threshold so that at least one local deflation vector will be selected. If you don't want this to error out, compile --with-debugging=0", requested, data->N, reused, data->N, pcpre ? pcpre : "", pcpre ? pcpre : "", data->N); // LCOV_EXCL_LINE
  }
  /* these solvers are created after PCSetFromOptions() is called */
  if (pc->setfromoptionscalled) {
    for (n = 0; n < data->N; ++n) {
      if (data->levels[n]->ksp) {
        ierr = KSPSetFromOptions(data->levels[n]->ksp);CHKERRQ(ierr);
      }
      if (data->levels[n]->pc) {
        ierr = PCSetFromOptions(data->levels[n]->pc);CHKERRQ(ierr);
      }
    }
    pc->setfromoptionscalled = 0;
  }
  data->N += reused;
  if (data->share) {
    /* swap back pointers so that variables follow the user-provided numbering */
    std::swap(C, data->aux);
    std::swap(uis, data->is);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = ISDestroy(&uis);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMSetCoarseCorrectionType - Sets the coarse correction type.

   Input Parameters:
+     pc - preconditioner context
-     type - PC_HPDDM_COARSE_CORRECTION_DEFLATED, PC_HPDDM_COARSE_CORRECTION_ADDITIVE, or PC_HPDDM_COARSE_CORRECTION_BALANCED

   Options Database Key:
.   -pc_hpddm_coarse_correction <deflated, additive, balanced> - type of coarse correction to apply

   Level: intermediate

.seealso:  PCHPDDMGetCoarseCorrectionType(), PCHPDDM, PCHPDDMCoarseCorrectionType
@*/
PetscErrorCode PCHPDDMSetCoarseCorrectionType(PC pc, PCHPDDMCoarseCorrectionType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscTryMethod(pc, "PCHPDDMSetCoarseCorrectionType_C", (PC, PCHPDDMCoarseCorrectionType), (pc, type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMGetCoarseCorrectionType - Gets the coarse correction type.

   Input Parameter:
.     pc - preconditioner context

   Output Parameter:
.     type - PC_HPDDM_COARSE_CORRECTION_DEFLATED, PC_HPDDM_COARSE_CORRECTION_ADDITIVE, or PC_HPDDM_COARSE_CORRECTION_BALANCED

   Level: intermediate

.seealso:  PCHPDDMSetCoarseCorrectionType(), PCHPDDM, PCHPDDMCoarseCorrectionType
@*/
PetscErrorCode PCHPDDMGetCoarseCorrectionType(PC pc, PCHPDDMCoarseCorrectionType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscUseMethod(pc, "PCHPDDMGetCoarseCorrectionType_C", (PC, PCHPDDMCoarseCorrectionType*), (pc, type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMSetCoarseCorrectionType_HPDDM(PC pc, PCHPDDMCoarseCorrectionType type)
{
  PC_HPDDM       *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  if (type < 0 || type > 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PCHPDDMCoarseCorrectionType %d", type); // LCOV_EXCL_LINE
  data->correction = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMGetCoarseCorrectionType_HPDDM(PC pc, PCHPDDMCoarseCorrectionType *type)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  *type = data->correction;
  PetscFunctionReturn(0);
}

/*@
     PCHPDDMGetSTShareSubKSP - Gets whether the KSP in SLEPc ST and the fine-level subdomain solver is shared.

   Input Parameter:
.     pc - preconditioner context

   Output Parameter:
.     share - whether the KSP is shared or not

   Notes:
     This is not the same as PCGetReusePreconditioner(). The return value is unlikely to be true, but when it is, a symbolic factorization can be skipped
     when using a subdomain PCType such as PCLU or PCCHOLESKY.

   Level: advanced

@*/
PetscErrorCode PCHPDDMGetSTShareSubKSP(PC pc, PetscBool *share)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscUseMethod(pc, "PCHPDDMGetSTShareSubKSP_C", (PC, PetscBool*), (pc, share));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHPDDMGetSTShareSubKSP_HPDDM(PC pc, PetscBool *share)
{
  PC_HPDDM *data = (PC_HPDDM*)pc->data;

  PetscFunctionBegin;
  *share = data->share;
  PetscFunctionReturn(0);
}

PetscErrorCode HPDDMLoadDL_Private(PetscBool *found)
{
  char           lib[PETSC_MAX_PATH_LEN], dlib[PETSC_MAX_PATH_LEN], dir[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(dir, "${PETSC_LIB_DIR}");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-hpddm_dir", dir, sizeof(dir), NULL);CHKERRQ(ierr);
  ierr = PetscSNPrintf(lib, sizeof(lib), "%s/libhpddm_petsc", dir);CHKERRQ(ierr);
  ierr = PetscDLLibraryRetrieve(PETSC_COMM_SELF, lib, dlib, 1024, found);CHKERRQ(ierr);
  if (*found) {
    ierr = PetscDLLibraryAppend(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, dlib);CHKERRQ(ierr);
#if defined(SLEPC_LIB_DIR) /* this variable is passed during SLEPc ./configure since    */
  } else {                 /* slepcconf.h is not yet built (and thus can't be included) */
    ierr = PetscStrcpy(dir, HPDDM_STR(SLEPC_LIB_DIR));CHKERRQ(ierr);
    ierr = PetscSNPrintf(lib, sizeof(lib), "%s/libhpddm_petsc", dir);CHKERRQ(ierr);
    ierr = PetscDLLibraryRetrieve(PETSC_COMM_SELF, lib, dlib, 1024, found);CHKERRQ(ierr);
    if (*found) {
      ierr = PetscDLLibraryAppend(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, dlib);CHKERRQ(ierr);
#endif
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s not found", lib); // LCOV_EXCL_LINE
#if defined(SLEPC_LIB_DIR)
  }
#endif
  PetscFunctionReturn(0);
}

/*MC
     PCHPDDM - Interface with the HPDDM library.

   This PC may be used to build multilevel spectral domain decomposition methods based on the GenEO framework [2011, 2019]. It may be viewed as an alternative to spectral AMGe or PCBDDC with adaptive selection of constraints. A chronological bibliography of relevant publications linked with PC available in HPDDM through PCHPDDM may be found below. The interface is explained in details in [2021].

   The matrix to be preconditioned (Pmat) may be unassembled (MATIS) or assembled (MATMPIAIJ, MATMPIBAIJ, or MATMPISBAIJ). For multilevel preconditioning, when using an assembled Pmat, one must provide an auxiliary local Mat (unassembled local operator for GenEO) using PCHPDDMSetAuxiliaryMat(). Calling this routine is not needed when using a MATIS Pmat (assembly done internally using MatConvert).

   Options Database Keys:
+   -pc_hpddm_define_subdomains <true, default=false> - on the finest level, calls PCASMSetLocalSubdomains() with the IS supplied in PCHPDDMSetAuxiliaryMat() (only relevant with an assembled Pmat)
.   -pc_hpddm_has_neumann <true, default=false> - on the finest level, informs the PC that the local Neumann matrix is supplied in PCHPDDMSetAuxiliaryMat()
-   -pc_hpddm_coarse_correction <type, default=deflated> - determines the PCHPDDMCoarseCorrectionType when calling PCApply

   Options for subdomain solvers, subdomain eigensolvers (for computing deflation vectors), and the coarse solver can be set with
.vb
      -pc_hpddm_levels_%d_pc_
      -pc_hpddm_levels_%d_ksp_
      -pc_hpddm_levels_%d_eps_
      -pc_hpddm_levels_%d_p
      -pc_hpddm_levels_%d_mat_type_
      -pc_hpddm_coarse_
      -pc_hpddm_coarse_p
      -pc_hpddm_coarse_mat_type_
.ve
   e.g., -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 10 -pc_hpddm_levels_2_p 4 -pc_hpddm_levels_2_sub_pc_type lu -pc_hpddm_levels_2_eps_nev 10 -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_mat_type baij will use 10 deflation vectors per subdomain on the fine "level 1", aggregate the fine subdomains into 4 "level 2" subdomains, then use 10 deflation vectors per subdomain on "level 2", and assemble the coarse matrix (of dimension 4 x 10 = 40) on two processes as a MATMPIBAIJ (default is MATMPISBAIJ).

   In order to activate a "level N+1" coarse correction, it is mandatory to call -pc_hpddm_levels_N_eps_nev <nu> or -pc_hpddm_levels_N_eps_threshold <val>. The default -pc_hpddm_coarse_p value is 1, meaning that the coarse operator is aggregated on a single process.

   This preconditioner requires that you build PETSc with SLEPc (--download-slepc=1). By default, the underlying concurrent eigenproblems are solved using SLEPc shift-and-invert spectral transformation. This is usually what gives the best performance for GenEO, cf. [2011, 2013]. As stated above, SLEPc options are available through -pc_hpddm_levels_%d_, e.g., -pc_hpddm_levels_1_eps_type arpack -pc_hpddm_levels_1_eps_threshold 0.1 -pc_hpddm_levels_1_st_type sinvert.

   References:
+   2011 - A robust two-level domain decomposition preconditioner for systems of PDEs. Spillane, Dolean, Hauret, Nataf, Pechstein, and Scheichl. Comptes Rendus Mathematique.
.   2013 - Scalable Domain Decomposition Preconditioners For Heterogeneous Elliptic Problems. Jolivet, Hecht, Nataf, and Prud'homme. SC13.
.   2015 - An Introduction to Domain Decomposition Methods: Algorithms, Theory, and Parallel Implementation. Dolean, Jolivet, and Nataf. SIAM.
.   2019 - A Multilevel Schwarz Preconditioner Based on a Hierarchy of Robust Coarse Spaces. Al Daas, Grigori, Jolivet, and Tournier.
-   2021 - KSPHPDDM and PCHPDDM: extending PETSc with advanced Krylov methods and robust multilevel overlapping Schwarz preconditioners. Jolivet, Roman, and Zampini. Computer & Mathematics with Applications.

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCHPDDMSetAuxiliaryMat(), MATIS, PCBDDC, PCDEFLATION, PCTELESCOPE
M*/
PETSC_EXTERN PetscErrorCode PCCreate_HPDDM(PC pc)
{
  PC_HPDDM       *data;
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!loadedSym) {
    ierr = HPDDMLoadDL_Private(&found);CHKERRQ(ierr);
    if (found) {
      ierr = PetscDLLibrarySym(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, NULL, "PCHPDDM_Internal", (void**)&loadedSym);CHKERRQ(ierr);
    }
  }
  if (!loadedSym) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCHPDDM_Internal symbol not found in loaded libhpddm_petsc"); // LCOV_EXCL_LINE
  ierr = PetscNewLog(pc, &data);CHKERRQ(ierr);
  pc->data                     = data;
  pc->ops->reset               = PCReset_HPDDM;
  pc->ops->destroy             = PCDestroy_HPDDM;
  pc->ops->setfromoptions      = PCSetFromOptions_HPDDM;
  pc->ops->setup               = PCSetUp_HPDDM;
  pc->ops->apply               = PCApply_HPDDM;
  pc->ops->matapply            = PCMatApply_HPDDM;
  pc->ops->view                = PCView_HPDDM;
  pc->ops->applytranspose      = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  pc->ops->presolve            = PCPreSolve_HPDDM;
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetAuxiliaryMat_C", PCHPDDMSetAuxiliaryMat_HPDDM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMHasNeumannMat_C", PCHPDDMHasNeumannMat_HPDDM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetRHSMat_C", PCHPDDMSetRHSMat_HPDDM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetCoarseCorrectionType_C", PCHPDDMSetCoarseCorrectionType_HPDDM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetCoarseCorrectionType_C", PCHPDDMGetCoarseCorrectionType_HPDDM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetSTShareSubKSP_C", PCHPDDMGetSTShareSubKSP_HPDDM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMInitializePackage - This function initializes everything in the PCHPDDM package. It is called from PCInitializePackage().

   Level: intermediate

.seealso:  PetscInitialize()
@*/
PetscErrorCode PCHPDDMInitializePackage(void)
{
  char           ename[32];
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PCHPDDMPackageInitialized) PetscFunctionReturn(0);
  PCHPDDMPackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PCHPDDMFinalizePackage);CHKERRQ(ierr);
  /* general events registered once during package initialization */
  /* some of these events are not triggered in libpetsc,          */
  /* but rather directly in libhpddm_petsc,                       */
  /* which is in charge of performing the following operations    */

  /* domain decomposition structure from Pmat sparsity pattern    */
  ierr = PetscLogEventRegister("PCHPDDMStrc", PC_CLASSID, &PC_HPDDM_Strc);CHKERRQ(ierr);
  /* Galerkin product, redistribution, and setup (not triggered in libpetsc)                */
  ierr = PetscLogEventRegister("PCHPDDMPtAP", PC_CLASSID, &PC_HPDDM_PtAP);CHKERRQ(ierr);
  /* Galerkin product with summation, redistribution, and setup (not triggered in libpetsc) */
  ierr = PetscLogEventRegister("PCHPDDMPtBP", PC_CLASSID, &PC_HPDDM_PtBP);CHKERRQ(ierr);
  /* next level construction using PtAP and PtBP (not triggered in libpetsc)                */
  ierr = PetscLogEventRegister("PCHPDDMNext", PC_CLASSID, &PC_HPDDM_Next);CHKERRQ(ierr);
  static_assert(PETSC_HPDDM_MAXLEVELS <= 9, "PETSC_HPDDM_MAXLEVELS value is too high");
  for (i = 1; i < PETSC_HPDDM_MAXLEVELS; ++i) {
    ierr = PetscSNPrintf(ename, sizeof(ename), "PCHPDDMSetUp L%1d", i);CHKERRQ(ierr);
    /* events during a PCSetUp() at level #i _except_ the assembly */
    /* of the Galerkin operator of the coarser level #(i + 1)      */
    ierr = PetscLogEventRegister(ename, PC_CLASSID, &PC_HPDDM_SetUp[i - 1]);CHKERRQ(ierr);
    ierr = PetscSNPrintf(ename, sizeof(ename), "PCHPDDMSolve L%1d", i);CHKERRQ(ierr);
    /* events during a PCApply() at level #i _except_              */
    /* the KSPSolve() of the coarser level #(i + 1)                */
    ierr = PetscLogEventRegister(ename, PC_CLASSID, &PC_HPDDM_Solve[i - 1]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
     PCHPDDMFinalizePackage - This function frees everything from the PCHPDDM package. It is called from PetscFinalize().

   Level: intermediate

.seealso:  PetscFinalize()
@*/
PetscErrorCode PCHPDDMFinalizePackage(void)
{
  PetscFunctionBegin;
  PCHPDDMPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}
