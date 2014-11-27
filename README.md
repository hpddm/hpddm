## HPDDM — high-performance unified framework for domain decomposition methods [![Build Status](https://travis-ci.org/hpddm/hpddm.svg?branch=master)](https://travis-ci.org/hpddm/hpddm)

##### What is HPDDM ?
HPDDM is an efficient implementation of various domain decomposition methods (DDM) such as one- and two-level Restricted Additive Schwarz methods, the Finite Element Tearing and Interconnecting (FETI) method, and the Balancing Domain Decomposition (BDD) method. These methods can be enhanced with deflation vectors computed automatically by the framework using:
* Generalized Eigenvalue problems on the Overlap (GenEO), an approach first introduced in a paper by [Spillane et al.](http://link.springer.com/article/10.1007%2Fs00211-013-0576-y#page-1), or
* local Dirichlet-to-Neumann operators, an approach first introduced in a paper by [Nataf et al.](http://epubs.siam.org/doi/abs/10.1137/100796376) and recently revisited by [Conen et al.](http://www.sciencedirect.com/science/article/pii/S0377042714001800)

This code has been proven to be efficient for solving various elliptic problems such as scalar diffusion equations, the system of linear elasticity, but also frequency domain problems like the Helmholtz equation. A comparison with modern multigrid methods can be found in the thesis of [Jolivet](https://www.ljll.math.upmc.fr/~jolivet/thesis.pdf).

##### How to use HPDDM ?
HPDDM is a header-only library written in C++11 with MPI and OpenMP for parallelism. While its interface relies on plain old data objects, it requires a modern C++ compiler: g++ 4.7.3 and above, clang++ 3.3 and above, icpc 15.0.0.090 and above&#185;. HPDDM has to be linked against BLAS and LAPACK (as found in [OpenBLAS](http://www.openblas.net/), in the [Accelerate framework](https://developer.apple.com/library/ios/documentation/Accelerate/Reference/AccelerateFWRef/_index.html) on OS X, in [IBM ESSL](http://www-03.ibm.com/systems/power/software/essl/), or in [Intel MKL](https://software.intel.com/en-us/intel-mkl)) as well as a direct solver like [MUMPS](http://mumps.enseeiht.fr/), [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html), [MKL PARDISO](https://software.intel.com/en-us/articles/intel-mkl-pardiso), or [PaStiX](http://pastix.gforge.inria.fr/). At compilation, just define before including `HPDDM.hpp` _one_ of these preprocessor macros `MUMPSSUB`, `SUITESPARSESUB`, `MKL_PARDISOSUB`, or `PASTIXSUB` (resp. `DMUMPS`, `DSUITESPARSE`, `DMKL_PARDISO`, or `DPASTIX`) to use the corresponding solver inside each subdomain (resp. for the coarse operator). Additionally, an eigenvalue solver is recommended. There is an existing interface to [ARPACK](http://www.caam.rice.edu/software/ARPACK/). Other (eigen)solvers can be easily added using the existing interfaces.  
For building robust two-level methods, an interface with a discretization kernel like [FreeFem++](http://www.freefem.org/ff++/) or [Feel++](http://www.feelpp.org/) is also needed. It can then be used to provide, for example, elementary matrices, that the GenEO approach requires. As such HPDDM is not an algebraic solver, unless only looking at one-level methods. Note that for substructuring methods, this is more of a limitation of the mathematical approach than of HPDDM itself.  
If you need to generate the documentation, you first have to retrieve [NaturalDocs](http://www.naturaldocs.org/download/version1.52.html). Then, just type in the root of the repository `NaturalDocs --input src --output HTML doc --project doc`.

&#185;The latest versions of icpc are not able to compile C++11 properly, if you want to use these compilers, please apply the following patch to the sources of HPDDM `sed -i '' 's/ nullptr>/ (void*)0>/g' *.hpp`.

##### Who is behind HPDDM ?
If you need help or have questions regarding HPDDM, feel free to contact [Pierre Jolivet](https://www.ljll.math.upmc.fr/~jolivet/) or [Frédéric Nataf](http://www.ann.jussieu.fr/nataf/).

##### How to cite HPDDM ?
If you use this software, please cite the following paper: http://dl.acm.org/citation.cfm?doid=2503210.2503212.

##### Acknowledgments
[Université Joseph Fourier](https://www.ujf-grenoble.fr/?language=en), Grenoble, France.  
[Université Pierre et Marie Curie](http://www.upmc.fr/), Paris, France.  
[Inria](http://www.inria.fr/en/) Paris-Rocquencourt, France.  
[Agence Nationale pour la Recherche](http://www.agence-nationale-recherche.fr/), France.  
[Partnership for Advanced Computing in Europe](http://www.prace-ri.eu/).  
[Fondation Sciences Mathématiques de Paris](http://www.sciencesmaths-paris.fr/en/), France.

###### Collaborators/contributors
[Lea Conen](http://icsweb.inf.unisi.ch/cms/index.php/people/12-lea-conen.html)  
[Victorita Dolean](http://www-math.unice.fr/~dolean/Home.html)  
[Frédéric Hecht](http://www.ann.jussieu.fr/hecht/)  
[Christophe Prud'homme](http://www.prudhomm.org/)  
[Nicole Spillane](http://www.ann.jussieu.fr/~spillane/)
