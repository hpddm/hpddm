## HPDDM — high-performance unified framework for domain decomposition methods [![Build Status](https://travis-ci.org/hpddm/hpddm.svg?branch=master)](https://travis-ci.org/hpddm/hpddm)

#### What is HPDDM?
HPDDM is an efficient implementation of various domain decomposition methods (DDM) such as one- and two-level Restricted Additive Schwarz methods, the Finite Element Tearing and Interconnecting (FETI) method, and the Balancing Domain Decomposition (BDD) method. These methods can be enhanced with deflation vectors computed automatically by the framework using:
* Generalized Eigenvalue problems on the Overlap (GenEO), an approach first introduced in a paper by [Spillane et al.](http://link.springer.com/article/10.1007%2Fs00211-013-0576-y#page-1), or
* local Dirichlet-to-Neumann operators, an approach first introduced in a paper by [Nataf et al.](http://epubs.siam.org/doi/abs/10.1137/100796376) and revisited by [Conen et al.](http://www.sciencedirect.com/science/article/pii/S0377042714001800)

This code has been proven to be efficient for solving various elliptic problems such as scalar diffusion equations, the system of linear elasticity, but also frequency domain problems like the Helmholtz equation. A comparison with modern multigrid methods can be found in the thesis of [Jolivet](http://jolivet.perso.enseeiht.fr/thesis.pdf). The preconditioners may be used with a variety of Krylov subspace methods (which all support right, left, and variable preconditioning):
* [GMRES](http://epubs.siam.org/doi/abs/10.1137/0907058) and [Block GMRES](http://www.sam.math.ethz.ch/~mhg/pub/delhipap.pdf),
* [CG](http://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf), [Block CG](http://www.sciencedirect.com/science/article/pii/0024379580902475), and [Breakdown-Free Block CG](http://link.springer.com/article/10.1007/s10543-016-0631-z),
* [GCRO-DR](http://epubs.siam.org/doi/abs/10.1137/040607277) and [Block GCRO-DR](http://dl.acm.org/citation.cfm?id=3014927).

#### How to use HPDDM?
HPDDM is a library written in C++11 with MPI and OpenMP for parallelism. While its interface relies on plain old data objects, it requires a modern C++ compiler: g++ 4.7.2 and above, clang++ 3.3 and above, icpc 15.0.0.090 and above&#185;, or pgc++ 15.1 and above&#185;. HPDDM has to be linked against BLAS and LAPACK (as found in [OpenBLAS](http://www.openblas.net/), in the [Accelerate framework](https://developer.apple.com/library/ios/documentation/Accelerate/Reference/AccelerateFWRef/_index.html) on OS X, in [IBM ESSL](http://www-03.ibm.com/systems/power/software/essl/), or in [Intel MKL](https://software.intel.com/en-us/intel-mkl)) as well as a direct solver like [MUMPS](http://mumps.enseeiht.fr/), [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html), [MKL PARDISO](https://software.intel.com/en-us/articles/intel-mkl-pardiso), or [PaStiX](http://pastix.gforge.inria.fr/). Additionally, an eigenvalue solver is recommended. There is an existing interface to [ARPACK](http://www.caam.rice.edu/software/ARPACK/). Other (eigen)solvers can be easily added using the existing interfaces.  
For building robust two-level methods, an interface with a discretization kernel like [FreeFem++](http://www.freefem.org/ff++/) or [Feel++](http://www.feelpp.org/) is also needed. It can then be used to provide, for example, elementary matrices, that the GenEO approach requires. As such preconditioners assembled by HPDDM are not algebraic, unless only looking at one-level methods. Note that for substructuring methods, this is more of a limitation of the mathematical approach than of HPDDM itself.  
If you need to generate the documentation, you first have to retrieve [NaturalDocs](http://www.naturaldocs.org/download/version1.52.html). Then, just type in the root of the repository `NaturalDocs --input include --output HTML doc --project doc`. The list of available options can be found in this [cheat sheet](https://github.com/hpddm/hpddm/raw/master/doc/cheatsheet.pdf).

&#185;The latest versions of ~~icpc and~~ (this has been fixed since version 16.0.2.181) pgc++ are not able to compile C++11 properly, if you want to use these compilers, please apply the following patch to the headers of HPDDM `sed -i\ '' 's/type\* = nullptr/type* = (void*)0/g; s/static constexpr const char/const char/g' include/*.hpp examples/*.cpp`.  

##### TL;DR
Create a `./Makefile.inc` by copying one from the folder `./Make.inc` and adapt it to your platform. Type `make test` to run C++, C, Python, and Fortran examples (just type `make test_language` with `language = [cpp|c|python|fortran]` if you want to try only one set of examples).

#### May HPDDM be embedded inside C, Python, or Fortran codes?
Yes, as long as you have a modern C++ compiler (cf. the previous paragraph). With Python, [NumPy](http://www.numpy.org/) and [mpi4py](https://bitbucket.org/mpi4py/) must also be available.

##### What if I don't want to deal with the library API myself?
You may access some functionalities of HPDDM through the following software:
* [FreeFem++](http://www.freefem.org/ff++/), all features of HPDDM,
* [Feel++](http://www.feelpp.org/), substructuring preconditioners,
* [PETSc](http://www.mcs.anl.gov/petsc/), advanced Krylov methods,
* [htool](https://github.com/PierreMarchand20/htool), overlapping Schwarz preconditioners for dense operators.

#### Who is behind HPDDM?
If you need help or have questions regarding HPDDM, feel free to contact [Pierre Jolivet](http://jolivet.perso.enseeiht.fr/) or [Frédéric Nataf](https://www.ljll.math.upmc.fr/nataf/).

#### How to cite HPDDM?
If you use this software, please cite this [paper](http://dl.acm.org/citation.cfm?doid=2503210.2503212) and this [book](http://www.siam.org/books/ot144/), thank you.

#### Acknowledgments
[Centre National de la Recherche Scientifique](http://www.cnrs.fr/index.php), France  
[Institut de Recherche en Informatique de Toulouse](http://www.irit.fr/?lang=en), France  
[Eidgenössische Technische Hochschule Zürich](https://www.ethz.ch/), Switzerland  
[Université Joseph Fourier](https://www.ujf-grenoble.fr/?language=en), Grenoble, France  
[Université Pierre et Marie Curie](http://www.upmc.fr/), Paris, France  
[Inria](http://www.inria.fr/en/) Paris, France  
[Agence Nationale de la Recherche](http://www.agence-nationale-recherche.fr/), France  
[Partnership for Advanced Computing in Europe](http://www.prace-ri.eu/)  
[Grand Equipement National de Calcul Intensif](http://www.genci.fr/en), France  
[Fondation Sciences Mathématiques de Paris](http://www.sciencesmaths-paris.fr/en/), France

###### Collaborators/contributors
[Lea Conen](https://www.linkedin.com/in/lea-conen-789111a5)  
[Victorita Dolean](http://dolean.blogspot.fr/)  
Ryadh Haferssas  
[Frédéric Hecht](https://www.ljll.math.upmc.fr/hecht/)  
[Pierre Marchand](https://www.ljll.math.upmc.fr/marchandp/)  
[Christophe Prud'homme](https://github.com/prudhomm)  
[Nicole Spillane](http://www.cmap.polytechnique.fr/~spillane/)  
Pierre-Henri Tournier
