!
!  This file is part of HPDDM.

!  Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
!       Date: 2016-05-01

!  Copyright (C) 2016-     Centre National de la Recherche Scientifique

!  HPDDM is free software: you can redistribute it and/or modify
!  it under the terms of the GNU Lesser General Public License as published
!  by the Free Software Foundation, either version 3 of the License, or
!  (at your option) any later version.

!  HPDDM is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!  GNU Lesser General Public License for more details.

!  You should have received a copy of the GNU Lesser General Public License
!  along with HPDDM.  If not, see <http://www.gnu.org/licenses/>.
!

      module my_operator
          implicit none
          contains
              subroutine my_mv(n, in, out, mu)
                  use, intrinsic :: iso_c_binding, only: c_int, c_double
                  use mpi
                  ! size of the local matrix
                  integer (c_int), intent (in) :: n
                  ! number of right-hand sides
                  integer (c_int), intent (in) :: mu
                  ! input vector x
                  real (c_double), intent (in), dimension(n, mu)        &
     &                                                       :: in
                  ! output vector y = A x
                  real (c_double), intent (inout), dimension(n, mu)     &
     &                                                       :: out
                  integer :: i, j, rank
                  call MPI_Comm_rank(MPI_COMM_WORLD, rank, i)
                  ! simple tridiagonal system
                  ! no exchange between processes because the first
                  ! and the last off-diagonal coefficients are removed
                  do 20 j = 1, mu
                      do 30 i = 1, n
                          if(i > 1 .and. i < n) then
                              out(i, j) = (n * rank + i) * in(i, j)     &
     &                                  + 0.5 * in(i - 1, j)            &
     &                                  + 1.5 * in(i + 1, j)
                          else
                              out(i, j) = (n * rank + i) * in(i, j)
                          end if
30                    continue
20                continue
              end subroutine my_mv
              subroutine my_prec (n, in, out, mu)
                  use, intrinsic :: iso_c_binding, only: c_int, c_double
                  use mpi
                  ! size of the local matrix
                  integer (c_int), intent (in) :: n
                  ! number of right-hand sides
                  integer (c_int), intent (in) :: mu
                  ! input vector x
                  real (c_double), intent (in), dimension(n, mu)        &
     &                                                       :: in
                  ! output vector y = M^-1 x
                  real (c_double), intent (inout), dimension(n, mu)     &
     &                                                       :: out
                  integer :: i, j, rank
                  call MPI_Comm_rank(MPI_COMM_WORLD, rank, i)
                  ! simple diagonal scaling
                  do 20 j = 1, mu
                      do 30 i = 1, n
                          out(i, j) = in(i, j) / (n * rank + i)
30                    continue
20                continue
              end subroutine my_prec
      end module my_operator

      program main
          use, intrinsic :: iso_c_binding
          use hpddm
          use mpi
          use my_operator

          implicit none
          type (c_funptr) :: mv, precond
          real (c_double), pointer :: rhs(:, :), sol(:, :)
          integer (c_int) :: rank, ierr
          integer (c_int), dimension (1:2) :: dim

          call MPI_Init(ierr)
          call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

          mv = c_funloc(my_mv)
          precond = c_funloc(my_prec)
          if(rank == 0) then
              write(*, *) "Enter the size of the local matrices and then&
     & the number of right-hand sides:"
              read(*, *) dim(1), dim(2)
          end if
          call MPI_Bcast(dim, 2, MPI_INT, 0, MPI_COMM_WORLD, ierr)

          allocate(rhs(dim(1), dim(2)))
          allocate(sol(dim(1), dim(2)))

          ! config file
          ierr = HpddmParseConfig(C_CHAR_"hpddm_f90.cfg"//C_NULL_CHAR)
          if(rank /= 0) then
              call HpddmOptionRemove(C_CHAR_"verbosity"//C_NULL_CHAR)
          end if

          ! random RHSs and first guesses
          call random_number(rhs)
          call random_number(sol)

          ! HPDDM solve
          ierr = HpddmCustomOperatorSolve(dim(1), mv, precond, rhs, sol,&
     &                                    dim(2), MPI_COMM_WORLD)
          if(rank == 0) then
              write(*, *) "Number of iterations: ", ierr
          end if
          deallocate(rhs)
          deallocate(sol)
          call MPI_Finalize(ierr)
      end program main
