!
!  This file is part of HPDDM.

!  Author(s): Pierre Jolivet <pierre@joliv.et>
!       Date: 2016-04-29

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

      module hpddm
          implicit none
          interface HpddmInterfaceRoutines
              subroutine HpddmOptionRemove(string)
                  use, intrinsic :: iso_c_binding, only: c_char
                  character (c_char), intent (in) :: string
              end subroutine HpddmOptionRemove
          end interface
          interface HpddmInterfaceFunctions
              function HpddmParseConfig(string)
                  use, intrinsic :: iso_c_binding, only: c_char, c_int
                  integer (c_int) :: HpddmParseConfig
                  character (c_char), intent (in) :: string
              end function HpddmParseConfig
              function HpddmCustomOperatorSolve(n, mv, precond,         &
     &                                          in, out, mu, comm)
                      use, intrinsic :: iso_c_binding, only:            &
     &                                        c_int, c_double, c_funptr
                      integer (c_int) :: HpddmCustomOperatorSolve
                      integer (c_int), intent (in) :: n, comm, mu
                      type (c_funptr), intent (in) :: mv, precond
                      real (c_double), intent (in), dimension(n, mu)    &
     &                                                           :: in
                      real (c_double), intent (inout), dimension(n, mu) &
     &                                                           :: out
              end function HpddmCustomOperatorSolve
          end interface
      end module hpddm
