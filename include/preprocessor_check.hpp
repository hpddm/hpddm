/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2016-02-17

   Copyright (C) 2016-     Centre National de la Recherche Scientifique

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

#ifndef _HPDDM_PREPROCESSOR_CHECK_
# define _HPDDM_PREPROCESSOR_CHECK_
# define HPDDM_STR_HELPER(x) #x
# define HPDDM_STR(x) HPDDM_STR_HELPER(x)
#endif // _HPDDM_PREPROCESSOR_CHECK_

#ifdef HPDDM_CHECK_COARSEOPERATOR
# ifdef COARSEOPERATOR
#  pragma message("COARSEOPERATOR macro already set to " HPDDM_STR(COARSEOPERATOR) ", it has now been reset")
#  undef COARSEOPERATOR
# endif
#endif
#ifdef HPDDM_CHECK_SUBDOMAIN
# ifdef SUBDOMAIN
#  pragma message("SUBDOMAIN macro already set to " HPDDM_STR(SUBDOMAIN) ", it has now been reset")
#  undef SUBDOMAIN
# endif
#endif
