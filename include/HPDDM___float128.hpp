/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2022-08-13

   Copyright (C) 2022-     Centre National de la Recherche Scientifique

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

#ifndef _HPDDM___FLOAT128_
#define _HPDDM___FLOAT128_

#if defined(__GNUC__)
#pragma GCC system_header
#endif

namespace std {
template<>
class numeric_limits<__float128> {
public:
    static constexpr bool is_specialized = true;
    static constexpr __float128 min() noexcept { return FLT128_MIN; }
    static constexpr __float128 max() noexcept { return FLT128_MAX; }
    static constexpr bool is_signed = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr __float128 epsilon() noexcept { return FLT128_EPSILON; }
};
} // std

#endif // _HPDDM___FLOAT128_
