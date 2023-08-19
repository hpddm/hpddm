/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
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

#ifndef HPDDM_SPECIFICATIONS_HPP_
#define HPDDM_SPECIFICATIONS_HPP_

#if defined(__GNUC__)
#pragma GCC system_header
#endif

#if defined(PETSC_HAVE_REAL___FP16)
namespace std {
template<>
class numeric_limits<__fp16> {
public:
    static constexpr bool is_specialized = true;
    static constexpr __fp16 min() noexcept { return 6.103515625e-5; }
    static constexpr __fp16 max() noexcept { return 6.5504e+4; }
    static constexpr bool is_signed = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr __fp16 epsilon() noexcept { return 9.765625e-4; }
};
} // std
#endif

#if defined(PETSC_HAVE_REAL___FLOAT128) && !(defined(__NVCC__) || defined(__CUDACC__))
# include <quadmath.h>
namespace std {
# if defined(PETSC_PCHPDDM_MAXLEVELS) && !defined(__MINGW32__)
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
# endif
inline __float128 pow(__float128 x, int p) { return powq(x, __float128(p)); }
} // std
namespace HPDDM {
template<class K, typename std::enable_if<!std::is_same<K, __complex128>::value>::type* = nullptr>
void assign(K& v, const underlying_type<K>& re, const underlying_type<K>& im) {
    v.real(re);
    v.imag(im);
}
template<class K, typename std::enable_if<std::is_same<K, __complex128>::value>::type* = nullptr>
void assign(K& v, const underlying_type<K>& re, const underlying_type<K>& im) {
    __real__ v = re;
    __imag__ v = im;
}
} // HPDDM
#endif

namespace HPDDM {
template<class K> struct Wrapper;
template<class K>
inline void selectNu(unsigned short target, std::vector<std::pair<unsigned short, HPDDM::complex<underlying_type<K>>>>& q, unsigned short n, const K* const alphar, const K* const alphai, const K* const beta = nullptr) {
    for(unsigned short i = 0; i < n; ++i) {
#if !defined(PETSC_PCHPDDM_MAXLEVELS) || !defined(PETSC_HAVE_REAL___FLOAT128) || defined(__NVCC__) || defined(__CUDACC__)
        HPDDM::complex<underlying_type<K>> tmp(Wrapper<K>::is_complex ? alphar[i] : HPDDM::complex<underlying_type<K>>(HPDDM::real(alphar[i]), HPDDM::real(alphai[i])));
#else
        HPDDM::complex<underlying_type<K>> tmp;
        if(Wrapper<K>::is_complex)
            tmp = alphar[i];
        else
            assign(tmp, HPDDM::real(alphar[i]), HPDDM::real(alphai[i]));
#endif
        if(beta)
             tmp /= beta[i];
        q.emplace_back(i, tmp);
    }
    using type = typename std::vector<std::pair<unsigned short, HPDDM::complex<underlying_type<K>>>>::const_reference;
    switch(target) {
        case HPDDM_RECYCLE_TARGET_LM: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return HPDDM::norm(lhs.second) > HPDDM::norm(rhs.second); }); break;
        case HPDDM_RECYCLE_TARGET_SR: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return HPDDM::real(lhs.second) < HPDDM::real(rhs.second); }); break;
        case HPDDM_RECYCLE_TARGET_LR: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return HPDDM::real(lhs.second) > HPDDM::real(rhs.second); }); break;
        case HPDDM_RECYCLE_TARGET_SI: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return HPDDM::imag(lhs.second) < HPDDM::imag(rhs.second); }); break;
        case HPDDM_RECYCLE_TARGET_LI: std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return HPDDM::imag(lhs.second) > HPDDM::imag(rhs.second); }); break;
        default:                      std::sort(q.begin(), q.end(), [](type lhs, type rhs) { return HPDDM::norm(lhs.second) < HPDDM::norm(rhs.second); });
    }
}
} // HPDDM

#endif // HPDDM_SPECIFICATIONS_HPP_
