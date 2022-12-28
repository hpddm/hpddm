/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre@joliv.et>
        Date: 2012-12-15

   Copyright (C) 2011-2014 Université de Grenoble
                 2015      Eidgenössische Technische Hochschule Zürich
                 2016-     Centre National de la Recherche Scientifique

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

#ifndef HPDDM_PRECONDITIONER_HPP_
#define HPDDM_PRECONDITIONER_HPP_

#define HPDDM_LAMBDA_F(in, input, inout, output, len, N)                                                     \
    if(len && *len) {                                                                                        \
        unsigned short* input = static_cast<unsigned short*>(in);                                            \
        unsigned short* output = static_cast<unsigned short*>(inout);                                        \
        output[0] = std::max(output[0], input[0]);                                                           \
        if(*len > 1) {                                                                                       \
            output[1] = std::max(output[1], input[1]);                                                       \
            if(*len > 2) {                                                                                   \
                output[2] = output[2] & input[2];                                                            \
                if(*len > 3) {                                                                               \
                    output[3] = output[3] & input[3];                                                        \
                    if(N == 4 && *len > 4)                                                                   \
                        output[4] = output[4] & input[4];                                                    \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
    }

#include "HPDDM_subdomain.hpp"
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD
#include "HPDDM_coarse_operator_impl.hpp"
#include "HPDDM_operator.hpp"
#endif

namespace HPDDM {
/* Class: Preconditioner
 *
 *  A base class from which <Schwarz> and <Schur> inherit.
 *
 * Template Parameters:
 *    Solver         - Solver used for the factorization of local matrices.
 *    CoarseOperator - Class of the coarse operator.
 *    K              - Scalar type. */
template<
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD || HPDDM_PETSC
#if !HPDDM_PETSC
    template<class> class Solver,
#endif
                                  class CoarseOperator,
#endif
    class K>
class Preconditioner : public Subdomain<K> {
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD || HPDDM_PETSC
    private:
#if defined(PETSC_HAVE_MPIUNI) || defined(__MINGW32__)
        template<unsigned short N>
        static void
#ifdef __MINGW32__
                    __stdcall
#endif
                              f(void* in, void* inout, int* len, MPI_Datatype*) {
            HPDDM_LAMBDA_F(in, input, inout, output, len, N)
        }
#endif
        template<typename... Types>
        CoarseOperator*& front(Types&&...) {
            return _co;
        }
        template<typename Type, typename... Types>
        CoarseOperator*& front(Type&& arg, Types&&...) {
            return std::forward<Type>(arg);
        }
    protected:
        typedef CoarseOperator co_type;
#if !HPDDM_PETSC
        /* Variable: s
         *  Solver used in <Schwarz::callNumfact> and <Schur::callNumfactPreconditioner> or <Schur::computeSchurComplement>. */
        Solver<K>           _s;
#endif
        /* Variable: co
         *  Pointer to a <Coarse operator>. */
        CoarseOperator*    _co;
        /* Variable: ev
         *  Array of deflation vectors as needed by <Preconditioner::co>. */
        K**                _ev;
        /* Variable: uc
         *  Workspace array of size <Coarse operator::local>. */
        K*                 _uc;
        /* Function: buildTwo
         *
         *  Assembles and factorizes the coarse operator.
         *
         * Template Parameter:
         *    excluded       - Greater than 0 if the main processes are excluded from the domain decomposition, equal to 0 otherwise.
         *
         * Parameters:
         *    A              - Operator used in the definition of the Galerkin matrix.
         *    comm           - Global MPI communicator. */
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD || HPDDM_SLEPC
        template<unsigned short excluded, class Operator, class Prcndtnr, typename... Types>
        typename CoarseOperator::return_type buildTwo(Prcndtnr* B, const MPI_Comm& comm,
#if HPDDM_SLEPC
                Mat A, PetscInt n, PetscInt M, PC_HPDDM_Level** const levels,
#endif
                    Types&... args) {
            static_assert(std::is_same<typename Prcndtnr::super&, decltype(*this)>::value || std::is_same<typename Prcndtnr::super::super&, decltype(*this)>::value, "Wrong preconditioner");
            typename CoarseOperator::return_type ret { };
            CoarseOperator*& co = front(args...);
            constexpr unsigned short N = std::is_same<typename Prcndtnr::super&, decltype(*this)>::value ? 3 : 4;
            unsigned short allUniform[N + 1];
            allUniform[0] = Subdomain<K>::_map.size();
#if !HPDDM_PETSC
            Option& opt = *Option::get();
            const std::string prefix = opt.getPrefix().size() > 0 ? "" : super::prefix();
            const unsigned short nu = allUniform[1] = allUniform[2] = (sizeof...(Types) == 0 && co ? co->getLocal() : opt.val<unsigned short>(prefix + "geneo_nu", opt.set(prefix + "geneo_threshold") ? 0 : 20));
#else
            unsigned short nu;
            std::string prefixC;
#if HPDDM_SLEPC
            {
                const char* prefix;
                KSPGetOptionsPrefix(levels[n]->ksp, &prefix);
                std::string prefixF(prefix);
                unsigned int pos = prefixF.rfind("levels_", prefixF.size() - 1);
                unsigned short levelF = std::stoi(prefixF.substr(pos + 7, prefixF.size() - 1));
                if(levelF + 1 == M)
                    prefixC = prefixF.substr(0, pos) + "coarse_";
                else
                    prefixC = prefixF.substr(0, pos + 7) + std::to_string(levelF + 1) + "_";
                nu = allUniform[1] = allUniform[2] = (sizeof...(Types) == 0 && co ? co->getLocal() : levels[n]->nu);
            }
#endif
#endif
            allUniform[3] = static_cast<unsigned short>(~nu);
            if(N == 4)
                allUniform[4] = nu > 0 ? nu : std::numeric_limits<unsigned short>::max();
            {
                MPI_Op op;
#if defined(PETSC_HAVE_MPIUNI) || defined(__MINGW32__)
                MPI_Op_create(&f<N>, 1, &op);
#else
                auto f = [](void* in, void* inout, int* len, MPI_Datatype*) -> void {
                    HPDDM_LAMBDA_F(in, input, inout, output, len, N)
                };
                MPI_Op_create(f, 1, &op);
#endif
                MPI_Allreduce(MPI_IN_PLACE, allUniform, N + 1, MPI_UNSIGNED_SHORT, op, comm);
                MPI_Op_free(&op);
            }
            if(nu > 0 || allUniform[2] != 0 || allUniform[3] != std::numeric_limits<unsigned short>::max()) {
#if !HPDDM_PETSC
                const bool uniformity = (N == 3 && opt.set(prefix + "geneo_force_uniformity") && allUniform[1] == static_cast<unsigned short>(~allUniform[3]));
#else
                bool uniformity = false;
                {
                    PetscBool flg;
                    PetscBool uniform;
                    PetscOptionsGetBool(nullptr, prefixC.c_str(), "-force_uniformity", &uniform, &flg);
                    if(flg)
                        uniformity = uniform;
                }
#endif
                if(sizeof...(Types) == 0) {
                    delete co;
                    co = new CoarseOperator;
                }
                co->setLocal(uniformity ? allUniform[1] : nu);
#if !HPDDM_PETSC
                double construction = MPI_Wtime();
                const std::string prev = opt.getPrefix();
                std::string level;
                if(prev.size() == 0) {
                    std::string sub;
                    if(prefix.size() >= 8) {
                        sub = prefix.substr(prefix.size() - 8, std::string::npos);
                        const std::size_t find = sub.find("level_", 0);
                        if(find == std::string::npos)
                            level = prefix + "level_2_";
                        else {
                            sub = sub.substr(6, 1);
                            level = prefix.substr(0, prefix.size() - 2) + std::to_string(std::stoi(sub) + 1) + "_";
                        }
                    }
                    else
                        level = prefix + "level_2_";
                }
                else {
                    std::string sub = prev.substr(6, std::string::npos);
                    const std::size_t find = sub.find("_", 0);
                    sub = sub.substr(0, find);
                    level = prefix.substr(0, prefix.size() - prev.size()) + "level_" + std::to_string(std::stoi(sub) + 1) + "_";
                }
                {
                    const unsigned short verbosity = opt.val<unsigned short>(prefix + "verbosity");
                    opt.setPrefix(level);
                    if(!opt.set("verbosity") && verbosity) {
                        opt["verbosity"] = verbosity;
                    }
                }
#endif
                if((allUniform[2] == nu && allUniform[3] == static_cast<unsigned short>(~nu)) || uniformity)
                    ret = co->template construction<1, excluded>(Operator(*B, allUniform[0], (allUniform[1] << 12) + allUniform[0],
#if HPDDM_SLEPC
                        A, levels[n + 1], prefixC,
#endif
                        args...), comm);
                else if(N == 4 && allUniform[2] == 0 && allUniform[3] == static_cast<unsigned short>(~allUniform[4]))
                    ret = co->template construction<2, excluded>(Operator(*B, allUniform[0], (allUniform[1] << 12) + allUniform[0],
#if HPDDM_SLEPC
                        A, levels[n + 1], prefixC,
#endif
                        args...), comm);
                else
                    ret = co->template construction<0, excluded>(Operator(*B, allUniform[0], (allUniform[1] << 12) + allUniform[0],
#if HPDDM_SLEPC
                        A, levels[n + 1], prefixC,
#endif
                        args...), comm);
#if !HPDDM_PETSC
                if(co->getRank() == 0 && opt.val<char>("verbosity", 0) > 1) {
                    std::stringstream ss;
                    construction = MPI_Wtime() - construction;
                    ss << std::setprecision(3) << construction;
                    const unsigned short p = opt.val<unsigned short>("p", 1);
                    const std::string line = " --- coarse operator transferred " + std::string(Operator::_factorize ? "and factorized " : "") + std::string("by ") + to_string(p) + " process" + (p == 1 ? "" : "es") + " (in " + ss.str() + "s)";
                    std::cout << line << std::endl;
                    std::cout << std::right << std::setw(line.size()) << "(criterion = " + to_string(allUniform[2] == nu && allUniform[3] == static_cast<unsigned short>(~nu) ? nu : (N == 4 && allUniform[3] == static_cast<unsigned short>(~allUniform[4]) ? -co->getLocal() : (uniformity ? allUniform[1] : 0))) + ")" << std::endl;
                    std::cout.unsetf(std::ios_base::adjustfield);
                }
                opt.setPrefix(prev);
#endif
            }
            else {
                delete co;
                co = nullptr;
#if HPDDM_SLEPC
                ret = PETSC_ERR_ARG_WRONG;
#endif
            }
            return ret;
        }
#endif
#if !HPDDM_PETSC
        void destroySolver() {
            _s.dtor();
            Option& opt = *Option::get();
            if(opt.val<unsigned short>("reuse_preconditioner") >= 1)
                opt["reuse_preconditioner"] = 1;
        }
#endif
    public:
        /* Function: start
         *
         *  Allocates the array <Preconditioner::uc> depending on the number of right-hand sides to be solved by an <Iterative method>.
         *
         * Parameter:
         *    mu             - Number of right-hand sides. */
        void start(const unsigned short& mu = 1) const {
            delete [] _uc;
            K** ptr = const_cast<K**>(&_uc);
            *ptr = new K[mu * _co->getSizeRHS()];
        }
#if HPDDM_SCHWARZ
        struct CoarseCorrection {
            virtual void operator()(const K* const in, K* const out) = 0;
            virtual void operator()(const K* const in, K* const out, int n, unsigned short mu) {
                for(unsigned short nu = 0; nu < mu; ++nu)
                    operator()(in + nu * n, out + nu * n);
            }
            virtual ~CoarseCorrection() { };
        };
        CoarseCorrection*  _cc;
#endif
        Preconditioner() : _co(), _ev(), _uc()
#if HPDDM_SCHWARZ
                                              , _cc()
#endif
                                                      { }
        Preconditioner(const Preconditioner&) = delete;
        ~Preconditioner() {
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD || (HPDDM_PETSC && defined(PETSCHPDDM_H))
            dtor();
#endif
        }
        /* Function: initialize
         *
         *  Initializes a two-level preconditioner.
         *
         * Parameter:
         *    deflation      - Number of local deflation vectors. */
        void initialize(const unsigned short& deflation) {
            if(!_co) {
                _co = new CoarseOperator;
                _co->setLocal(deflation);
            }
        }
#if !HPDDM_PETSC
        /* Function: callSolve
         *
         *  Applies <Preconditioner::s> to multiple right-hand sides in-place.
         *
         * Parameters:
         *    x              - Input right-hand sides, solution vectors are stored in-place.
         *    n              - Number of input right-hand sides. */
        void callSolve(K* const x, const unsigned short& n = 1) const { _s.solve(x, n); }
#endif
        /* Function: getVectors
         *  Returns a constant pointer to <Preconditioner::ev>. */
        const K* const* getVectors() const { return _ev; }
        /* Function: setVectors
         *  Sets the pointer <Preconditioner::ev>. */
        void setVectors(K** const& ev) { _ev = ev; }
        /* Function: destroyVectors
         *  Destroys the pointer <Preconditioner::ev> using a custom deallocator. */
        void destroyVectors(void (*dtor)(void*)) {
            if(_ev)
                dtor(*_ev);
            dtor(_ev);
            _ev = nullptr;
        }
        /* Function: getLocal
         *  Returns the value of <Coarse operator::local>. */
        constexpr unsigned short getLocal() const { return _co ? _co->getLocal() : 0; }
        /* Function: getAddrLocal
         *  Returns the address of <Coarse operator::local> or <i__0> if <Preconditioner::co> is not allocated. */
        const int* getAddrLocal() const { return _co ? _co->getAddrLocal() : &i__0; }
#else
    protected:
        Preconditioner() { };
#endif
    protected:
        explicit Preconditioner(const Subdomain<K>& s) : super(s)
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD || HPDDM_PETSC
                                                                 , _co(), _ev(), _uc()
#if HPDDM_SCHWARZ
                                                                                      , _cc()
#endif
#endif
                                                                                              { };
#if HPDDM_SCHWARZ || HPDDM_FETI || HPDDM_BDD || (HPDDM_PETSC && defined(PETSCHPDDM_H))
        void dtor() {
#if !HPDDM_PETSC
            _s.dtor();
#endif
            delete _co;
            _co = nullptr;
            if(_ev)
                delete [] *_ev;
            delete [] _ev;
            _ev = nullptr;
            delete [] _uc;
            _uc = nullptr;
#if HPDDM_SCHWARZ
            delete _cc;
            _cc = nullptr;
#endif
        }
#endif
    public:
        /* Typedef: super
         *  Type of the immediate parent class <Subdomain>. */
        typedef Subdomain<K> super;
#if HPDDM_INEXACT_COARSE_OPERATOR
        template<class Preconditioner, class T> friend class MatrixAccumulation;
#endif
};
} // HPDDM
#endif // HPDDM_PRECONDITIONER_HPP_
