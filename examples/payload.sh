#! /bin/bash

#
#  This file is part of HPDDM.
#
#  Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
#       Date: 2015-12-14
#
#  Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
#                2016-     Centre National de la Recherche Scientifique
#
#  HPDDM is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  HPDDM is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with HPDDM.  If not, see <http://www.gnu.org/licenses/>.
#

# https://github.com/fearside/ProgressBar/
function ProgressBar {
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done

    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")
    printf "\r[ ${_fill// /#}${_empty// /-} ] ${_progress}%% ${3}"
}

TMPFILE=$(mktemp /tmp/hpddm-payload.XXXXXX)
make clean > /dev/null
for CXX in g++ clang++ # icpc
do
    I=0
    START=$SECONDS
    if [ "$CXX" == "g++" ]; then
        if [[ "$OSTYPE" == darwin* ]]; then
            SUFFIX=-5
        fi
        export OMPI_CC=gcc${SUFFIX}
        export MPICH_CC=gcc${SUFFIX}
        export OMPI_CXX=g++${SUFFIX}
        export MPICH_CXX=g++${SUFFIX}
    elif [ "$CXX" == "icpc" ]; then
        export OMPI_CC=icc
        export MPICH_CC=icc
        export OMPI_CXX=icpc
        export MPICH_CXX=icpc
        sed -i\ '' 's/ nullptr>/ (void*)0>/g; s/static constexpr const char/const char/g' include/*.hpp
    else
        export OMPI_CC=clang
        export MPICH_CC=clang
        export OMPI_CXX=clang++
        export MPICH_CXX=clang++
    fi
    for SUBSOLVER in "MUMPS"
    do
        for N in C F
        do
            for OTHER in "" "-DFORCE_COMPLEX" "-DGENERAL_CO" "-DFORCE_SINGLE" "-DFORCE_SINGLE -DFORCE_COMPLEX"
            do
                if [[ "$OTHER" == "-DGENERAL_CO" ]];
                then
                    SOLVER_LIST="MUMPS HYPRE"
                else
                    SOLVER_LIST="MUMPS"
                fi
                for SOLVER in $SOLVER_LIST
                do
                    ProgressBar $I 36 "HPDDMFLAGS=\"-DHPDDM_NUMBERING=\\\'$N\\\' $OTHER\" SOLVER=${SOLVER} SUBSOLVER=${SUBSOLVER}"
                    make -j4 all ./bin/driver HPDDMFLAGS="-DHPDDM_NUMBERING=\'$N\' $OTHER" SOLVER=${SOLVER} SUBSOLVER=${SUBSOLVER} 1> /dev/null 2>$TMPFILE
                    if [ $? -ne 0 ]; then
                        echo -e "\n[ \033[91;1mFAIL\033[0m ]"
                        cat $TMPFILE
                        unlink $TMPFILE
                        exit 1
                    fi
                    I=$((I + 1))
                    if  [[ (! "$CXX" == "g++" || ! "$OSTYPE" == darwin*) ]];
                    then
                        ProgressBar $I 36 "test                                                                                                 "
                        if [[ ! "$SUBSOLVER" == "PASTIX" ]];
                        then
                            if [[ ("$OTHER" == "" || "$OTHER" == "-DFORCE_COMPLEX" || "$OTHER" == "-DGENERAL_CO" || ! "$OSTYPE" == darwin*) ]];
                            then
                                make test HPDDMFLAGS="-DHPDDM_NUMBERING=\'$N\' $OTHER" SOLVER=${SOLVER} SUBSOLVER=${SUBSOLVER} 1> $TMPFILE 2>&1
                            elif [[ "$OSTYPE" == darwin* ]];
                            then
                                unlink lib/libhpddm_python.dylib 2>/dev/null
                                make test_cpp test_c HPDDMFLAGS="-DHPDDM_NUMBERING=\'$N\' $OTHER" SOLVER=${SOLVER} SUBSOLVER=${SUBSOLVER} 1> $TMPFILE 2>&1
                            fi
                            if [ $? -ne 0 ]; then
                                echo -e "\n[ \033[91;1mFAIL\033[0m ] HPDDMFLAGS=\"-DHPDDM_NUMBERING=\\\'$N\\\' $OTHER\" SOLVER=${SOLVER} SUBSOLVER=${SUBSOLVER}"
                                cat $TMPFILE
                                unlink $TMPFILE
                                exit 1
                            fi
                        fi
                        if [[ ("$OTHER" == "" || "$OTHER" == "-DFORCE_COMPLEX") ]];
                        then
                            make test_bin/driver HPDDMFLAGS="-DHPDDM_NUMBERING=\'$N\' $OTHER" SOLVER=${SOLVER} SUBSOLVER=${SUBSOLVER} 1> $TMPFILE 2>&1
                        fi
                        if [ $? -ne 0 ]; then
                            echo -e "\n[ \033[91;1mFAIL\033[0m ] HPDDMFLAGS=\"-DHPDDM_NUMBERING=\\\'$N\\\' $OTHER\" SOLVER=${SOLVER} SUBSOLVER=${SUBSOLVER}"
                            cat $TMPFILE
                            unlink $TMPFILE
                            exit 1
                        fi
                    fi
                    I=$((I + 1))
                    ProgressBar $I 36 "                                                                                                     "
                    make clean > /dev/null
                    I=$((I + 1))
                    ProgressBar $I 36 "                                                                                                     "
                done
            done
        done
    done
    echo -e "\n[ \033[92;1mOK\033[0m ] ($CXX, $((SECONDS - START)) seconds)"
done
unlink $TMPFILE
