#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2016-02-29

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
"""

from __future__ import print_function
import sys
import re
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from subprocess import Popen, PIPE
import pandas
pandas.set_option('display.width', pandas.util.terminal.get_terminal_size()[0])

if len(sys.argv) < 2:
    sys.exit(1)

pair = re.compile(r'\(([^,\)]+),([^,\)]+)\)')

with open(sys.argv[2], 'r') as input:
    k = 0
    for line in input:
        if not line.startswith('# '):
            words = line.split()
            y = 0
            for w in words:
                if k != 0:
                    if y == 2:
                        match = pair.match(w)
                        if match is not None:
                            scalar = "-DFORCE_COMPLEX"
                        else:
                            scalar = None
                        break
                y += 1
            else:
                k += 1
                continue
            break
        else:
            continue
        break

candidate = [ "MUMPS", "PASTIX", "MKL", "SUITESPARSE", "DISSECTION" ];
with open('Makefile.inc', 'r') as input:
    for line in input:
        if not line.startswith("#"):
            word = line.split("_LIBS")[0]
            if candidate.count(word):
                if word == "MKL":
                    word = "MKL_PARDISO"
                if word == "MUMPS" or word == "PASTIX":
                    numbering = "F"
                else:
                    numbering = "C"
                print(" --- compiling " + sys.argv[1] + " with solver " + word )
                process = Popen("make " + sys.argv[1] + " SUBSOLVER=" + word + " HPDDMFLAGS=\"-O3 -DHPDDM_NUMBERING=\\'" + numbering + "\\' -DHPDDM_SCHWARZ=0 -DHPDDM_FETI=0 -DHPDDM_BDD=0 " + (scalar if scalar is not None else "") + "\"", stdout = PIPE, stderr = PIPE, shell = True)
                (output, err) = process.communicate()
                exit_code = process.wait()
                if err is not "":
                    print(" --- compilation failed with the following message:")
                    print(err)
                    continue
                else:
                    print(" --- benchmarking solver " + word + " with matrix " + sys.argv[2], end = "")
                    if len(sys.argv) > 3:
                        print(" and additional flags \"" + ' '.join(sys.argv[3:]) + "\"")
                    else:
                        print("")
                    process = Popen(' '.join(sys.argv[1:]), stdout = PIPE, shell = True)
                    (output, err) = process.communicate()
                    exit_code = process.wait()
                    if exit_code != 0:
                        print(" --- benchmarking failed with the following message:")
                        print(err)
                        continue
                    print(''.join(re.findall("//.*?\n" , output)), end = "")
                    df = pandas.read_table(StringIO(re.sub(re.compile("//.*?\n" ), "" , output)), sep = "\t", lineterminator = "\n", header = None)
                    print(df.describe(percentiles = [ 0.5 ]).transpose())
