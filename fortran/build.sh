#!/bin/sh
# Build the delta model with gfortran.
#   sh fortran/build.sh          then run ../fortran/delta from an empty directory
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
gfortran -O2 -w -ffree-line-length-none -o delta \
  delta5_main.f95 backwater.f95 bed_elev.f95 incise.f95 incise_H.f95 \
  sedplain.f95 slope5.f95 strat3D.f95 tim_visit.f95
echo "built $here/delta"
