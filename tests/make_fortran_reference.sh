#!/bin/sh
# Rebuild the Fortran reference used by tests/test_fortran.py.
#
# Copies fortran/, makes only the changes a test needs, and runs 30 years:
#   - i_etaflag = 1: read the initial bed from eta_init.txt (tests/data/eta_init.txt.gz),
#     so the run is deterministic
#   - tmax = 30, teq = 5
#   - a full-precision dump of the final state (dump_full.txt)
set -e
here=$(cd "$(dirname "$0")" && pwd)
work=${1:-$here/fortran_build}
mkdir -p "$work/run"
cp "$here/../fortran/"*.f95 "$work/"
gunzip -c "$here/data/eta_init.txt.gz" > "$work/run/eta_init.txt"
cd "$work"
python3 - <<'PY'
s = open('delta5_main.f95', encoding='latin-1').read()
s = s.replace("i_etaflag = 0  !", "i_etaflag = 1  !")
s = s.replace("tmax = 710.d0", "tmax = 30.d0").replace("teq = 110.d0", "teq = 5.d0")
dump = """open(99,file='dump_full.txt')
write(99,'(a,i10)') 'num_avul', num_avul
write(99,'(a,i10)') 'num_act', num_act
write(99,'(a,es25.16e3)') 't', t
do zz=1,xmax
   write(99,'(51es25.16e3)') (eta(zz,zzz),zzz=1,ymax+1)
enddo
do zz=1,xmax
   write(99,'(51es25.16e3)') (levee(zz,zzz),zzz=1,ymax+1)
enddo
do zz=1,xmax
   write(99,'(51i8)') (top_lay(zz,zzz),zzz=1,ymax+1)
enddo
do zz=1,num_avul
   write(99,'(3es25.16e3)') (avul_out(zz,zzz),zzz=1,3)
enddo
do zz=1,xmax
   write(99,'(51es25.16e3)') (depth_chan(zz,zzz),zzz=1,ymax+1)
enddo
do zz=1,xmax
   write(99,'(51es25.16e3)') (visit(zz,zzz),zzz=1,ymax+1)
enddo
close(99)
"""
i = s.rindex("STOP")
open('delta5_main.f95', 'w', encoding='latin-1').write(s[:i] + dump + s[i:])
PY
gfortran -O2 -w -ffree-line-length-none -o delta_ref *.f95
cd run && ../delta_ref > /dev/null
echo "reference written to $work/run"
