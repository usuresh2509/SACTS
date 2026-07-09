%nprocshared=2
%mem=2GB
%chk=h2co.chk
#p External="/lustre/isaac24/scratch/usuresh3/xtb-gaussian/xtb-g" Opt=(TS, CalcFC, NoEigenTest, NoMicro)

Title: h2co ts search via xTB-Gaussian

0 1
C                 -0.40519700   -0.30412900   -0.00000700
O                  0.67650600    0.11471000    0.00000300
H                 -1.49653600    1.12981600   -0.00000800
H                 -1.48432900   -0.22272600    0.00002900

