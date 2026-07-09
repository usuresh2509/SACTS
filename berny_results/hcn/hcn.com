%nprocshared=2
%mem=2GB
%chk=hcn.chk
#p External="/lustre/isaac24/scratch/usuresh3/xtb-gaussian/xtb-g" Opt=(TS, CalcFC, NoEigenTest, NoMicro)

Title: hcn ts search via xTB-Gaussian

0 1
C                  0.07753800    0.62881900    0.00000000
H                 -1.00799600    0.23097200    0.00000000
N                  0.07753800   -0.57198400    0.00000000

