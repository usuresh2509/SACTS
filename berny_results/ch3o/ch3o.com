%nprocshared=2
%mem=2GB
%chk=ch3o.chk
#p External="/lustre/isaac24/scratch/usuresh3/xtb-gaussian/xtb-g" Opt=(TS, CalcFC, NoEigenTest, NoMicro)

Title: ch3o ts search via xTB-Gaussian

0 2
C                 -0.64419200   -0.00000300    0.03454400
O                  0.74982000    0.00010400   -0.08805700
H                 -1.14783500    0.92591800   -0.22096900
H                 -1.14794500   -0.92545800   -0.22245100
H                  0.16237600   -0.00127200    0.94061100

