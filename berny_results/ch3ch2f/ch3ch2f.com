%nprocshared=2
%mem=2GB
%chk=ch3ch2f.chk
#p External="/lustre/isaac24/scratch/usuresh3/xtb-gaussian/xtb-g" Opt=(TS, CalcFC, NoEigenTest, NoMicro)

Title: ch3ch2f ts search via xTB-Gaussian

0 1
C                  0.98758000   -0.55478400    0.00005000
C                  0.60860300    0.78998800   -0.00000500
F                 -1.45007800   -0.13804500    0.00002100
H                  1.39815200   -0.96648100    0.91334000
H                  0.43123600    1.32271800    0.91828100
H                  1.39940300   -0.96634500   -0.91275500
H                 -0.18624800   -0.88127100   -0.00096900
H                  0.43105600    1.32256100   -0.91835000

