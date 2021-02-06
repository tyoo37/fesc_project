import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time

read_old_01Zsun = '/Volumes/THYoo/RHD_10pc/'

read_old_002Zsun = '/Volumes/THYoo/RHD_10pc_lowZ/'

read_new_01Zsun = '/Volumes/THYoo/kisti/RHD_10pc_0.1Zsun/'

read_new_1Zsun = '/Volumes/THYoo/kisti/RHD_10pc_1Zsun/'

read_new_01Zsun_re = '/blackwhale/dbahck37/kisti/0.1Zsun/'

read_new_1Zsun_re = '/blackwhale/dbahck37/kisti/1Zsun/'

read_new_1Zsun_highSN = '/Volumes/THYoo/kisti/1Zsun_highSN/'


class Part():
    def __init__(self, dir, nout):
        self.dir = dir
        self.nout = nout
        partdata = readsav(self.dir + '/SAVE/part_%05d.sav' % (self.nout))
        self.snaptime = partdata.info.time*4.70430e14/365/3600/24/1e6
        pc = 3.08e18
        self.boxpc = partdata.info.boxlen * partdata.info.unit_l / pc
        xp = partdata.star.xp[0]
        self.xp = xp * partdata.info.unit_l / 3.08e18
        self.unit_l = partdata.info.unit_l
        self.unit_d = partdata.info.unit_d
        self.starage = self.snaptime - partdata.star.tp[0]*4.70430e14/365/3600/24/1e6
        self.starid = np.abs(partdata.star.id[0])
        self.mp0 = partdata.star.mp0[0] * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l

        tp = (partdata.info.time-partdata.star.tp[0]) * 4.70430e14 / 365. /24./3600/1e6
        sfrindex = np.where((tp >= 0) & (tp < 10))[0]
        self.SFR = np.sum(self.mp0[sfrindex]) / 1e7


        dat2 = FortranFile(dir + 'ray_nside4/ray_%05d.dat' % (nout), 'r')
        npart, nwave2 = dat2.read_ints()
        wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attH = dat2.read_reals(dtype=np.double)
        sed_attD = dat2.read_reals(dtype=np.double)
        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)

class Cell():
    def __init__(self, dir, nout, Part):
        self.dir = dir
        self.nout = nout
        celldata = readsav(self.dir + '/SAVE/cell_%05d.sav' % (self.nout))
        self.nH = celldata.cell[0][4][0] * 30.996344
        self.x = celldata.cell.x[0] * Part.boxpc
        self.y = celldata.cell.y[0] * Part.boxpc
        self.z = celldata.cell.z[0] * Part.boxpc
        self.dx = celldata.cell.dx[0] * Part.boxpc
        self.mindx = np.min(celldata.cell.dx[0])
        nHI = self.nH * celldata.cell[0][4][7]
        nHII = self.nH * celldata.cell[0][4][8]
        nH2 = self.nH * (1 - celldata.cell[0][4][7] - celldata.cell[0][4][8])/2
        YY= 0.24/(1-0.24)/4
        nHeII = self.nH * YY*celldata.cell[0][4][9]
        nHeIII = self.nH * YY*celldata.cell[0][4][10]
        nHeI = self.nH * YY*(1 - celldata.cell[0][4][9] - celldata.cell[0][4][10])
        ne = nHII + nHeII + nHeIII *2
        ntot = nHI + nHII + nHeI + nHeII + nHeIII + ne + nH2
        mu = celldata.cell[0][4][0] * Part.unit_d / 1.66e-24 / ntot

        self.T = celldata.cell[0][4][5]/celldata.cell[0][4][0] * 517534.72 * mu



class Clump():
    def __init__(self, dir, nout, Part):
        self.dir = dir
        self.nout = nout
        unit_d = Part.unit_d
        unit_l = Part.unit_l
        clumpdata = np.loadtxt(self.dir + '/clump3/clump_%05d.txt' % (self.nout),
                               dtype=np.double)
        self.xclump = clumpdata[:, 4] * Part.unit_l / 3.08e18
        self.yclump = clumpdata[:, 5] * Part.unit_l / 3.08e18
        self.zclump = clumpdata[:, 6] * Part.unit_l / 3.08e18

        self.massclump = clumpdata[:,
                    10] * unit_d * unit_l / 1.989e33 * unit_l * unit_l
        self.rclump = (clumpdata[:, 10] / clumpdata[:, 9] / 4 / np.pi * 3) ** (0.333333) * unit_l / 3.08e18
        self.nclump = len(self.xclump)

class YoungClumpFind():
    def __init__(self,Part,Clump,dir,snap):
        #generating star informaton belongs to clump
        self.dir=dir

        nmemberarr = np.array([])
        staridarr = np.array([])
        youngindexarr = np.array([])
        starmassarr=np.array([])
        nmembersort=np.array([])
        self.youngclumpindex=[]
        prevPart = Part(dir,snap-1)

        Part1 = Part(dir,snap)
        Clump1 = Clump(dir,snap,Part1)
        nclump = len(Clump1.xclump)

        for i in range(nclump):
            rr = np.sqrt((Clump1.xclump[i] - Part1.xp[0]) ** 2 + (Clump1.yclump[i] - Part1.xp[1]) ** 2 + (Clump1.zclump[i] - Part1.xp[2]) ** 2)
            youngindex = np.where((rr < Clump1.rclump[i]) & (Part1.starage < Part1.snaptime - prevPart.snaptime))
            preexistindex = np.where((rr<Clump1.rclump[i]) & (Part1.starage>=Part1.snaptime - prevPart.snaptime))

            youngindexarr = np.append(youngindexarr, youngindex)
            if len(youngindex[0])>0 and len(preexistindex[0])==0:
                nmember = len(youngindex[0])
                nmemberarr = np.append(nmemberarr, nmember)


                starid = Part1.starid[youngindex]
                staridarr = np.append(staridarr, starid)

                starmass = np.sum(Part1.mp0[youngindex])
                starmassarr=np.append(starmassarr,starmass)
                self.youngclumpindex.append(i)

        self.youngindexarr = youngindexarr
        self.nmember=nmemberarr
        self.starid=staridarr
        self.sf=starmassarr
        cum_nmember=np.zeros(len(self.nmember))
        for j in range(len(self.nmember)):
            cum_nmember[j]=np.sum(self.nmember[:j+1])
        self.cum_nmember=cum_nmember


def sfe_in_clump(Part,Clump,YoungClumpFind,dir,inisnap,endsnap,trace):

    numsnap = endsnap - inisnap + 1
    lifetimearr = np.array([])
    lifetimearr1 = np.array([])
    lifetimearr2 = np.array([])
    lifetimearr3 = np.array([])

    for m in range(numsnap):
        nout = inisnap + m

        Init = YoungClumpFind(Part,Clump,dir,nout)
        InitPart=Part(dir,nout)

        #self.initclumpid = np.arange(len(Init.nmember))
        print('the number of star-forming clumps = ', len(Init.youngclumpindex))
        print('the number of stars younger than 1 Myr = ', len(np.where(InitPart.starage < 1)[0]))
        print('the number of young stars in clumps = ', int(np.sum(Init.nmember)))
        for i in range(len(Init.youngclumpindex)):

            tff = np.

            print('*********************')
            print('#%d' % i, 'inisnap = ', nout, ' ,')
            print('*********************')


            nmember = int(Init.nmember[i])

            for j in range(nmember):
                print(j,'/',nmember)

                for n in range(trace):

                    #Clump1 = YoungClumpFind(Part, Clump, dir, nout)
                    traceout = inisnap + n + m + 1

                    Part1 = Part(dir, traceout)
                    Clump1=Clump(dir,traceout,Part1)

                    starindex = np.unique(np.where(Part1.starid == Init.starid[int(np.sum(Init.nmember[:i]))+j])[0])
                    if(len(starindex))!=1:
                        print(starindex)
                        continue
                    else:
                        xstar = Part1.xp[0][int(starindex)]
                        ystar = Part1.xp[1][int(starindex)]
                        zstar = Part1.xp[2][int(starindex)]
                        rr = np.sqrt((Clump1.xclump - xstar) ** 2 + (Clump1.yclump - ystar) ** 2 + (
                                Clump1.zclump - zstar) ** 2)
                        clumpindex = np.where((rr - Clump1.rclump < 0))[0]
                        if len(clumpindex)==0:
                            nn = n
                            break
                        if n==trace-1:
                            print('reaching maximum trace time')
                            nn = n
                if n ==0:
                    lifetime = nn + InitPart.starage[int(Init.youngindexarr[i])]
                else:
                    lifetime = nn + InitPart.starage[int(Init.youngindexarr[i])] - 0.5
                print(lifetime)

                if nmember ==1:
                    if n == 0:
                        lifetime1 = nn + InitPart.starage[int(Init.youngindexarr[i])]
                    else:
                        lifetime1 = nn + InitPart.starage[int(Init.youngindexarr[i])] - 0.5
                    lifetimearr1 = np.append(lifetimearr1, lifetime1)

                elif nmember > 1 and nmember <=4:
                    if n == 0:
                        lifetime2 = nn + InitPart.starage[int(Init.youngindexarr[i])]
                    else:
                        lifetime2 = nn + InitPart.starage[int(Init.youngindexarr[i])] - 0.5
                    lifetimearr2 = np.append(lifetimearr2, lifetime2)

                else:
                    if n == 0:
                        lifetime3 = nn + InitPart.starage[int(Init.youngindexarr[i])]
                    else:
                        lifetime3 = nn + InitPart.starage[int(Init.youngindexarr[i])] - 0.5
                    lifetimearr3 = np.append(lifetimearr3, lifetime3)

                lifetimearr = np.append(lifetimearr, lifetime)

    return lifetimearr, lifetimearr1, lifetimearr2, lifetimearr3




plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.titlesize'] = 13

fig = plt.figure(figsize=(16, 8), dpi=144)

ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.8])
ax2 = fig.add_axes([0.4, 0.1, 0.2, 0.8])
ax3 = fig.add_axes([0.7, 0.1, 0.2, 0.8])
"""
lifetimearr1, lifetimearr1_1, lifetimearr1_2, lifetimearr1_3 = lifetime_in_clump(Part,Clump,YoungClumpFind,read_new_01Zsun,120,300,20)

np.savetxt('liftimearr1.dat',lifetimearr1,newline='\n',delimiter=' ')
np.savetxt('liftimearr1_1.dat',lifetimearr1_1,newline='\n',delimiter=' ')
np.savetxt('liftimearr1_2.dat',lifetimearr1_2,newline='\n',delimiter=' ')
np.savetxt('liftimearr1_3.dat',lifetimearr1_3,newline='\n',delimiter=' ')
"""
lifetimearr1 = np.genfromtxt('liftimearr1.dat')
lifetimearr1_1 = np.genfromtxt('liftimearr1_1.dat')
lifetimearr1_2 = np.genfromtxt('liftimearr1_2.dat')
lifetimearr1_3 = np.genfromtxt('liftimearr1_3.dat')

print(lifetimearr1)
print(np.mean(lifetimearr1), np.mean(lifetimearr1_1), np.mean(lifetimearr1_2), np.mean(lifetimearr1_3))
ax1.hist(lifetimearr1, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='total',edgecolor='k')
ax1.hist(lifetimearr1_1, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='1 star',edgecolor='b')
ax1.hist(lifetimearr1_2, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='2-4 stars',edgecolor='purple')
ax1.hist(lifetimearr1_3, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='>5 stars',edgecolor='r')
"""
lifetimearr2, lifetimearr2_1, lifetimearr2_2, lifetimearr2_3 = lifetime_in_clump(Part,Clump,YoungClumpFind,read_new_1Zsun,120,300,20)
np.savetxt('liftimearr2.dat',lifetimearr2,newline='\n',delimiter=' ')
np.savetxt('liftimearr2_1.dat',lifetimearr2_1,newline='\n',delimiter=' ')
np.savetxt('liftimearr2_2.dat',lifetimearr2_2,newline='\n',delimiter=' ')
np.savetxt('liftimearr2_3.dat',lifetimearr2_3,newline='\n',delimiter=' ')
"""
lifetimearr2 = np.genfromtxt('liftimearr2.dat')
lifetimearr2_1 = np.genfromtxt('liftimearr2_1.dat')
lifetimearr2_2 = np.genfromtxt('liftimearr2_2.dat')
lifetimearr2_3 = np.genfromtxt('liftimearr2_3.dat')
ax2.hist(lifetimearr2, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='total',edgecolor='k')
ax2.hist(lifetimearr2_1, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='1 star',edgecolor='b')
ax2.hist(lifetimearr2_2, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='2-4 stars',edgecolor='purple')
ax2.hist(lifetimearr2_3, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='>5 stars',edgecolor='r')

print(np.mean(lifetimearr2), np.mean(lifetimearr2_1), np.mean(lifetimearr2_2), np.mean(lifetimearr2_3))
"""
lifetimearr3, lifetimearr3_1, lifetimearr3_2, lifetimearr3_3 = lifetime_in_clump(Part,Clump,YoungClumpFind,read_new_1Zsun_highSN,120,230,20)
np.savetxt('liftimearr3.dat',lifetimearr3,newline='\n',delimiter=' ')
np.savetxt('liftimearr3_1.dat',lifetimearr3_1,newline='\n',delimiter=' ')
np.savetxt('liftimearr3_2.dat',lifetimearr3_2,newline='\n',delimiter=' ')
np.savetxt('liftimearr3_3.dat',lifetimearr3_3,newline='\n',delimiter=' ')
"""
lifetimearr3 = np.genfromtxt('liftimearr3.dat')
lifetimearr3_1 = np.genfromtxt('liftimearr3_1.dat')
lifetimearr3_2 = np.genfromtxt('liftimearr3_2.dat')
lifetimearr3_3 = np.genfromtxt('liftimearr3_3.dat')
ax3.hist(lifetimearr3, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='total',edgecolor='k')
ax3.hist(lifetimearr3_1, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='1 star',edgecolor='b')
ax3.hist(lifetimearr3_2, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='2-4 stars',edgecolor='purple')
ax3.hist(lifetimearr3_3, bins=20, range=(0,20), histtype='stepfilled', fill=False,label='>5 stars',edgecolor='r')
print(np.mean(lifetimearr3), np.mean(lifetimearr3_1), np.mean(lifetimearr3_2), np.mean(lifetimearr3_3))

ax1.set_ylabel('N')
ax2.set_ylabel('N')
ax3.set_ylabel('N')
ax1.set_xlabel('lifetime(Myr)')
ax2.set_xlabel('lifetime(Myr)')
ax3.set_xlabel('lifetime(Myr)')
ax1.legend()
ax2.legend()
ax3.legend()
ax1.set_title('ref')
ax2.set_title('metal_rich')
ax3.set_title('metal_rich_SNboost')

plt.show()




