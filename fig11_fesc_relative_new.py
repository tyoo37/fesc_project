import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import os.path

read_old_01Zsun = '/Volumes/THYoo/RHD_10pc/'

read_old_002Zsun = '/Volumes/THYoo/RHD_10pc_lowZ/'

read_new_01Zsun = '/Volumes/THYoo/kisti/RHD_10pc_0.1Zsun/'

read_new_1Zsun = '/Volumes/THYoo/kisti/RHD_10pc_1Zsun/'

read_new_01Zsun_re = '/blackwhale/dbahck37/kisti/0.1Zsun/'

read_new_1Zsun_re = '/blackwhale/dbahck37/kisti/1Zsun/'

read_new_gasrich = '/Volumes/THYoo/kisti/RHD_10pc_gasrich/G9_gasrich/'

read_new_1Zsun_highSN_old = '/Volumes/gdrive/1Zsun_SNen_old/'
read_new_1Zsun_highSN_new = '/Volumes/gdrive/1Zsun_SNen_new/'

read_new_03Zsun_highSN = '/Volumes/gdrive/0.3Zsun_SNen/'

read_new_01Zsun_05pc = '/Volumes/gdrive/0.1Zsun_5pc/'

class Part():
    def __init__(self, dir, nout):
        self.dir = dir
        self.nout = nout

        partdata = readsav(self.dir + '/SAVE/part_%05d.sav' % (self.nout))
        self.snaptime = partdata.info.time*4.70430e14/365/3600/24/1e6
        pc = 3.08e18
        self.boxlen = partdata.info.boxlen
        self.boxpc = partdata.info.boxlen * partdata.info.unit_l / pc
        xp = partdata.star.xp[0]
        self.xp = xp * partdata.info.unit_l / 3.08e18
        self.unit_l = partdata.info.unit_l
        self.unit_d = partdata.info.unit_d
        self.starage = self.snaptime - partdata.star.tp[0]*4.70430e14/365/3600/24/1e6
        self.starid = np.abs(partdata.star.id[0])
        self.mp0 = partdata.star.mp0[0] * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l

        tp = partdata.star.tp[0] * 4.70430e14 / 365. /24./3600/1e6
        self.tp = tp
        starage = self.snaptime-tp
        sfrindex = np.where((starage >= 0) & (starage < 10))[0]
        self.SFR = np.sum(self.mp0[sfrindex]) / 1e7

        dmxp = partdata.part.xp[0]* partdata.info.unit_l / 3.08e18
        dmm = partdata.part.mp[0]

        dmm = dmm * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l
        dmindex = np.where(dmm>2000)
        self.dmxpx = dmxp[0][dmindex]
        self.dmxpy = dmxp[1][dmindex]
        self.dmxpz = dmxp[2][dmindex]

        self.dmm = dmm[dmindex]
        """
        dat = FortranFile(dir+'ray_nside4_80pc/ray_%05d.dat' % (nout), 'r')
        npart, nwave2 = dat.read_ints()
        wave = dat.read_reals(dtype=np.double)
        sed_intr = dat.read_reals(dtype=np.double)
        sed_attH = dat.read_reals(dtype=np.double)
        sed_attD = dat.read_reals(dtype=np.double)
        npixel = dat.read_ints()
        tp = dat.read_reals(dtype='float32')
        fescH = dat.read_reals(dtype='float32')
        self.fescD_80pc = dat.read_reals(dtype='float32')
        photonr = dat.read_reals(dtype=np.double)
        self.fesc_80pc = np.sum(self.fescD_80pc*photonr)/np.sum(photonr)
        """



class Cell():
    def __init__(self, dir, nout, Part):
        self.dir = dir
        self.nout = nout

        celldata = readsav(self.dir + '/SAVE/cell_%05d.sav' % (self.nout))

        self.nH = celldata.cell[0][4][0] * 30.996344
        self.x = celldata.cell.x[0] * Part.boxpc
        self.y = celldata.cell.y[0] * Part.boxpc
        self.z = celldata.cell.z[0] * Part.boxpc
        #self.vx = celldata.cell[0][4][1] * Part.unit_l / 4.70430e14
        #self.vy = celldata.cell[0][4][2] * Part.unit_l / 4.70430e14

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
        self.m = celldata.cell[0][4][0] *Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l *Part.unit_l *(celldata.cell.dx[0]*Part.boxlen)**3
        #self.T = celldata.cell[0][4][5]/celldata.cell[0][4][0] * 517534.72 * mu
class Fesc_old():
    def __init__(self, dir, nout):
        dat2 = FortranFile(dir + 'ray_nside4/ray_%05d.dat' % (nout), 'r')
        npart, nwave2 = dat2.read_ints()
        wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attH = dat2.read_reals(dtype=np.double)
        sed_attD = dat2.read_reals(dtype=np.double)
        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        self.fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)

class Fesc_new():
    def __init__(self, dir, nout):
        dat2 = FortranFile(dir + 'ray_nside4_laursen/ray_%05d.dat' % (nout), 'r')
        npart, nwave2, version = dat2.read_ints()

        self.wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attHHe = dat2.read_reals(dtype=np.double)
        sed_attHHeD = dat2.read_reals(dtype=np.double)
        sed_attHHI = dat2.read_reals(dtype=np.double)
        sed_attHH2 = dat2.read_reals(dtype=np.double)
        sed_attHHe= dat2.read_reals(dtype=np.double)
        sed_attD= dat2.read_reals(dtype=np.double)
        c = 2.9979e10
        nu = c / self.wave * 1e8
        flam2fnu = c / nu ** 2
        self.sed_intr =sed_intr *flam2fnu
        self.sed_attHHeD = sed_attHHeD *flam2fnu
        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        self.fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
        self.fesc2 = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)
        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)

class Fesc_new8():
    def __init__(self, dir, nout):
        dat2 = FortranFile(dir + 'ray_nside8_laursen/ray_%05d.dat' % (nout), 'r')
        npart, nwave2, version = dat2.read_ints()
        self.wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attHHe = dat2.read_reals(dtype=np.double)
        sed_attHHeD = dat2.read_reals(dtype=np.double)
        sed_attHHI = dat2.read_reals(dtype=np.double)
        sed_attHH2 = dat2.read_reals(dtype=np.double)
        sed_attHHe= dat2.read_reals(dtype=np.double)
        sed_attD= dat2.read_reals(dtype=np.double)
        c = 2.9979e10
        nu = c / self.wave * 1e8
        flam2fnu = c / nu ** 2
        self.sed_intr = sed_intr * flam2fnu
        self.sed_attHHeD = sed_attHHeD * flam2fnu
        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        self.fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
        self.fesc2 = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)

        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)


def fesc_rel_snapshot(Fesc1, edge900, edge1500):

    ind900 = np.where((Fesc1.wave >= edge900[0]) & (Fesc1.wave < edge900[1]))
    ind1500 = np.where((Fesc1.wave >= edge1500[0]) & (Fesc1.wave < edge1500[1]))
    """
    int900 = np.sum(Fesc1.sed_intr[ind900]) / (edge900[1] - edge900[0])
    int1500 = np.sum(Fesc1.sed_intr[ind1500]) / (edge1500[1] - edge1500[0])

    att900 = np.sum(Fesc1.sed_attHHeD[ind900]) / (edge900[1] - edge900[0])
    att1500 = np.sum(Fesc1.sed_attHHeD[ind1500]) / (edge1500[1] - edge1500[0])
    """
    int900 = np.mean(Fesc1.sed_intr[ind900])
    int1500 = np.mean(Fesc1.sed_intr[ind1500])

    att900 = np.mean(Fesc1.sed_attHHeD[ind900])
    att1500 = np.mean(Fesc1.sed_attHHeD[ind1500])
    fescrel = (att900 / att1500) / (int900 / int1500)


    return int900/int1500, fescrel, att900/att1500, att900/int900


def fesc_rel(ax, ax2, read, inisnap, endsnap, Fesc, label, loadsave):

    numsnap = endsnap - inisnap + 1
    if loadsave==False:
        fescrelarr1 = []; fescrelarr2=[]; timearr=[]; intarr1 = []; intarr2=[] ;FL900arr=[];FL900arr2=[]
        attarr = [];attarr2=[]
        for n in range(numsnap):
            nout = n + inisnap

            if not os.path.isfile(read+'/SAVE/part_%05d.sav'%nout):
                print(read+'/SAVE/part_%05d.sav'%nout)
                continue
            if not os.path.isfile(read+'/SAVE/cell_%05d.sav'%nout):
                print(read+'/SAVE/part_%05d.sav'%nout)
                continue
            if Fesc==Fesc_new8:

                if not os.path.isfile(read+'/ray_nside8_laursen/ray_%05d.dat'%nout):
                    print(read+'/ray_nside8_laursen/ray_%05d.dat'%nout)
                    continue
            if Fesc == Fesc_new:
                if not os.path.isfile(read + '/ray_nside4_laursen/ray_%05d.dat' % nout):
                    print(read + '/ray_nside4_laursen/ray_%05d.dat' % nout)
                    continue
            if n>0:
                if prev > Part1.snaptime:
                    print('skip error_info')
                    continue

            Part1 = Part(read, nout)
            Fesc1 = Fesc(read, nout)

            M17edge900 = np.array([880, 910])
            M17edge1500 = np.array([1420, 1520])
            S18edge900 = np.array([880, 910])
            S18edge1500 = np.array([1475, 1525])

            M17intratio, M17fescrel, M17attratio, FL900M17 = fesc_rel_snapshot(Fesc1, M17edge900, M17edge1500)
            S18intratio, S18fescrel, S18attratio, FL900S18 = fesc_rel_snapshot(Fesc1, S18edge900, S18edge1500)

            if np.isnan(M17fescrel)==True or np.isnan(S18fescrel)==True:
                continue


            fescrelarr1.append(M17fescrel)
            fescrelarr2.append(S18fescrel)

            timearr.append(Part1.snaptime)
            intarr1.append(M17intratio)
            intarr2.append(S18intratio)
            attarr.append(M17attratio)
            attarr2.append(S18attratio)
            FL900arr.append(FL900M17)
            FL900arr2.append(FL900S18)

            prev = Part1.snaptime

        fescrelarr11 = np.array(fescrelarr1)
        fescrelarr22 = np.array(fescrelarr2)

        intarr11 = np.array(intarr1)
        intarr22 = np.array(intarr2)
        attarr = np.array(attarr)
        attarr2 = np.array(attarr2)
        FL900arr =np.array(FL900arr)
        FL900arr2 =np.array(FL900arr2)

        np.savetxt(read+'fescrelarr_m17.dat',fescrelarr11,newline='\n',delimiter=' ')
        np.savetxt(read+'fescrelarr_s18.dat',fescrelarr22,newline='\n',delimiter=' ')
        np.savetxt(read+'intarr_m17.dat',intarr11,newline='\n',delimiter=' ')
        np.savetxt(read+'intarr_s18.dat',intarr22,newline='\n',delimiter=' ')
        np.savetxt(read+'attarr_m17.dat',attarr,newline='\n',delimiter=' ')
        np.savetxt(read+'attarr_s18.dat',attarr2,newline='\n',delimiter=' ')
        np.savetxt(read+'FL900arr_m17.dat',FL900arr,newline='\n',delimiter=' ')
        np.savetxt(read+'FL900arr_s18.dat',FL900arr2,newline='\n',delimiter=' ')

    else:
        fescrelarr11 = np.loadtxt('fescrelarr_m17.dat')
        fescrelarr22 = np.loadtxt('fescrelarr_s18.dat')
        intarr11 = np.loadtxt('intarr_m17.dat')
        intarr22 = np.loadtxt('intarr_s18.dat')
        attarr = np.loadtxt('attarr_m17.dat')
        attarr2 = np.loadtxt('attarr_s18.dat')
        FL900arr = np.loadtxt('FL900arr_m17.dat')
        FL900arr2 = np.loadtxt('FL900arr_s18.dat')

    print('median value of fesc_rel (M17)'+label, np.median(fescrelarr11))
    print('median value of fesc_rel (S18)'+label, np.median(fescrelarr22))
    print('median value of L_int ratio (M17)'+label, np.median(intarr11))
    print('median value of L_int ratio (S18)'+label, np.median(intarr22))
    print('median value of F_att ratio (M17)'+label, np.median(attarr))
    print('median value of F_att ratio (S18)'+label, np.median(attarr2))
    print('median value of F900/L900 (M17)'+label, np.median(FL900arr))
    print('median value of F900/L900 (S18)'+label, np.median(FL900arr2))


    ax.plot(timearr, fescrelarr1, color='k', label=label )
    ax.plot(timearr, fescrelarr2, color='r', label=label, ls='dashed')

    ax2.plot(timearr, intarr11, color='k', label=label )
    ax2.plot(timearr, intarr22, color='r', label=label, ls='dashed')

    ax.text(60, 0.29, label)

    ax.set_ylim(0,0.35)
    ax.set_xlim(0,500)
    ax2.set_xlim(0,500)
    ax2.set_ylim(0,0.6)
    ax2.set_yticks([0,0.2,0.4])
    ax.set_yticks([0,0.1,0.2,0.3,0.4])


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] =16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

fig = plt.figure(figsize=(11,11))

ax1 = fig.add_axes([0.13, 0.19, 0.35, 0.17])
ax2 = fig.add_axes([0.5, 0.19, 0.35, 0.17])
ax3 = fig.add_axes([0.13, 0.49, 0.35, 0.17])
ax4 = fig.add_axes([0.5, 0.49, 0.35, 0.17])
ax5 = fig.add_axes([0.13, 0.79, 0.35, 0.17])
ax6 = fig.add_axes([0.5, 0.79, 0.35, 0.17])

ax11 = fig.add_axes([0.13, 0.1, 0.35, 0.09])
ax22 = fig.add_axes([0.5, 0.1, 0.35, 0.09])
ax33 = fig.add_axes([0.13, 0.4, 0.35, 0.09])
ax44 = fig.add_axes([0.5, 0.4, 0.35, 0.09])
ax55 = fig.add_axes([0.13, 0.7, 0.35, 0.09])
ax66 = fig.add_axes([0.5, 0.7, 0.35, 0.09])

def calz(lamb):
    return 4.126+0.931/(lamb/1e4)
def redd(lamb):
    return 2.191+0.974/(lamb/1e4)

fesc_rel(ax1,ax11, read_new_01Zsun, 30, 480, Fesc_new8,'G9_01Zsun',False)
fesc_rel(ax2,ax22, read_new_01Zsun_05pc, 3, 380, Fesc_new, 'G9_01Zsun_5pc',False)
fesc_rel(ax3,ax33, read_new_1Zsun, 30, 480, Fesc_new8,  'G9_1Zsun',False)
fesc_rel(ax4,ax44, read_new_1Zsun_highSN_new, 3, 380, Fesc_new8,  'G9_1Zsun_SNboost',False)
fesc_rel(ax5,ax55, read_new_03Zsun_highSN, 3, 380, Fesc_new8,  'G9_03Zsun_SNboost',False)
fesc_rel(ax6,ax66, read_new_gasrich, 30, 300, Fesc_new8, 'G9_01Zsun_gasrich',False)


axlist=[ax1,ax2,ax3,ax4,ax5,ax6]
for ax in axlist:
    ax.plot([15, 500],[0.09,0.09], c='purple',ls='dotted')
    ax.plot([15, 500],[0.057/0.28,0.057/0.28], c='cyan', ls='dotted')
    ax.plot([15,500])
ax2list=[ax11,ax22,ax33,ax44,ax55,ax66]
for ax in ax2list:
    ax.plot([15, 500],[0.33,0.33], c='purple',ls='dotted')
    ax.plot([15, 500],[0.27,0.27], c='cyan', ls='dotted')
   # ax.set_xlabel('time (Myr)')
    #ax.set_ylabel('$f_{esc,rel}$')
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax5.set_xticks([])
ax6.set_xticks([])
ax2.set_yticks([])
ax4.set_yticks([])
ax6.set_yticks([])
ax1.set_ylabel('$f_{esc,rel}$',fontsize=20)
ax5.set_ylabel('$f_{esc,rel}$',fontsize=20)

ax3.set_ylabel('$f_{esc,rel}$',fontsize=20)
fig.text(0.43,0.04,'time (Myr)',fontsize=20)
ax22.set_yticks([])
ax44.set_yticks([])
ax66.set_yticks([])
ax11.set_ylabel('$L_{900}/L_{1500}$',fontsize=15)
ax33.set_ylabel('$L_{900}/L_{1500}$',fontsize=15)
ax55.set_ylabel('$L_{900}/L_{1500}$',fontsize=15)

print(0.057/0.28)
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig11.pdf')
plt.show()















