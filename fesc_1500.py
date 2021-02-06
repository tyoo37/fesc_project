import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import os.path
from astropy.io import fits
from astropy.table import Table
from multiprocessing import Process, Queue
import scipy.stats as st


read_old_01Zsun = '/Volumes/THYoo/RHD_10pc/'

read_old_002Zsun = '/Volumes/THYoo/RHD_10pc_lowZ/'

read_new_01Zsun = '/Volumes/THYoo/kisti/RHD_10pc_0.1Zsun/'

read_new_1Zsun = '/Volumes/THYoo/kisti/RHD_10pc_1Zsun/'

read_new_gasrich= '/Volumes/THYoo/kisti/RHD_10pc_gasrich/G9_gasrich/'

read_new_1Zsun_highSN_new = '/Volumes/gdrive/1Zsun_SNen_new/'

read_new_03Zsun_highSN = '/Volumes/gdrive/0.3Zsun_SNen/'

read_new_01Zsun_05pc = '/Volumes/gdrive/0.1Zsun_5pc/'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Modern Computer'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] =15
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['legend.fontsize']=13

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
        sfrindex = np.where((self.starage >= 0) & (self.starage < 10))[0]
        self.SFR = np.sum(self.mp0[sfrindex]) / 1e7
        self.boxlen = partdata.info.boxlen

        self.dmxp = partdata.part.xp[0]* partdata.info.unit_l / 3.08e18
        dmm = partdata.part.mp[0]

        self.dmm = dmm * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l
        dmindex = np.where(dmm>2000)
        self.dmxpx = self.dmxp[0][dmindex]
        self.dmxpy = self.dmxp[1][dmindex]
        self.dmxpz = self.dmxp[2][dmindex]

        #self.dmm = dmm[dmindex]
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
        self.fescD = dat.read_reals(dtype='float32')

        self.photonr = dat.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD*self.photonr)/np.sum(self.photonr)
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
        self.dx = celldata.cell.dx[0] * Part.boxpc
        self.mindx = np.min(celldata.cell.dx[0])
        self.m = celldata.cell[0][4][0] *Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l *Part.unit_l *(celldata.cell.dx[0]*Part.boxlen)**3
        xHI = celldata.cell[0][4][7]
        xHII = celldata.cell[0][4][8]
        self.mHIH2 = self.m*(1-xHII)
class Cellfromdat():
    def __init__(self, dir, nout, Part):
        celldata = FortranFile(dir + 'dat/cell_%05d.dat' % nout, 'r')
        nlines,xx,nvarh=celldata.read_ints(dtype=np.int32)
        print(nlines)
        print(nvarh)
        xc = celldata.read_reals(dtype=np.double)
        yc = celldata.read_reals(dtype=np.double)
        zc = celldata.read_reals(dtype=np.double)
        dxc = celldata.read_reals(dtype=np.double)

        self.x = xc * Part.boxpc
        self.y = yc * Part.boxpc
        self.z = zc * Part.boxpc
        self.dx = dxc * Part.boxpc

        var = np.zeros((nlines,nvarh))
        for i in range(nvarh):

            var[:,i] = celldata.read_reals(dtype=np.double)

        self.nH = var[:, 0] * 30.996344
        self.m = var[:, 0] * Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l * Part.unit_l * (
                    dxc * Part.boxlen) ** 3
        xHI = var[:,7]
        xHII = var[:,8]
        self.mHIH2 = self.m * (1-xHII)

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
        self.sed_attHHe = sed_attHHe * flam2fnu
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
        self.sed_attHHe = sed_attHHe * flam2fnu

        self.sed_attHHeD = sed_attHHeD * flam2fnu
        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        self.fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
        self.fesc2 = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)

        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)


def fesc_1500(ax1,  Fesc, read, inisnap, endsnap, label,load, color,color2, edge1500,text,legend):
    numsnap = endsnap - inisnap + 1
    if load == False:
        sfrdarr = np.array([])
        timearr = np.array([])
        frarr = np.array([])
        frarrD = np.array([])
        fescarr = np.array([])
        for n in range(numsnap):
            nout = n + inisnap

            if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
                print(read + '/SAVE/cell_%05d.sav' % (nout))
                continue
            if Fesc==Fesc_new8:
                if not os.path.isfile(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout)):
                    print(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout))
                    continue
            if Fesc==Fesc_new:
                if not os.path.isfile(read + '/ray_nside4_laursen/ray_%05d.dat' % (nout)):
                    print(read + '/ray_nside4_laursen/ray_%05d.dat' % (nout))
                    continue
            Part1 = Part(read, nout)
            Fesc1 = Fesc(read, nout)

            ind1500 = np.where((Fesc1.wave >= edge1500[0]) & (Fesc1.wave < edge1500[1]))
            int1500 = np.mean(Fesc1.sed_intr[ind1500])
            attD1500 = np.mean(Fesc1.sed_attHHeD[ind1500])
            att1500 = np.mean(Fesc1.sed_attHHe[ind1500])
            print(Fesc1.sed_attHHe.size, Fesc1.sed_attHHeD.size)
            print(int1500, att1500, attD1500)
            print(Fesc1.fesc,Fesc1.fesc2)
            fluxratio = att1500/int1500
            fluxratioD = attD1500/int1500
            if np.isnan(fluxratio)==True:
                print('nan')
                continue
            if np.isnan(fluxratioD) == True:
                print('nan')
                continue
            frarr = np.append(frarr, fluxratio)
            frarrD = np.append(frarrD, fluxratioD)
            fescarr = np.append(fescarr, Fesc1.fesc)
            timearr = np.append(timearr,Part1.snaptime)

        np.savetxt(read + 'fesc_fesc1500.dat', fescarr, newline='\n', delimiter=' ')
        np.savetxt(read + 'frarr_fesc1500.dat',frarr,newline='\n',delimiter=' ')
        np.savetxt(read + 'frarrD_fesc1500.dat',frarrD,newline='\n',delimiter=' ')

        np.savetxt(read + 'timearr_fesc1500.dat',timearr,newline='\n',delimiter=' ' )
    else:
        fescarr = np.loadtxt(read + 'fesc_fesc1500.dat')
        frarr = np.loadtxt(read+'frarr_fesc1500.dat')
        timearr = np.loadtxt(read+'timearr_fesc1500.dat')
        frarrD = np.loadtxt(read+'frarrD_fesc1500.dat')

    pp1 = ax1.plot(timearr, frarrD, color=color, ls='-', lw=1.5, label=r'$\langle f_{1500} \rangle$')
    pp1 = ax1.plot([0,500], np.ones(2)*np.mean(frarrD), color='gray', ls='dashed', lw=1, label=r'$\langle f_{esc} \rangle$')

    # ax.plot(timearray, fescarray2, color=color2, ls='dotted',lw=2)

    # pp2=ax.plot(timearray, cumfescarr, color='grey', ls='dotted',lw=1, label=r'$\langle f_{esc}(t) \rangle$')
    #pp2 = ax2.plot(timearr, frarr, color=color2, ls='dashed', lw=1.5, label=r'$\langle f_{1500} \rangle w/o dust$')
    # ax2.plot(timearray, np.log10(SFRarray), color=color, label=label, ls=ls)
    # ax.text(np.min(timearray),0.1, label)
    print(np.mean(frarr))

    if text == True:
        ax1.text(60, 1, label)

   # pp = pp1 + pp2

   # labs = [l.get_label() for l in pp]

    #if legend == True:
    #    ax1.legend(pp, labs, loc=1, frameon=False)
    #ax2.set_yticks([])

    # ax.set_ylabel('log($f_{es c}$)')
    # ax2.set_ylabel('$dM_*/dt (M_\odot/yr)$')
    #ax1.set_yscale('log')
    #ax2.set_yscale('log')
    ax1.set_ylim(0.05, 1.2)
    #ax2.set_ylim(0.01, 10)
    ax1.set_xlim(0, 450)


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
#plt.style.use('dark_background')

fig = plt.figure(figsize=(8, 8), dpi=144)

ax1 = fig.add_axes([0.13, 0.1, 0.35, 0.28])
ax2 = fig.add_axes([0.58, 0.1, 0.35, 0.28])
ax3 = fig.add_axes([0.13, 0.4, 0.35, 0.28])
ax4 = fig.add_axes([0.58, 0.4, 0.35, 0.28])
ax5 = fig.add_axes([0.13, 0.7, 0.35, 0.28])
ax6 = fig.add_axes([0.58, 0.7, 0.35, 0.28])



"""

timearr1, SFR1 = SFR(Part,read_new_01Zsun, 30, 480)
timearr2, SFR2 = SFR(Part,read_new_1Zsun, 30, 473)
timearr3, SFR3 = SFR(Part,read_new_gasrich, 30, 190)
timearr4, SFR4 = SFR(Part,read_new_1Zsun_highSN, 30, 240)
"""
#ax11=ax1.twinx()
#ax22=ax2.twinx()
#ax33=ax3.twinx()
#ax44=ax4.twinx()
#ax55=ax5.twinx()
#ax66=ax6.twinx()
S18edge1500 = np.array([1475, 1525])

fesc_1500(ax1,  Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',False, 'dodgerblue','k',S18edge1500,True,False)

#fesc_1500(ax2, ax22, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True, 'dodgerblue','r',S18edge1500,False,False)
#fesc_1500(ax3, ax33, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True, 'dodgerblue','r',S18edge1500,False,False)
#fesc_1500(ax4, ax44, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True, 'dodgerblue','r',S18edge1500,False,False)
#fesc_1500(ax5, ax55, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True, 'dodgerblue','r',S18edge1500,False,False)
#fesc_1500(ax6, ax66, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True, 'dodgerblue','r',S18edge1500,False,False)

fesc_1500(ax4, Fesc_new8, read_new_1Zsun_highSN_new, 3,380,  'G9_1Zsun_SNboost',False,'dodgerblue', 'k',  S18edge1500,True, False)
fesc_1500(ax5,  Fesc_new8, read_new_03Zsun_highSN, 3,380, 'G9_03Zsun_SNboost',False,'dodgerblue', 'k',  S18edge1500,True, False)
fesc_1500(ax2,  Fesc_new, read_new_01Zsun_05pc, 3,380, 'G9_01Zsun_5pc',False,'dodgerblue', 'k',  S18edge1500,True, False)
fesc_1500(ax6,  Fesc_new8, read_new_gasrich, 30, 300, 'G9_01Zsun_gasrich',False,'dodgerblue', 'k', S18edge1500, True, False)
fesc_1500(ax3,  Fesc_new8, read_new_1Zsun, 30, 483, 'G9_1Zsun',False,'dodgerblue', 'k',  S18edge1500,True, False)

#fescplot(ax1, ax11, Part, Fesc_new8, read_new_01Zsun_lyaoff_se, 1, 30, 'G9_01Zsun_lyaoff',False, 'dodgerblue','r',True,True)
#fescplot(ax2, ax22, Part, Fesc_new, read_new_01Zsun_5pc_se, 3,380, 'G9_01Zsun_5pc',False,'dodgerblue', 'r',  True, False)

ax3.set_xticks([])
ax4.set_xticks([])
ax5.set_xticks([])
ax6.set_xticks([])


#ax11.set_yticks([])
#ax33.set_yticks([])
#ax55.set_yticks([])
#ax1.legend(frameon=False)
#ax1.set_xlabel('time (Myr)')
#ax1.set_ylabel(r'$\langle f_{esc} \rangle$')
ax3.set_ylabel('$f_{1500}$',fontsize=20)
#ax44.set_ylabel('$f_{esc}$',fontsize=20)
fig.text(0.43,0.04,'time (Myr)',fontsize=20)

plt.show()