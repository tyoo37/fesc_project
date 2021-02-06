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


    return int900/int1500, fescrel, att900/att1500, att900/int900, att1500/int1500, att900, int900,int1500

def onesigerr(arr):
    return np.median(arr)-np.percentile(arr,25), np.percentile(arr,75)-np.median(arr)

def fesc_rel(ax, ax2, ax3,read, inisnap, endsnap, Fesc, label, loadsave):

    numsnap = endsnap - inisnap + 1
    if loadsave==False:
        fescrelarr1 = []; fescrelarr2=[]; timearr=[]; intarr1 = []; intarr2=[] ;FL900arr=[];FL900arr2=[]
        attarr = [];attarr2=[]; att900arr=[]; int900arr =[];int1500arr=[];FL1500arr=[]
        prev=0

        for n in range(numsnap):
            nout = n + inisnap
            if not os.path.isfile(read+'/SAVE/part_%05d.sav'%nout):
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

            print(nout)
            if read==read_new_1Zsun:
                if nout==26 or nout==27 or nout==28 or nout==29:
                    continue
            Part1 = Part(read, nout)
            if n > 1:
                if prev > Part1.snaptime:
                    print('skip error_info')
                    continue
            Fesc1 = Fesc(read, nout)
            if np.isnan(Fesc1.fesc) == True:
                continue
            M17edge900 = np.array([880, 910])
            M17edge1500 = np.array([1420, 1520])
            S18edge900 = np.array([880, 910])
            S18edge1500 = np.array([1475, 1525])

            M17intratio, M17fescrel, M17attratio, FL900M17,FL1500M17, att900M17, int900M17,int1500M17 = fesc_rel_snapshot(Fesc1, M17edge900, M17edge1500)
            S18intratio, S18fescrel, S18attratio, FL900S18,FL1500S18, att900S18, int900S18, int1500S18 = fesc_rel_snapshot(Fesc1, S18edge900, S18edge1500)

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
            att900arr.append(att900M17)
            int900arr.append(int900M17)
            int1500arr.append(int1500M17)
            FL1500arr.append(FL1500M17)


            prev = Part1.snaptime

        fescrelarr11 = np.array(fescrelarr1)
        fescrelarr22 = np.array(fescrelarr2)

        intarr11 = np.array(intarr1)
        intarr22 = np.array(intarr2)
        attarr = np.array(attarr)
        attarr2 = np.array(attarr2)
        FL900arr =np.array(FL900arr)
        FL900arr2 =np.array(FL900arr2)
        timearr = np.array(timearr)
        att900arr =np.array(att900arr)
        int1500arr = np.array(int1500arr)
        int900arr = np.array(int900arr)
        FL1500arr = np.array(FL1500arr)

        np.savetxt(read+'fescrelarr_m17.dat',fescrelarr11,newline='\n',delimiter=' ')
        np.savetxt(read+'fescrelarr_s18.dat',fescrelarr22,newline='\n',delimiter=' ')
        np.savetxt(read+'intarr_m17.dat',intarr11,newline='\n',delimiter=' ')
        np.savetxt(read+'intarr_s18.dat',intarr22,newline='\n',delimiter=' ')
        np.savetxt(read+'attarr_m17.dat',attarr,newline='\n',delimiter=' ')
        np.savetxt(read+'attarr_s18.dat',attarr2,newline='\n',delimiter=' ')
        np.savetxt(read+'FL900arr_m17.dat',FL900arr,newline='\n',delimiter=' ')
        np.savetxt(read+'FL900arr_s18.dat',FL900arr2,newline='\n',delimiter=' ')
        np.savetxt(read+'timearr_rel.dat',timearr,newline='\n',delimiter=' ')
        np.savetxt(read+'att900arr.dat',att900arr,newline='\n',delimiter=' ')
        np.savetxt(read+'int900arr.dat',int900arr,newline='\n',delimiter=' ')
        np.savetxt(read+'int1500arr.dat',int1500arr,newline='\n',delimiter=' ')
        np.savetxt(read+'FL1500arr.dat',FL1500arr,newline='\n',delimiter=' ')
    else:
        fescrelarr11 = np.loadtxt(read+'fescrelarr_m17.dat')
        fescrelarr22 = np.loadtxt(read+'fescrelarr_s18.dat')
        intarr11 = np.loadtxt(read+'intarr_m17.dat')
        intarr22 = np.loadtxt(read+'intarr_s18.dat')
        attarr = np.loadtxt(read+'attarr_m17.dat')
        attarr2 = np.loadtxt(read+'attarr_s18.dat')
        FL900arr = np.loadtxt(read+'FL900arr_m17.dat')
        FL900arr2 = np.loadtxt(read+'FL900arr_s18.dat')
        FL1500arr = np.loadtxt(read+'FL1500arr.dat')
        timearr = np.loadtxt(read+'timearr_rel.dat')
        att900arr = np.loadtxt(read+'att900arr.dat')
        FL1500arr = np.loadtxt(read+'FL1500arr.dat')
        int900arr = np.loadtxt(read+'int900arr.dat')
        int1500arr = np.loadtxt(read+'int1500arr.dat')

    print('median value of fesc_rel (M17)'+label, np.median(fescrelarr11), onesigerr(fescrelarr11))
    print('median value of fesc_rel (S18)'+label, np.median(fescrelarr22), onesigerr(fescrelarr22))
    print('median value of L_int ratio (M17)'+label, np.median(intarr11), onesigerr(intarr11))
    print('median value of L_int ratio (S18)'+label, np.median(intarr22), onesigerr(intarr22))
    print('median value of F_att ratio (M17)'+label, np.median(attarr), onesigerr(attarr))
    print('median value of F_att ratio (S18)'+label, np.median(attarr2), onesigerr(attarr2))
    print('median value of F900/L900 (M17)'+label, np.median(FL900arr), onesigerr(FL900arr))
    print('median value of F900/L900 (S18)'+label, np.median(FL900arr2), onesigerr(FL900arr2))
    print(np.mean(FL900arr))
   # print(np.median(attarr)-np.percentile(attarr,50-18.2689),np.percentile(attarr,68.2689)-np.median(attarr))
    #ind = np.where(timearr>150)[0]
    f900lum = np.sum(int900arr*FL900arr)/np.sum(int900arr)
    f1500lum = np.sum(int1500arr*FL1500arr)/np.sum(int1500arr)
    print('median value of F900/L900 /w luminsoity weighted'+label, f900lum)
    print('median value of F1500/L1500 /w luminsoity weighted'+label, f1500lum)

    print(' ')
    #ax.plot(timearr, fescrelarr1, color='k', label=label )
    #ax.plot(timearr, fescrelarr2, color='r', label=label, ls='dashed')
    ax.plot([0, 500], np.ones(2) * np.median(attarr), c='gray', ls='dashed',lw=2)
    #ax.fill_between([0, 500], np.ones(2) * np.percentile(attarr, 50 - 18.2689), np.ones(2) * np.percentile(attarr, 68.2689),
    #                facecolors='orange', alpha=0.5)

    ax.plot(timearr, attarr, color='k', label=label)

    #ax.plot(timearr, np.ones(len(timearr))*np.median(attarr), color='gray',ls='dashed',lw=1.5)
    #ax.plot(timearr, attarr2, color='r', label=label, ls='dashed')
    ax2.plot([0, 500], np.ones(2) * np.median(intarr11), c='gray', ls='dashed',lw=2)
    #ax2.fill_between([0, 500], np.ones(2) * np.percentile(intarr11, 50 - 18.2689),
    #                 np.ones(2) * np.percentile(intarr11, 68.2689), facecolors='orange', alpha=0.5)

    ax2.plot(timearr, intarr11, color='k', label=label )

    ax3.plot(timearr,FL900arr, color='k')

    #ax2.plot(timearr, np.ones(len(timearr))*np.median(intarr11), color='gray',ls='dashed',lw=1.5)

    #ax2.plot(timearr, intarr22, color='r', label=label, ls='dashed')
    ax.text(480, 0.095, label, ha='right')

    ax2.text(480, 0.55, label, ha='right')
    ax3.text(480,0.55,label,ha='right')
    ax.set_ylim(0,0.12)
    ax3.set_ylim(0,0.12)
    ax.set_xlim(0,500)
    ax2.set_xlim(0,500)
    ax3.set_xlim(0,500)
    ax2.set_ylim(0,0.7)
    ax2.set_yticks([0.00,0.25,0.50])
    ax.set_yticks([0,0.05,0.10])


    #ax.set_yticks([])



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

ax1 = fig.add_axes([0.7, 0.1, 0.25, 0.13])
ax2 = fig.add_axes([0.7, 0.25, 0.25, 0.13])
ax3 = fig.add_axes([0.7, 0.4, 0.25, 0.13])
ax4 = fig.add_axes([0.7, 0.55, 0.25, 0.13])
ax5 = fig.add_axes([0.7, 0.7, 0.25, 0.13])
ax6 = fig.add_axes([0.7, 0.85, 0.25, 0.13])

ax11 = fig.add_axes([0.13, 0.1, 0.25, 0.13])
ax22 = fig.add_axes([0.13, 0.25, 0.25, 0.13])
ax33 = fig.add_axes([0.13, 0.4, 0.25, 0.13])
ax44 = fig.add_axes([0.13, 0.55, 0.25, 0.13])
ax55 = fig.add_axes([0.13, 0.7, 0.25, 0.13])
ax66 = fig.add_axes([0.13, 0.85, 0.25, 0.13])

ax111 = fig.add_axes([0.4, 0.1, 0.25, 0.13])
ax222 = fig.add_axes([0.4, 0.25, 0.25, 0.13])
ax333 = fig.add_axes([0.4, 0.4, 0.25, 0.13])
ax444 = fig.add_axes([0.4, 0.55, 0.25, 0.13])
ax555 = fig.add_axes([0.4, 0.7, 0.25, 0.13])
ax666 = fig.add_axes([0.4, 0.85, 0.25, 0.13])

def calz(lamb):
    return 4.126+0.931/(lamb/1e4)
def redd(lamb):
    return 2.191+0.974/(lamb/1e4)
axlist=[ax1,ax2,ax3,ax4,ax5,ax6]
ax2list=[ax11,ax22,ax33,ax44,ax55,ax66]

for ax in ax2list:
    ax.plot([0, 500],[0.008/0.27,0.008/0.27], c='c')
    ax.fill_between([0, 500],[0.008/0.27-0.04/3,0.008/0.27-0.04/3],[0.008/0.27+0.04/3,0.008/0.27+0.04/3],facecolors="none",alpha=0.3,edgecolor='c',hatch='////')
    ax.plot([0, 500],[0.047,0.047], c='tomato')
    ax.fill_between([0, 500],[0.047-0.005,0.047-0.005],[0.047+0.005,0.047+0.005],facecolors='none',edgecolor='tomato',alpha=0.3,hatch='\\\\\\\\')
for ax in axlist:
    ax.plot([0, 500],[0.33,0.33], c='c')
    ax.plot([0, 500],[0.27,0.27], c='tomato')

fesc_rel(ax66,ax6, ax666,read_new_01Zsun, 150, 480, Fesc_new8,'G9_Zlow',True)
fesc_rel(ax11,ax1, ax111,read_new_01Zsun_05pc, 150, 380, Fesc_new, 'G9_Zlow_HR',True)
fesc_rel(ax55,ax5, ax555,read_new_1Zsun, 150, 480, Fesc_new8,  'G9_Zhigh',True)
fesc_rel(ax22,ax2, ax222,read_new_1Zsun_highSN_new, 70, 380, Fesc_new8,  'G9_Zhigh_SN5',True)
fesc_rel(ax33,ax3, ax333,read_new_03Zsun_highSN, 70, 380, Fesc_new8,  'G9_Zmid_SN5',True)
fesc_rel(ax44,ax4, ax444,read_new_gasrich, 150, 300, Fesc_new8, 'G9_Zlow_gas5',True)

ax2.xaxis.set_major_formatter(plt.NullFormatter())
ax3.xaxis.set_major_formatter(plt.NullFormatter())
ax4.xaxis.set_major_formatter(plt.NullFormatter())
ax5.xaxis.set_major_formatter(plt.NullFormatter())
ax6.xaxis.set_major_formatter(plt.NullFormatter())
ax22.xaxis.set_major_formatter(plt.NullFormatter())
ax33.xaxis.set_major_formatter(plt.NullFormatter())
ax44.xaxis.set_major_formatter(plt.NullFormatter())
ax55.xaxis.set_major_formatter(plt.NullFormatter())
ax66.xaxis.set_major_formatter(plt.NullFormatter())
ax1.set_xlabel('time (Myr)',fontsize=23)
ax11.set_xlabel('time (Myr)',fontsize=23)
   # ax.set_xlabel('time (Myr)')
    #ax.set_ylabel('$f_{esc,rel}$')

#ax11.set_ylabel('$(F_{900}/F_{1500})_{ISM+CGM}$',fontsize=13)
#ax55.set_ylabel('$(F_{900}/F_{1500})_{ISM+CGM}$',fontsize=13)

#ax33.set_ylabel('$(F_{900}/F_{1500})_{ISM+CGM}$',fontsize=13)
#fig.text(0.45,0.04,'time (Myr)',fontsize=23)
fig.text(0.03,0.65,'$(F_{900}/F_{1500})_{ISM+CGM}$',fontsize=23,rotation=90)
fig.text(0.5,0.6,'$L_{900}/L_{1500}$',fontsize=23,rotation=90)


#ax1.set_ylabel('$L_{900}/L_{1500}$',fontsize=14)
#ax3.set_ylabel('$L_{900}/L_{1500}$',fontsize=14)
#ax5.set_ylabel('$L_{900}/L_{1500}$',fontsize=14)

print(0.008/0.27)
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig11_new3.pdf')















