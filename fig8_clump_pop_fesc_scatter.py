import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import os.path

import time
read_old_01Zsun = '/Volumes/THYoo/RHD_10pc/'

read_old_002Zsun = '/Volumes/THYoo/RHD_10pc_lowZ/'

read_new_01Zsun = '/Volumes/THYoo/kisti/RHD_10pc_0.1Zsun/'
read_new_01Zsun_highSN = '/Volumes/THYoo/kisti/0.1Zsun_SNen/'

read_new_1Zsun = '/Volumes/THYoo/kisti/RHD_10pc_1Zsun/'

read_new_01Zsun_re = '/blackwhale/dbahck37/kisti/0.1Zsun/'

read_new_1Zsun_re = '/blackwhale/dbahck37/kisti/1Zsun/'

read_new_gasrich = '/Volumes/THYoo/kisti/RHD_10pc_gasrich/G9_gasrich/'

read_new_1Zsun_highSN_old = '/Volumes/gdrive/1Zsun_SNen_old/'
read_new_1Zsun_highSN_new = '/Volumes/gdrive/1Zsun_SNen_new/'

read_new_03Zsun_highSN = '/Volumes/gdrive/0.3Zsun_SNen/'

read_new_01Zsun_05pc = '/Volumes/gdrive/0.1Zsun_5pc/'

class readSED():
    def __init__(self, dir, nout):
        dat = FortranFile(dir+'/ray_%05d.dat' % (nout), 'r')
        npart, nwave2 = dat.read_ints()
        wave = dat.read_reals(dtype=np.double)
        sed_intr = dat.read_reals(dtype=np.double)
        sed_attH = dat.read_reals(dtype=np.double)
        sed_attD = dat.read_reals(dtype=np.double)

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
        self.starid = np.abs(partdata.star.id[0])
        self.mp0 = partdata.star.mp0[0] * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l
        self.starage = (partdata.info.time-partdata.star.tp[0]) * 4.70430e14 / 365. /24./3600/1e6
        self.boxlen = partdata.info.boxlen
        youngindex = np.where(self.starage<10)
        self.SFR = np.sum(self.mp0[youngindex])/1e7

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
        wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attHHe = dat2.read_reals(dtype=np.double)
        sed_attHHeD = dat2.read_reals(dtype=np.double)
        sed_attHHI = dat2.read_reals(dtype=np.double)
        sed_attHH2 = dat2.read_reals(dtype=np.double)
        sed_attHHe = dat2.read_reals(dtype=np.double)
        sed_attD = dat2.read_reals(dtype=np.double)

        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        self.fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)
class Fesc_new8():
    def __init__(self, dir, nout):
        dat2 = FortranFile(dir + 'ray_nside8_laursen/ray_%05d.dat' % (nout), 'r')
        npart, nwave2, version = dat2.read_ints()
        wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attHHe = dat2.read_reals(dtype=np.double)
        sed_attHHeD = dat2.read_reals(dtype=np.double)
        sed_attHHI = dat2.read_reals(dtype=np.double)
        sed_attHH2 = dat2.read_reals(dtype=np.double)
        sed_attHHe= dat2.read_reals(dtype=np.double)
        sed_attD= dat2.read_reals(dtype=np.double)

        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        self.fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
        self.fesc2 = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)

        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)

def FindClumpHostStar(Part1, Clump1):

    indexarr = np.array([])
    for i in range(len(Clump1.xclump)):

        dd = np.sqrt((Part1.xp[0]-Clump1.xclump[i])**2+(Part1.xp[1]-Clump1.yclump[i])**2+(Part1.xp[2]-Clump1.zclump[i])**2)
        index = np.where(dd-Clump1.rclump[i]<0)[0]
        indexarr = np.append(indexarr, index.astype(int))

    noindexarr = np.delete(np.arange(len(Part1.xp[0])),indexarr)

    return indexarr, noindexarr

def std_weight(val, wei):
    av = np.average(val, weights=wei)
    var = np.average((val-av)**2,weights=wei)
    return np.sqrt(var)


def main(ax,ax2, read, inisnap, endsnap, Fesc, label, saveload, color):
    numsnap = endsnap - inisnap + 1

    if saveload==False:
        clumpfrac = np.array([])
        clumpphotonrfrac = np.array([])
        clumpfesc = np.array([])
        noclumpfesc = np.array([])
        timearr = np.array([])
        fesc =np.array([])
        photonr= np.array([])

        for n in range(numsnap):
            nout = n + inisnap

            if not os.path.isfile(read+'/SAVE/part_%05d.sav' % (nout)):
                print(read+'/SAVE/part_%05d.sav' % (nout))
                continue

            if Fesc==Fesc_new:
                if not os.path.isfile(read+'ray_nside4_laursen/ray_%05d.dat' % (nout)):
                    print(read+'ray_nside4_laursen/ray_%05d.dat' % (nout))
                    continue
            if Fesc==Fesc_new8:
                if not os.path.isfile(read + 'ray_nside8_laursen/ray_%05d.dat' % (nout)):
                    print(read + 'ray_nside8_laursen/ray_%05d.dat' % (nout))
                    continue
            if not os.path.isfile(read+'clump3/clump_%05d.txt'%nout):
                print(read + 'clump3/clump_%05d.txt'%nout)
                continue
            Part1 = Part(read, nout)
            """
            if n>0:

                if Part1.snaptime < prevsnap:
                    print('snaptime error')
                    continue
            """
            Fesc1 = Fesc(read, nout)
            Clump1 = Clump(read, nout, Part1)

            if len(Fesc1.fescD)!=len(Part1.xp[0]):
                print(nout, 'no fesc, star matching')
                continue
            print(nout, Part1.snaptime)

            clumpindex, noclumpindex = FindClumpHostStar(Part1, Clump1)
            clumpindex = np.unique(clumpindex.astype(int))
            noclumpindex = np.unique(noclumpindex.astype(int))

            clumpfrac = np.append(clumpfrac, len(clumpindex)/len(Part1.xp[0]))
            clumpphotonr = np.sum(Fesc1.photonr[clumpindex])
            clumpphotonrfrac = np.append(clumpphotonrfrac, clumpphotonr/np.sum(Fesc1.photonr))
            print(clumpphotonr/np.sum(Fesc1.photonr))

            clumpfesc = np.append(clumpfesc, np.sum(Fesc1.fescD[clumpindex]*Fesc1.photonr[clumpindex])/np.sum(Fesc1.photonr[clumpindex]))
            noclumpfesc = np.append(noclumpfesc, np.sum(Fesc1.fescD[noclumpindex]*Fesc1.photonr[noclumpindex])/np.sum(Fesc1.photonr[noclumpindex]))
            fesc = np.append(fesc, Fesc1.fesc)
            photonr = np.append(photonr, np.sum(Fesc1.photonr))
            timearr = np.append(timearr, Part1.snaptime)

            prevsnap = Part1.snaptime

        np.savetxt(read+'clumpfrac2.dat',clumpfrac,delimiter=' ',newline='\n')
        np.savetxt(read+'clumpphotonrfrac2.dat',clumpphotonrfrac,delimiter=' ',newline='\n')
        np.savetxt(read+'clumpfesc2.dat',clumpfesc,delimiter=' ',newline='\n')
        np.savetxt(read+'noclumpfesc2.dat',noclumpfesc,delimiter=' ',newline='\n')
        np.savetxt(read+'timearr_cf2.dat',timearr,delimiter=' ',newline='\n')
        np.savetxt(read+'fesc_scatter2.dat',fesc,delimiter=' ', newline='\n')
        np.savetxt(read+'photonr2.dat',photonr,delimiter=' ', newline='\n')


    else:
        clumpfrac = np.loadtxt(read+'clumpfrac2.dat')
        clumpphotonrfrac = np.loadtxt(read+'clumpphotonrfrac2.dat')
        clumpfesc = np.loadtxt(read+'clumpfesc2.dat')
        noclumpfesc = np.loadtxt(read+'noclumpfesc2.dat')
        timearr = np.loadtxt(read+'timearr_cf2.dat')
        fesc = np.loadtxt(read+'fesc_scatter2.dat')
        photonr = np.loadtxt(read+'photonr2.dat')

    print(np.mean(clumpfrac), np.mean(clumpphotonrfrac), np.mean(clumpfesc), np.mean(noclumpfesc), np.mean(fesc),(1-np.mean(clumpphotonrfrac))*np.mean(noclumpfesc) )

 #   ax.plot(timearr, clumpfrac, label=label, color=color)
    ax.scatter(clumpphotonrfrac, fesc, color=color,s=2,alpha=1)
    #ax2.scatter(clumpphotonrfrac, noclumpfesc, color=color, s=2, alpha=0.7,label=label)

    xbin = [np.min(clumpphotonrfrac), np.percentile(clumpphotonrfrac,20), np.percentile(clumpphotonrfrac,40), np.percentile(clumpphotonrfrac,60), np.percentile(clumpphotonrfrac,80),np.max(clumpphotonrfrac)]

    avfescarr = np.zeros(len(xbin)-1)
    upp = np.zeros(len(xbin)-1)
    low = np.zeros(len(xbin)-1)
    for j in range(len(xbin)-1):
        ind = np.where((clumpphotonrfrac>=xbin[j])&(clumpphotonrfrac<xbin[j+1]))
        avfescarr[j] = np.median(fesc[ind])
        upp[j] = np.abs(avfescarr[j]-np.percentile(fesc[ind],75))
        low[j] = np.abs(avfescarr[j]-np.percentile(fesc[ind],25))




    xxbin = [np.percentile(clumpphotonrfrac,10),np.percentile(clumpphotonrfrac,30),np.percentile(clumpphotonrfrac,50),np.percentile(clumpphotonrfrac,70),np.percentile(clumpphotonrfrac,90)]

    ax.errorbar(xxbin, avfescarr,yerr=(low,upp), marker='s', color='w',lw=6)

    ax.errorbar(xxbin, avfescarr, yerr=(low,upp), marker='s', color=color, lw=3,label=label)
    ax.set_ylabel('$f_{esc}^{3D}$',fontsize=35)

    ax2.hist(clumpphotonrfrac, bins=25, range=(0,1), density=True, histtype='stepfilled',edgecolor=color,facecolor="None")
 #   ax2.set_ylabel('$f_{esc, nonclump}$')
    ax.set_xlabel('$f^{\gamma}_{clump}$',fontsize=35)
    ax2.set_ylabel('pdf')
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
  #  ax2.set_xlabel('$f_{clump}$')



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Modern Computer'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] =25
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['legend.fontsize']=20
plt.rcParams['patch.linewidth']=2.5


"""
fig = plt.figure(figsize=(10,14))
ax1 = fig.add_axes([0.1,0.1,0.8,0.25])
ax2 = fig.add_axes([0.1,0.35,0.8,0.25])
ax3 = fig.add_axes([0.1,0.6,0.8,0.35])
ax1.set_ylabel('#$_{*,clump}$/#$_{*,tot}$')

main(read_new_01Zsun, 140, 330, Fesc_new8, 'olive', 'G9_01Zsun',False)
main(read_new_1Zsun, 140, 330, Fesc_new8, 'r', 'G9_1Zsun',False)
main(read_new_1Zsun_highSN_new, 46, 255, Fesc_new8, 'orange', 'G9_1Zsun_SNboost',False)
#main(read_new_gasrich, 120, 199, Fesc_new8, 'b', 'G9_gasrich')

ax1.legend()
ax1.set_xlabel('time (Myr)')
ax2.set_ylabel(r'$\dot N_{clump}/\dot N_{tot}$')
ax3.set_ylabel('$f_{esc}$')
ax2.set_xticks([])
ax3.set_xticks([])

"""
fig = plt.figure(figsize=(10, 10), dpi=72)

ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.6])
ax2 = fig.add_axes([0.15, 0.75, 0.8,0.2])

#main(ax1, read_new_1Zsun_highSN_new, 130, 380, Fesc_new8, 'G9_Zhigh_SN5',True,'lightseagreen')
main(ax1,ax2,read_new_01Zsun, 150, 480, Fesc_new8, 'G9_Zlow',True,'k')

main(ax1, ax2,read_new_1Zsun, 150, 480, Fesc_new8, 'G9_Zhigh',True,'r')
main(ax1, ax2,read_new_gasrich, 150, 285, Fesc_new8,  'G9_Zlow_gas5',True,'b')

main(ax1,ax2,read_new_03Zsun_highSN, 70, 380, Fesc_new8,  'G9_Zmid_SN5',True,'orange')
#main(ax1,read_new_01Zsun_highSN, 150, 436, Fesc_new8, 'G9_Zlow_SN5',False,'olive')

#main(ax3,ax33,read_new_1Zsun, 100, 480, Fesc_new8,  'G9_1Zsun',False)

#main(ax2,ax22,read_new_01Zsun_05pc, 30, 380, Fesc_new,  'G9_01Zsun_5pc',False)
ax1.set_ylim(0,0.2)
ax1.legend()
ax1.set_yticks([0,0.05,0.1,0.15])

#ax1.legend(frameon=False)
#ax1.set_xlabel('time (Myr)')
#ax1.set_ylabel(r'$\langle f_{esc} \rangle$')
#ax3.set_ylabel('$f_{esc}$',fontsize=20)
#ax44.set_ylabel(r'$\dot N_{clump}/\dot N_{tot}$',fontsize=20)
#plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig8_thesis.pdf')
plt.show()



