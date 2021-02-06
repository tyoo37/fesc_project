import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import os.path
import matplotlib
read_old_01Zsun = '/Volumes/THYoo/RHD_10pc/'

read_old_002Zsun = '/Volumes/THYoo/RHD_10pc_lowZ/'

read_new_01Zsun = '/Volumes/THYoo/kisti/RHD_10pc_0.1Zsun/'

read_new_1Zsun = '/Volumes/THYoo/kisti/RHD_10pc_1Zsun/'

read_new_gasrich= '/Volumes/THYoo/kisti/RHD_10pc_gasrich/G9_gasrich/'


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
        self.vx = celldata.cell[0][4][1] * Part.unit_l /4.70430e14
        self.vy = celldata.cell[0][4][2] * Part.unit_l /4.70430e14
        self.vz = celldata.cell[0][4][3] * Part.unit_l /4.70430e14
        self.den = celldata.cell[0][4][0] *Part.unit_d


class GasrichCell():
    def __init__(self, dir, nout, Part):
        self.dir = dir
        self.nout = nout
        celldata = readsav(self.dir + '/SAVE/cell_%05d.sav' % (self.nout))
        self.nH = celldata.cell.nh[0] * 30.996344
        self.x = celldata.cell.x[0] * Part.boxpc
        self.y = celldata.cell.y[0] * Part.boxpc
        self.z = celldata.cell.z[0] * Part.boxpc
        self.dx = celldata.cell.dx[0] * Part.boxpc
        self.mindx = np.min(celldata.cell.dx[0])
        self.m = celldata.cell.nh[0] *Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l *Part.unit_l *(celldata.cell.dx[0]*Part.boxlen)**3
        self.vx = celldata.cell.vx[0]* Part.unit_l /4.70430e14
        self.vy = celldata.cell.vy[0]* Part.unit_l /4.70430e14
        self.vz = celldata.cell.vz[0]* Part.unit_l /4.70430e14
        self.den = celldata.cell[0][4][0] *Part.unit_d
class Cellfromdat():
    def __init__(self, dir, nout, Part):
        celldata = FortranFile(dir + 'dat/cell_%05d.dat' % nout, 'r')
        nlines, xx, nvarh = celldata.read_ints(dtype=np.int32)

        xc = celldata.read_reals(dtype=np.double)
        yc = celldata.read_reals(dtype=np.double)
        zc = celldata.read_reals(dtype=np.double)
        dxc = celldata.read_reals(dtype=np.double)

        self.x = xc * Part.boxpc
        self.y = yc * Part.boxpc
        self.z = zc * Part.boxpc
        self.dx = dxc * Part.boxpc

        var = np.zeros((nlines, nvarh))
        for i in range(nvarh):
            var[:, i] = celldata.read_reals(dtype=np.double)

        self.nH = var[:, 0] * 30.996344
        self.den = var[:,0] * Part.unit_d
        self.m = var[:, 0] * Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l * Part.unit_l * (
                dxc * Part.boxlen) ** 3
        xHI = var[:, 7]
        xHII = var[:, 8]
        self.mHIH2 = self.m * (1 - xHII)
        self.vx = var[:,1] * Part.unit_l /4.70430e14
        self.vy = var[:,2] * Part.unit_l /4.70430e14
        self.vz = var[:,3] * Part.unit_l /4.70430e14

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

class Fesc_new8_dist():
    def __init__(self,dir,nout):
        dat2 = FortranFile(dir + 'ray_nside8_dist/ray_%05d.dat' % (nout), 'r')
        self.npart, nwave2, version = dat2.read_ints()
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
        self.fescD = self.fescD.reshape(8,self.npart)
        self.fescH = self.fescH.reshape(8,self.npart)
nkpc=8
ls = ['-','--',':','-.','-','--',':','-.']
def test(ax2, Fesc,read, inisnap, endsnap, label, color,load):
    numsnap = endsnap - inisnap + 1
    if load==False:
        fescarr = np.zeros(nkpc+1)
        timearr = []
        #fescarr2 = np.array([])
        for i in range(numsnap):
            nout = i + inisnap

            if not os.path.isfile(read + '/SAVE/cell_%05d.sav'%nout):
                print(read + '/SAVE/cell_%05d.sav'%nout)
                continue
            if not os.path.isfile(read + '/ray_nside8_dist/ray_%05d.dat'%nout):
                print(read + '/ray_nside8_dist/ray_%05d.dat'%nout)
                continue
            print(nout)
            Part1 = Part(read, nout)
            Fesc1 = Fesc(read,nout)
            Fesc_dist1 = Fesc_new8_dist(read, nout)



            npart = Fesc_dist1.npart
            fescD= Fesc_dist1.fescD

            photonr = Fesc_dist1.photonr
            print(fescD.T)
            print(photonr)
            for j in range(nkpc):
                fescarr[j] = np.sum(fescD[j,:]*photonr)/np.sum(photonr)
            fescarr[-1]=Fesc1.fesc
            #print(fescarr)

            if i==0:
                fescarr2 = fescarr
            else:
                fescarr2 = np.vstack((fescarr2, fescarr))

            timearr.append(Part1.snaptime)



        timearr = np.array(timearr)
     #   f
        np.savetxt(read+'timearr_fescdist.dat',timearr,newline='\n',delimiter=' ')
        np.savetxt(read+'fescarr2_fescdist.dat',fescarr2,newline='\n',delimiter=' ')
        #np.savetxt(read+'fescarr3_fescdist.dat',fescarr3,newline='\n',delimiter=' ')


    else:
        timearr = np.loadtxt(read+'timearr_fescdist.dat')
        fescarr2 = np.loadtxt(read+'fescarr2_fescdist.dat')
        #fescarr3 = np.loadtxt(read+'fescarr3_fescdist.dat')
    fescarr3 = np.zeros(nkpc+1)
    for n in range(nkpc+1):

        #ax1.plot(timearr, fescarr2[:, n], ls=ls[n], color=color)
        fescarr3[n] = np.mean(fescarr2[:,n])
    rkpc = np.logspace(np.log10(0.04),np.log10(8),8)
    rkpc = np.append(rkpc,89)
    ax2.plot(rkpc, fescarr3, color=color, label=label,lw=3,marker='s',markersize=10)

    #ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    #ax1.set_ylim(0.001,5)

    #ax1.text(160, 2, label)
    #ax1.set_xlim(150,300)
    print(fescarr3/fescarr3[0])
    return fescarr3
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] =24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20
fig = plt.figure(figsize=(10,8))
#ax1 = fig.add_axes([0.14,0.5,0.38,0.2])
#ax2 = fig.add_axes([0.56,0.5,0.38,0.2])
#ax3 = fig.add_axes([0.14,0.74,0.38,0.2])
#ax4 = fig.add_axes([0.56,0.74,0.38,0.2])
ax5 = fig.add_axes([0.15,0.15,0.8,0.8])

fesc1 = test( ax5, Fesc_new8, read_new_01Zsun, 150,300, 'G9_Zlow', 'k',True)
fesc3=test(ax5, Fesc_new8, read_new_1Zsun,151,300, 'G9_Zhigh', 'firebrick',True)
fesc2 = test( ax5, Fesc_new8,read_new_gasrich, 150,300, 'G9_Zlow_gas5', 'dodgerblue',True)
fesc4=test( ax5, Fesc_new8,read_new_1Zsun_highSN_new, 64,214, 'G9_Zhigh_SN5', 'magenta',True)

#ax2.set_xscale('log')
#ax1.set_xlabel('time (Myr)')
#ax2.set_xlabel('time (Myr)')

#ax1.set_ylabel(r'$f_{esc}^{3D}(r)$')
#ax3.set_ylabel(r'$f_{esc}^{3D}(r)$')

ax5.set_xlabel('$d_{star}$ (kpc)')
ax5.set_ylabel(r'$f_{esc}^{3D}(r)$')

ax5.legend()
ax5.set_ylim(0.007,1)


#ax3.xaxis.set_major_formatter(plt.NullFormatter())
#ax4.xaxis.set_major_formatter(plt.NullFormatter())



#ax2.yaxis.set_major_formatter(plt.NullFormatter())
#ax4.yaxis.set_major_formatter(plt.NullFormatter())
#ax3.xaxis.set_major_formatter(plt.NullFormatter())
#ax4.xaxis.set_major_formatter(plt.NullFormatter())
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig9_new2.pdf')
#plt.show()