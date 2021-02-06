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
def getr(x,y,z,xcenter,ycenter,zcenter):
    return np.sqrt((x-xcenter)**2+(y-ycenter)**2+(z-zcenter)**2)
def get2dr(x,y,xcen,ycen):
    return np.sqrt((x-xcen)**2+(y-ycen)**2)
def getmass(marr, rarr, r):
    ind = np.where(rarr<r)
    return np.sum(marr[ind])
def mxsum(marr, xarr,ind):
    return np.sum(marr[ind]*xarr[ind])
def msum(marr,ind):
    return np.sum(marr[ind])
def simpleCoM(x,y,z,marr,rarr,r):
    ind = np.where(rarr<r)
    xx = np.sum(x[ind]*marr[ind])/np.sum(marr[ind])
    yy = np.sum(y[ind]*marr[ind])/np.sum(marr[ind])
    zz = np.sum(z[ind]*marr[ind])/np.sum(marr[ind])

    return xx,yy,zz

#half-mass CoM

def CoM_pre(Part1, Cell1,rgrid,totmass, xcen, ycen, zcen, gasonly):
    rstar = getr(Part1.xp[0],Part1.xp[1],Part1.xp[2],xcen, ycen, zcen)
    rpart = getr(Part1.dmxp[0],Part1.dmxp[1],Part1.dmxp[2],xcen, ycen, zcen)
    rcell = getr(Cell1.x,Cell1.y,Cell1.z,xcen, ycen, zcen)


    for i in range(len(rgrid)):
        mstar = getmass(Part1.mp0, rstar, rgrid[i])
        mpart = getmass(Part1.dmm, rpart, rgrid[i])
        mcell = getmass(Cell1.m, rcell, rgrid[i])
        summass = mstar + mpart + mcell
        if summass > totmass/2:
            rrr = rgrid[i]
            break
        if i == len(rgrid)-1:
            rrr = rgrid[-1]

    if gasonly==False:

        indstar = np.where(rstar < rrr)
        indpart = np.where(rpart < rrr)
        indcell = np.where(rcell < rrr)

        totalmx = mxsum(Part1.xp[0], Part1.mp0, indstar) + mxsum(Part1.dmxp[0], Part1.dmm, indpart) + mxsum(Cell1.x,
                                                                                                            Cell1.m,
                                                                                                            indcell)
        totalmy = mxsum(Part1.xp[1], Part1.mp0, indstar) + mxsum(Part1.dmxp[1], Part1.dmm, indpart) + mxsum(Cell1.y,
                                                                                                            Cell1.m,
                                                                                                            indcell)
        totalmz = mxsum(Part1.xp[2], Part1.mp0, indstar) + mxsum(Part1.dmxp[2], Part1.dmm, indpart) + mxsum(Cell1.z,
                                                                                                            Cell1.m,
                                                                                                            indcell)
        totalm = msum(Part1.mp0, indstar) + msum(Part1.dmm, indpart) + msum(Cell1.m, indcell)

    else:
        indcell = np.where(rcell < rrr)

        totalmx = mxsum(Cell1.x, Cell1.m,indcell);totalmy = mxsum(Cell1.y, Cell1.m,indcell);totalmz = mxsum(Cell1.z, Cell1.m,indcell)

        totalm=msum(Cell1.m, indcell)
    xx = totalmx/totalm
    yy = totalmy/totalm
    zz = totalmz/totalm

    return xx, yy, zz

def CoM_main(Part1,Cell1,diskmass):

    rgrid1=np.linspace(100, 4000, num=40)
    boxcen = Part1.boxpc/2
    x1, y1, z1 = CoM_pre(Part1,Cell1,rgrid1,1e11,boxcen,boxcen,boxcen,False)
    x2, y2, z2 = CoM_pre(Part1,Cell1,rgrid1,diskmass,x1,y1,z1,True)
    x3, y3, z3 = CoM_pre(Part1,Cell1,rgrid1,diskmass,x2,y2,z2,True)

    #print(x2,y2,z2)
    return x3, y3, z3
def gashmr(Cell1,xcen,ycen,zcen,zrange,rbin,diskmass):
    for r in range(len(rbin)):
        ind = np.where((get2dr(Cell1.x+Cell1.dx/2, Cell1.y+Cell1.dx/2, xcen,ycen)<rbin[r])&(np.abs(Cell1.z-zcen)<zrange))

        if np.sum(Cell1.m[ind])>diskmass/2:
            gashmr=rbin[r]
            break
    return gashmr

def test(ax1, ax2, read, inisnap, endsnap, Cell, label, color,load, diskmass):
    numsnap = endsnap - inisnap + 1
    if load==False:
        fescarr = np.zeros(nkpc)
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
            Cell1 = Cell(read, nout, Part1)
            Fesc_dist1 = Fesc_new8_dist(read, nout)
            xcen, ycen, zcen = CoM_main(Part1, Cell1, diskmass)
            gashmr1 =gashmr(Cell1,xcen,ycen,zcen,7000,np.linspace(100,7000,num=70),diskmass)
            ind = np.where((get2dr(Part1.xp[0],Part1.xp[1],xcen,ycen)<gashmr1)&(Part1.xp[2]-zcen<500))

            fescD= Fesc_dist1.fescD

            photonr = Fesc_dist1.photonr
            print(fescD.T)
            print(photonr)
            for j in range(nkpc):
                fescarr[j] = np.sum(fescD[j,ind]*photonr[ind])/np.sum(photonr[ind])

            #print(fescarr)

            if i==0:
                fescarr2 = fescarr
            else:
                fescarr2 = np.vstack((fescarr2, fescarr))

            timearr.append(Part1.snaptime)



        timearr = np.array(timearr)
        fescarr3 = np.zeros(nkpc)
        np.savetxt(read+'timearr_fescdist.dat',timearr,newline='\n',delimiter=' ')
        np.savetxt(read+'fescarr2_fescdist.dat',fescarr2,newline='\n',delimiter=' ')
        np.savetxt(read+'fescarr3_fescdist.dat',fescarr3,newline='\n',delimiter=' ')

    else:
        timearr = np.loadtxt(read+'timearr_fescdist.dat')
        fescarr2 = np.loadtxt(read+'fescarr2_fescdist.dat')
        fescarr3 = np.loadtxt(read+'fescarr3_fescdist.dat')

    for n in range(nkpc):

        ax1.plot(timearr, fescarr2[:, n], ls=ls[n], color=color)



        fescarr3[n] = np.mean(fescarr2[:,n])
    rkpc = np.logspace(np.log10(0.04),np.log10(8),8)
    ax2.plot(rkpc, fescarr3, color=color, label=label,lw=1.5,marker='s')

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax1.set_ylim(0.001,5)

    ax1.text(160, 2, label)
    ax1.set_xlim(150,300)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] =15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 20
fig = plt.figure(figsize=(10,14))
ax1 = fig.add_axes([0.14,0.5,0.38,0.2])
ax2 = fig.add_axes([0.56,0.5,0.38,0.2])
ax3 = fig.add_axes([0.14,0.74,0.38,0.2])
ax4 = fig.add_axes([0.56,0.74,0.38,0.2])
ax5 = fig.add_axes([0.14,0.1,0.8,0.3])
test(ax1, ax5, read_new_1Zsun, 151,300,Cell, 'G9_Zhigh', 'lightseagreen',True,1.75e09)

test(ax3, ax5, read_new_01Zsun, 150,300,Cell, 'G9_Zlow', 'gray',True,1.75e09)
test(ax4, ax5, read_new_gasrich, 150,300,Cellfromdat, 'G9_Zlow_gas5', 'b',True,1.75e09*5)
test(ax2, ax5, read_new_1Zsun_highSN_new, 64,214, Cell,'G9_Zhigh_SN5', 'r',True,1.75e09)

#ax2.set_xscale('log')
ax1.set_xlabel('time (Myr)')
ax2.set_xlabel('time (Myr)')

ax1.set_ylabel(r'$f_{esc}^{3D}(r)$')
ax3.set_ylabel(r'$f_{esc}^{3D}(r)$')

ax5.set_xlabel('$d_{star}$ (kpc)')
ax5.set_ylabel(r'$f_{esc}^{3D}(r)$')

ax5.legend()


ax3.xaxis.set_major_formatter(plt.NullFormatter())
ax4.xaxis.set_major_formatter(plt.NullFormatter())



ax2.yaxis.set_major_formatter(plt.NullFormatter())
ax4.yaxis.set_major_formatter(plt.NullFormatter())
ax3.xaxis.set_major_formatter(plt.NullFormatter())
ax4.xaxis.set_major_formatter(plt.NullFormatter())
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig9_new2.pdf')
plt.show()