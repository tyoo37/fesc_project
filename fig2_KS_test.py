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
read_new_01Zsun_highSN = '/Volumes/THYoo/kisti/0.1Zsun_SNen/'

read_new_1Zsun = '/Volumes/THYoo/kisti/RHD_10pc_1Zsun/'

read_new_gasrich= '/Volumes/THYoo/kisti/RHD_10pc_gasrich/G9_gasrich/'

read_new_1Zsun_highSN_new = '/Volumes/gdrive/1Zsun_SNen_new/'

read_new_03Zsun_highSN = '/Volumes/gdrive/0.3Zsun_SNen/'

read_new_01Zsun_05pc = '/Volumes/gdrive/0.1Zsun_5pc/'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Modern Computer'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] =20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
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
        self.mHIH2 = self.m * (1-xHII) *0.76

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

def CoM_main(Part1,Cell1):

    rgrid1=np.linspace(100, 4000, num=40)
    boxcen = Part1.boxpc/2
    x1, y1, z1 = CoM_pre(Part1,Cell1,rgrid1,1e11,boxcen,boxcen,boxcen,False)
    #x2, y2, z2 = CoM_pre(Part1,Cell1,rgrid1,1.75e9,x1,y1,z1,True)
    #print(x2,y2,z2)
    return x1, y1, z1




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

class GenerateArray():

    def __init__(self, Part, Cell, xcen,ycen,zcen, hawidplot, depth):


        pc = 3.08e18

        mindx = np.min(Cell.dx)

        maxgrid = int(np.log2(np.max(Cell.dx) / mindx))

        xind = Cell.x / mindx - 0.5
        yind = Cell.y / mindx - 0.5
        zind = Cell.z / mindx - 0.5

        #center = int(Part.boxpc / 2 / mindx)

        self.hawidplot =hawidplot

        xwid = self.hawidplot
        ywid = self.hawidplot
        zwid = depth

        self.xfwd = 2 * int(xwid)
        self.yfwd = 2 * int(ywid)
        self.zfwd = 2 * int(zwid)

        xcenter = int(xcen/mindx)
        ycenter = int(ycen/mindx)
        zcenter = int(zcen/mindx)

        xini = xcenter - xwid + 1
        yini = ycenter - ywid + 1
        zini = zcenter - zwid + 1
        xfin = xcenter + xwid
        yfin = ycenter + ywid
        zfin = zcenter + zwid
        # print(max(celldata.cell[0][4][0]))

        self.leafcell = np.zeros((self.xfwd, self.yfwd, self.zfwd))

        ind_1 = np.where((Cell.dx == mindx) & (xind >= xini) & (xind <= xfin)
                         & (yind >= yini) & (yind <= yfin) & (zind >= zini) & (zind <= zfin))[0]

        self.leafcell[xind[ind_1].astype(int) - int(xini), yind[ind_1].astype(int) - int(yini), zind[ind_1].astype(int) - int(zini)] = ind_1.astype(int)
        print(np.max(xind[ind_1] - xini), np.max(yind[ind_1] - yini), np.max(zind[ind_1] - zini),
              np.min(xind[ind_1] - xini), np.min(yind[ind_1] - yini), np.min(zind[ind_1] - zini))
        print('leaf cells are allocated (n=%d)' % len(ind_1))

        mul1 = np.arange(2 ** (maxgrid - 1)) + 0.5
        mul2 = np.arange(2 ** (maxgrid - 1)) * (-1) - 0.5
        mul = np.zeros(2 ** maxgrid)
        for k in range(2 ** (maxgrid - 1)):
            mul[2 * k] = mul1[k]
            mul[2 * k + 1] = mul2[k]
        nn = 0

        for n in range(maxgrid):
            nnn = 0
            ind = np.where(
                (Cell.dx == mindx * 2 ** (n + 1)) & (xind + Cell.dx / 2 / mindx >= xini) & (xind - Cell.dx / 2 / mindx <= xfin) & (
                        yind + Cell.dx / 2 / mindx >= yini) & (yind - Cell.dx / 2 / mindx <= yfin) & (
                        zind + Cell.dx / 2 / mindx >= zini) & (zind - Cell.dx / 2 / mindx <= zfin))[0]
            print(len(ind), len(ind) * (2 ** (n + 1)) ** 3)
            for a in range(2 ** (n + 1)):
                for b in range(2 ** (n + 1)):
                    for c in range(2 ** (n + 1)):
                        xx = xind[ind] - xini + mul[a]
                        yy = yind[ind] - yini + mul[b]
                        zz = zind[ind] - zini + mul[c]
                        xyzind = np.where(
                            (xx >= 0) & (xx <= self.xfwd - 1) & (yy >= 0) & (yy <= self.yfwd - 1) & (zz >= 0) & (zz <= self.zfwd - 1))[0]
                        self.leafcell[xx[xyzind].astype(int), yy[xyzind].astype(int), zz[xyzind].astype(int)] = ind[xyzind]
                        ##    print('zerocell')
                        nnn = nnn + len(xyzind)
            nn = nn + nnn
            print('level %d grids are allocated(n = %d)' % (n + 2, nnn))
            if nnn == 0:
                break
        nonzero = len(np.where(self.leafcell != 0)[0])
        print('total allocated cells are = ', len(ind_1) + nn)
        print('total box cells are = ', self.xfwd * self.yfwd * self.zfwd)
        print('total non zero cell in the box are = ', nonzero)

        if len(ind_1) + nn != self.xfwd * self.yfwd * self.zfwd:
            raise ValueError("allocation failure")
        else:
            print('no error in allocation')
        self.mindx = mindx
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.zcenter = zcenter

        self.xwid = xwid
        self.ywid = ywid
        self.zwid = zwid



    def projectionPlot(self, Cell, ax, cm, direction, field, vmin, vmax):
        #XY projection
        if field == 'nH':
            var = Cell.nH

            #XY projection
            if direction == 'xy':
                self.plane = np.log10(np.sum(var[self.leafcell[:,:,:].astype(int)],axis=2)/self.zfwd)

            if direction == 'yz':
                self.plane = np.log10(np.sum(var[self.leafcell[:,:,:].astype(int)],axis=0)/self.xfwd)

            if direction == 'zx':
                self.plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=1) / self.yfwd)

        cax = ax.imshow(self.plane, origin='lower', cmap = cm, extent=[-self.hawidplot*9.1/1000, self.hawidplot*9.1/1000, -self.hawidplot*9.1/1000, self.hawidplot*9.1/1000], vmin=vmin, vmax=vmax)
        return cax

    def star_plot(self,Part, ax):
        print('star plotting...')
        sxind = Part.xp[0] / self.mindx - self.xcenter + self.xwid
        syind = Part.xp[1] / self.mindx - self.ycenter + self.ywid
        szind = Part.xp[2] / self.mindx - self.zcenter + self.zwid
        sind = np.where(
            (sxind >= 3) & (sxind < self.xfwd - 3) & (syind >= 3) & (syind < self.yfwd - 3) & (szind >= 3) & (
                        szind < self.zfwd - 3))[
            0]
        # sind = np.where((sxind >= 0) & (sxind < xfwd) & (syind >= 0) & (syind < yfwd) & (szind >= 0) & (
        # szind < zfwd))[0]
        sxind = sxind[sind]
        syind = syind[sind]
        szind = szind[sind]

        sxplot = (sxind - self.xwid) * self.mindx
        syplot = (syind - self.ywid) * self.mindx
        szplot = (szind - self.zwid) * self.mindx

        cax = ax.scatter(sxplot / 1000, syplot / 1000, c='grey', s=0.4, alpha=0.2)
        return cax

def CoM_check_plot(Part1, Cell1, wid, depth, xcen,ycen,zcen):
    fig = plt.figure(figsize=(8, 8),dpi=144)

    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    cm1 = plt.get_cmap('rainbow')

    a = GenerateArray(Part1, Cell1, xcen,ycen,zcen, wid, depth)
    ss1 = a.projectionPlot(Cell1, ax1, cm1, 'xy', 'nH', -3, 2)
    a.star_plot(Part1, ax1)
    ax1.scatter((xcen - a.mindx * a.xcenter) / 1000, (ycen - a.mindx * a.ycenter) / 1000, s=100, marker='*')

    ax1.set_xlabel('X(kpc)')
    ax1.set_ylabel('Y(kpc)')
    cax1 = fig.add_axes([0.9, 0.1, 0.02, 0.3])

    plt.colorbar(ss1, cax=cax1, cmap=cm1)
    plt.show()
    plt.close()

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
def getmass_zlim(marr, rarr, r,z,zcen,zlim):
    ind = np.where((rarr < r)&(np.abs(z-zcen)<zlim))
    return np.sum(marr[ind])

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
    for i in range(10):
        x1, y1, z1 = CoM_pre(Part1,Cell1,rgrid1,diskmass,x1,y1,z1,True)
    #print(x2,y2,z2)
    return x1, y1, z1

def halfmassrad_KS(Part,Cell,xcen,ycen,zcen,rbin, zrange):
    totstarmass = np.sum(Part.mp0)
    cellrr = get2dr(Cell.x,Cell.y, xcen,ycen)
    cellzdist = np.abs(Cell.z-zcen)
    partrr = get2dr(Part.xp[0],Part.xp[1],xcen,ycen)
    partzdist = np.abs(Part.xp[2]-zcen)
    for i in range(len(rbin)):
        #cellind = np.where((cellrr>=rbin[i])&(cellrr<rbin[i+1])&(cellzdist<zrange))
        partind = np.where((partrr<rbin[i])&(partzdist<zrange))
        if np.sum(Part.mp0[partind])>np.sum(Part.mp0)/2:
            halfmassrad = rbin[i]
            break
    print('hmr',halfmassrad)
    halfmassind_cell = np.where((cellrr<halfmassrad)&(cellzdist<zrange))
    halfmassind_part = np.where((partrr<halfmassrad)&(partzdist<zrange)&(Part.starage<10))


    area = np.pi * (halfmassrad)**2
    sd = np.sum(Cell.mHIH2[halfmassind_cell])/area
    sfrd = np.sum(Part.mp0[halfmassind_part])/area/10


    return sd, sfrd,halfmassrad


def CoM_Main(Part1):
    xcen = np.zeros(11)
    ycen = np.zeros(11)
    zcen = np.zeros(11)
    hmr = np.zeros(11)
    rgrid = np.linspace(100,4000,num=40)
    xcen[0] = np.sum(Part1.mp0*Part1.xp[0])/np.sum(Part1.mp0)
    ycen[0] = np.sum(Part1.mp0*Part1.xp[1])/np.sum(Part1.mp0)
    zcen[0] = np.sum(Part1.mp0*Part1.xp[2])/np.sum(Part1.mp0)
    for j in range(len(rgrid)):
        mass = getmass_zlim(Part1.mp0,get2dr(Part1.xp[0],Part1.xp[1],xcen[0],ycen[0]),rgrid[j],Part1.xp[2],zcen[0],2000)
        if mass>np.sum(Part1.mp0)/2:
            hmr[0]=rgrid[j]
            break
    print(hmr)

    for i in range(10):

        ind = np.where((get2dr(Part1.xp[0],Part1.xp[1],xcen[i],ycen[i])<hmr[i])&(np.abs(Part1.xp[2]-zcen[i])<2000))
        xcen[i+1]=np.sum(Part1.xp[0][ind]*Part1.mp0[ind])/np.sum(Part1.mp0[ind])
        ycen[i+1]=np.sum(Part1.xp[1][ind]*Part1.mp0[ind])/np.sum(Part1.mp0[ind])
        zcen[i+1]=np.sum(Part1.xp[2][ind]*Part1.mp0[ind])/np.sum(Part1.mp0[ind])
        for j in range(len(rgrid)):
            mass = getmass_zlim(Part1.mp0, get2dr(Part1.xp[0], Part1.xp[1], xcen[i + 1], ycen[i + 1]), rgrid[j],
                                Part1.xp[2],
                                zcen[i + 1], 2000)
            if mass > np.sum(Part1.mp0) / 2:
                hmr[i + 1] = rgrid[j]
                break

    print(xcen,ycen,zcen)



    return xcen, ycen, zcen


def fits_to_arr(colnum, filename):
    hdul = fits.open(filename)

    data = hdul[1].data
    arr = np.column_stack((data.field(0),data.field(1)))
    for i in range(colnum-2):
        arr = np.column_stack((arr,data.field(i+2)))
    return arr
def bigiel08_test():
    arr = fits_to_arr(8,'/Users/taehwau/projects/KS/bigiel10.fit')
    HI = 10**arr[:,2].astype(float)
    logHIerr = arr[:,3].astype(float)
    H2 = 10**arr[:,4].astype(float)
    logH2err = arr[:,5].astype(float)
    SFR = 10**arr[:,6].astype(float)
    fig2 = plt.figure()
    logbin = np.logspace(0,3,60)
    plt.hist(HI+H2, bins=logbin)
    plt.xscale('log')
    plt.show()

#bigiel08_test()
def scatter(ax,filename,colnum,color,sfrind,Hind,marker,label):
    arr = fits_to_arr(colnum,filename)
    SFR = 10**arr[:,sfrind].astype(float)
    H = 10**arr[:,Hind].astype(float)

    ax.scatter(H, SFR, color=color,marker=marker,label=label,alpha=0.5)
def scatter2(ax,filename,colnum,color1,color2,sfrind,Hind,marker1,marker2,label1,label2):
    arr = fits_to_arr(colnum, filename)
    SFR = 10 ** arr[:, sfrind].astype(float)
    H = 10 ** arr[:, Hind].astype(float)
    print(H)
    z = arr[:,0].astype(float)
    ind1 = np.where((z>0.)&(z<2))
    ind2 = np.where(z>=2)
    ax.scatter(H[ind1], SFR[ind1], edgecolors=color1,facecolors='none', marker=marker1,label=label1,alpha=0.5)
    ax.scatter(H[ind2], SFR[ind2], edgecolors=color2,facecolors='none', marker=marker2,label=label2,alpha=0.5)

def hist2d(ax, filename1,filename2,colnum1,colnum2,cmap,label):

    arr = fits_to_arr(colnum1,filename1)
    arr2 = fits_to_arr(colnum2,filename2)

    HI = 10**arr[:,2].astype(float)
    logHIerr = arr[:,3].astype(float)
    H2 = 10**arr[:,4].astype(float)
    logH2err = arr[:,5].astype(float)
    SFR = 10**arr[:,6].astype(float)
    logSFRerr = arr[:,7].astype(float)
    index = np.where((HI!=1.)&(H2!=1.)&(SFR!=1.))
    HI = HI[index]
    H2 = H2[index]
    SFR = SFR[index]

    HIH2 = HI+H2


    HIH22 = 10**arr2[:,2].astype(float)
    SFR2 = arr2[:,4].astype(float)/1e5

    index2 = np.where((HIH22!=1.)&(SFR2!=1.))
    HIH2 = np.append(HIH2, HIH22[index2])
    SFR = np.append(SFR,SFR2[index2])

    #print(np.min(HIH2),np.max(HIH2),np.min(SFR),np.max(SFR))
    xedges = np.logspace(-1,3,80)
    yedges = np.logspace(-5,0,125)
    H, xedges, yedges = np.histogram2d(HIH2, SFR,bins=(xedges, yedges))
    #print(H)
    #print(np.max(H))

    ax.pcolormesh(xedges,yedges,H.T,cmap=cmap,label=label,vmin=0,vmax=15)

def sigma3(f):
    sort = np.sort(f.flatten())[::-1]

    for i in range(len(f.flatten())):
        if np.sum(sort[:i])>0.682*np.sum(sort):
            sigma1 = sort[i]
            break
    for i in range(len(f.flatten())):
        if np.sum(sort[:i]) > 0.866 * np.sum(sort):
            sigma2 = sort[i]
            break

    for i in range(len(f.flatten())):
        if np.sum(sort[:i]) > 0.954 * np.sum(sort):
            sigma3 = sort[i]
            break
    #sigma3 = np.sort(f.flatten())[::-1][int(0.997*len(f.flatten()))]
    print('sigma',sigma1, sigma2, sigma3)
    print(int(0.682*len(f.flatten())),int(0.866*len(f.flatten())),int(0.954*len(f.flatten())))
    print(len(f.flatten()))
    return sigma3, sigma2, sigma1

def gauss_cont(ax, filename1,filename2,colnum1,colnum2):

    arr = fits_to_arr(colnum1,filename1)
    arr2 = fits_to_arr(colnum2,filename2)

    HI = 10**arr[:,2].astype(float)
    logHIerr = arr[:,3].astype(float)
    H2 = 10**arr[:,4].astype(float)
    logH2err = arr[:,5].astype(float)
    SFR = 10**arr[:,6].astype(float)
    logSFRerr = arr[:,7].astype(float)
    index = np.where((HI!=1.)&(H2!=1.)&(SFR!=1.))
    HI = HI[index]
    H2 = H2[index]
    SFR = SFR[index]

    HIH2 = HI+H2


    HIH22 = 10**arr2[:,2].astype(float)
    SFR2 = arr2[:,4].astype(float)/1e5

    index2 = np.where((HIH22!=1.)&(SFR2!=1.))
    #HIH2 = np.append(HIH2, HIH22[index2])
    #SFR = np.append(SFR,SFR2[index2])

    print(np.min(HIH2),np.max(HIH2),np.min(SFR),np.max(SFR))

    xx, yy = np.mgrid[np.min(np.log10(HIH2))-0.5:np.max(np.log10(HIH2))+0.5:200j, np.min(np.log10(SFR))-0.5:np.max(np.log10(SFR))+0.5:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([np.log10(HIH2), np.log10(SFR)])
    #ax.scatter(HIH2,SFR,s=1)
    #ax.scatter(HIH22,SFR2,s=1)

    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    sigma_1, sigma_2, sigma_3 = sigma3(f)

    ss1 = ax.contour(np.power(10,xx),np.power(10,yy),f,levels=[sigma_1,sigma_2,sigma_3],colors='grey',linewidths=2,origin='lower',alpha=0.4)
    #ss1 = ax.contour(xx,yy,f,colors='darkgoldenrod',linewidths=0.5)

    xx, yy = np.mgrid[np.min(np.log10(HIH22))-0.5:np.max(np.log10(HIH22))+0.5:200j, np.min(np.log10(SFR2))-0.5:np.max(np.log10(SFR2))+0.5:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([np.log10(HIH22), np.log10(SFR2)])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    #print('xx',xx)
    #print(f)
    sigma_1, sigma_2, sigma_3 = sigma3(f)
    #ax.pcolormesh(xx,yy,f)

    ss2 = ax.contour(np.power(10,xx), np.power(10,yy), f, levels=[sigma_1, sigma_2, sigma_3],colors='grey',linewidths=2,origin='lower',alpha=0.4)
    #ss2 = ax.contour(xx, yy, f ,colors='lime',linewidths=0.5)

    labels = ['Bigiel+10']
    ss1.collections[0].set_label(labels[0])
    #ss1.collections[-1].set_label(labels[1])
   # ss2.collections[0].set_label(labels[1])
    #ss2.collections[-1].set_label(labels[3])



def KS(ax,read, Part1, Cell1, saveload,label,color,diskmass):

    if saveload==False:
        sfrdarr= np.array([])

        sdarr= np.array([])
        diskgasmass = 1.75e9 # solarmass
        #rgrid = np.linspace(0, 5000, num=11)
        rgrid = np.linspace(0,2000,num=9)
        rgrid2 = np.linspace(0,5000,num=100)
        #xcen, ycen, zcen = CoM_Main(Part1)
        #xcen=xcen[-1]
        #ycen=ycen[-1]
        #zcen=zcen[-1]
        xcen = np.sum(Part1.xp[0]*Part1.mp0)/np.sum(Part1.mp0)
        ycen = np.sum(Part1.xp[1]*Part1.mp0)/np.sum(Part1.mp0)
        zcen = np.sum(Part1.xp[2]*Part1.mp0)/np.sum(Part1.mp0)

        #CoM_check_plot(Part1,Cell1,300,300,xcen,ycen,zcen)
        #xcen = Part1.boxpc/2; ycen=Part1.boxpc/2;zcen=Part1.boxpc/2
        rstar = get2dr(Part1.xp[0], Part1.xp[1],xcen,ycen)
        rcell = get2dr(Cell1.x, Cell1.y, xcen,ycen)
        zdistcell = np.abs(Cell1.z-zcen)
        zdistpart = np.abs(Part1.xp[2]-zcen)
        for i in range(len(rgrid)-1):

            indstar = np.where((rstar < rgrid[i+1])&(rstar>=rgrid[i])&(Part1.starage>=0)&(Part1.starage<10)&(zdistpart<8000))
            indcell = np.where((rcell < rgrid[i+1])&(rcell>=rgrid[i])&(zdistcell<8000))
            area = np.pi * (rgrid[i+1]**2-rgrid[i]**2)
            sd = np.sum(Cell1.mHIH2[indcell])/area #solar mass/pc^2
            sfrd = np.sum(Part1.mp0[indstar])/10/area #solar mass /yr/kpc^2
            sdarr=np.append(sdarr,sd)
            sfrdarr=np.append(sfrdarr,sfrd)
        halfmass_sd, halfmass_sfrd,halfmassrad = halfmassrad_KS(Part1,Cell1,xcen,ycen,zcen,rgrid2,8000)
        print('hmr',halfmassrad)
        sdarr = np.append(sdarr, halfmass_sd)
        sfrdarr = np.append(sfrdarr, halfmass_sfrd)
        np.savetxt(read+'sd.dat',sdarr,delimiter=' ',newline='\n')
        np.savetxt(read+'sfrd.dat',sfrdarr,delimiter=' ',newline='\n')

        for i in range(len(rgrid) - 1):
            if i == 3:
                ax.scatter(sdarr[i], sfrdarr[i], color=color, s=200 * 0.75 ** i, label=label)
            else:
                ax.scatter(sdarr[i], sfrdarr[i], color=color, s=200 * 0.75 ** i)


    else:

        sdarr=np.loadtxt(read+'sd.dat')
        sfrdarr=np.loadtxt(read+'sfrd.dat')


        #ax.scatter(sdarr, sfrdarr,  color=color, s=np.linspace(25, 5, len(sdarr)))


def findsnap(read, inisnap, endsnap, time): # time in unit of Myr
    numsnap = endsnap - inisnap + 1

    for i in range(numsnap):
        nout = i + inisnap
        if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
            print(read + '/SAVE/cell_%05d.sav' % (nout))
            continue
        Part1 = Part(read, nout)
        snaptime = Part1.snaptime
        if snaptime > time:
            if i==0:
                print(nout)
                return nout
            else:
                Part2 = Part(read,nout+1)
                snaptime2 = Part2.snaptime
                if abs(snaptime2-time) > abs(snaptime-time):
                    print(nout)
                    return nout
                else:
                    print(nout+1)
                    return nout+1
        if i==numsnap-1:
            print('end of iteration')
            return nout

def main(ax, read, Cell,label,color, saveload, noutdirect,diskmass):
    if noutdirect==0:
        nout=findsnap(read, 200,250, 200)
    else:
        nout=noutdirect
    Part1=Part(read,nout)
    Cell1=Cell(read,nout,Part1)

    KS(ax,read, Part1,Cell1,saveload,label,color,diskmass)



def emp(x):
    return 1.5e-4*x**1.4
fig = plt.figure(figsize=(9,8),dpi=54)
ax1=fig.add_axes([0.2,0.15,0.75,0.75])
#ax1.plot([0.03,200],[emp(0.03),emp(200)],ls='dashed')
ax1.plot([0.1,1e5],[emp(0.1),emp(1e5)],ls='dotted',c='grey')
ax1.annotate('$\Sigma_*\propto \Sigma_{gas}^{1.4}$',xy=(1e3,2*emp(1e3)),xycoords='data',rotation=np.degrees(np.arctan2(np.log10(emp(1e5))-np.log10(emp(0.1)),6)))
ax1.plot([0.1,1e5],[0.1/1e4,1e5/1e4],ls='dotted',c='grey')
ax1.annotate('1%',xy=(1,1.1*1/1e4),xycoords='data',rotation=40)
ax1.plot([0.1,1e5],[0.1/1e3,1e5/1e3],ls='dotted',c='grey')
ax1.annotate('10%',xy=(1,1.1*1/1e3),xycoords='data',rotation=40)
ax1.plot([0.1,1e5],[0.1/1e2,1e5/1e2],ls='dotted',c='grey')
ax1.annotate('100%',xy=(1,1.1*1/1e2),xycoords='data',rotation=40)


cmap1 = plt.get_cmap('Purples')
cmap2 = plt.get_cmap('Greys')
gauss_cont(ax1, '/Users/taehwau/projects/KS/bigiel10_inner.fit', '/Users/taehwau/projects/KS/bigiel10_outer.fit',8,6)

#hist2d(ax1,'/Users/taehwau/projects/KS/bigiel10_inner.fit','/Users/taehwau/projects/KS/bigiel10_outer.fit',8,6,cmap1,'Bigiel+10 ')
#hist2d(ax1,'/Users/taehwau/projects/KS/bigiel10_outer.fit',6,cmap2,'Bigiel+10 (outer disk)')
scatter(ax1,'/Users/taehwau/projects/KS/kennicutt07.fit',4,'grey',0,3,'x', 'Kennicutt+07 (M51)')
scatter2(ax1,'/Users/taehwau/projects/KS/tacconi13.fit',5,'grey','grey',4,3,'^', 's','Tacconi+13 (z<2)','Tacconi+13 (z>2)')

main(ax1,read_new_01Zsun,Cell,'G9_Zlow','k',False,300,1.75e9)
main(ax1, read_new_1Zsun, Cell,'G9_Zhigh','firebrick',False,300,1.75e9)
main(ax1,read_new_gasrich,Cellfromdat,'G9_Zlow_gas5','dodgerblue',False,300,1.75e9*5)
main(ax1,read_new_1Zsun_highSN_new,Cellfromdat,'G9_Zhigh_SN5','magenta',False,220,1.75e9)
main(ax1,read_new_03Zsun_highSN,Cellfromdat,'G9_Zmid_SN5','orange',False,220,1.75e9)
main(ax1,read_new_01Zsun_05pc,Cellfromdat,'G9_Zlow_HR','lightseagreen',False,220,1.75e9)
#main(ax1,read_new_01Zsun,Cell,'G9_Zlow','k',False,300,1.75e9)
#main(ax1,read_new_01Zsun_highSN,Cell,'G9_Zlow_SN5','olive',False,300,1.75e9)

ax1.set_xlabel(r'$\Sigma_{HI+H2}$ $(M_\odot/pc^2)$',fontsize=25)
ax1.set_ylabel(r'$\Sigma_*$ $(M_\odot/yr/kpc^2)$',fontsize=25)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.1,1e4)
ax1.set_ylim(1e-5,1e2)
ax1.legend(frameon=False)
#plt.show()
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/ks_fig2-2test.pdf')
plt.show()
