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
from skimage.transform import rescale
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
        self.nH2 = self.nH * ((1-xHI-xHII)/2)
        self.mH2 = self.m * 0.76 * ((1-xHI-xHII)/2)

        self.lev = np.round(np.log2(1 / celldata.cell.dx[0]), 0).astype(int)
        print('minmax', np.min(self.lev), np.max(self.lev))

        self.minlev = round(np.log2(1 / np.max(celldata.cell.dx[0])))
        self.maxlev = round(np.log2(1 / np.min(celldata.cell.dx[0])))
class Cellfromdat():
    def __init__(self, dir, nout, Part):
        celldata = FortranFile(dir + 'dat/cell_%05d.dat' % nout, 'r')
        nlines,xx,nvarh=celldata.read_ints(dtype=np.int32)

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
        self.nH2 = self.nH * ((1-xHI-xHII)/2)

        self.lev = np.round(np.log2(1/dxc),0)
        self.minlev = round(np.log2(1/np.max(dxc)))
        self.maxlev = round(np.log2(1/np.min(dxc)))
        self.mH2 = self.m * 0.76 * ((1-xHI-xHII)/2)


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


class new_projection:
    def __init__(self, Cell, xcenter, ycenter, zcenter, xwid, ywid, zwid, var, projection):
        maxlev = Cell.maxlev
        minlev = Cell.minlev
        numlev = int(maxlev - minlev + 1)
        mindx = np.min(Cell.dx)
        self.mindx = mindx

        self.var = var
        start = time.time()
        self.xcenter = int(xcenter / mindx)
        self.ycenter = int(ycenter / mindx)
        self.zcenter = int(zcenter / mindx)

        self.xwid = xwid
        self.ywid = ywid
        self.zwid = zwid

        self.xwid2 = int(xwid / mindx)
        self.ywid2 = int(ywid / mindx)
        self.zwid2 = int(zwid / mindx)

        xind_root = (Cell.x)/mindx/2**(maxlev-minlev)
        yind_root = (Cell.y)/mindx/2**(maxlev-minlev)
        zind_root = (Cell.z)/mindx/2**(maxlev-minlev)


        #select the root grids which are interested

        xcenind_root = int(xcenter/mindx/2**(maxlev-minlev))
        ycenind_root = int(ycenter/mindx/2**(maxlev-minlev))
        zcenind_root = int(zcenter/mindx/2**(maxlev-minlev))

        xwidind_root = int(xwid/mindx / 2**(maxlev-minlev))
        ywidind_root = int(ywid/mindx / 2**(maxlev-minlev))
        zwidind_root = int(zwid/mindx / 2**(maxlev-minlev))

        xstaind_root = int(xcenind_root-1-xwidind_root)
        xendind_root = int(xcenind_root+1+xwidind_root)
        ystaind_root = int(ycenind_root-1-ywidind_root)
        yendind_root = int(ycenind_root+1+ywidind_root)
        zstaind_root = int(zcenind_root-1-zwidind_root)
        zendind_root = int(zcenind_root+1+zwidind_root)
        #print(self.xcenter,xwid2,xcenind_root,xwidind_root)
        sumvol=0
        numind=0
        numind3=0
        ind2 = np.where((xind_root>=xstaind_root) & (xind_root<=xendind_root+1) & (yind_root>=ystaind_root) & (yind_root<=yendind_root+1) & (zind_root>=zstaind_root) & (zind_root<=zendind_root+1))
#        print('minmax',minmax(xind_root),minmax(yind_root),minmax(zind_root))


        zmin = (zstaind_root) * 2 ** (maxlev - minlev) * self.mindx
        zmax = (zendind_root+1) * 2 ** (maxlev - minlev) * self.mindx

        xmin = (xstaind_root) * 2 ** (maxlev - minlev) * self.mindx
        xmax = (xendind_root+1) * 2 ** (maxlev - minlev) * self.mindx
        ymin = (ystaind_root) * 2 ** (maxlev - minlev) * self.mindx
        ymax = (yendind_root+1) * 2 ** (maxlev - minlev) * self.mindx

        if projection =='xy':
            histrange = [[xmin, xmax], [ymin, ymax]]
        if projection == 'xz':
            histrange = [[xmin, xmax], [zmin, zmax]]



        ind4 = np.where((Cell.x-Cell.dx/2>=xmin)&(Cell.x+Cell.dx/2<=xmax)&(Cell.y-Cell.dx/2>=ymin)&(Cell.y+Cell.dx/2<=ymax)&(Cell.z-Cell.dx/2>=zmin)&(Cell.z+Cell.dx/2<=zmax))

        for n in range(numlev):
            lev = minlev + n
            #ind3 = np.where((Cell.dx==mindx*(2**(maxlev-minlev-n)))&(Cell.x-Cell.dx/2>=xmin)&(Cell.x+Cell.dx/2<=xmax)&(Cell.y-Cell.dx/2>=ymin)&(Cell.y+Cell.dx/2<=ymax)&(Cell.z-Cell.dx/2>=zmin)&(Cell.z+Cell.dx/2<=zmax))
            #numind3 = numind3 + len(ind3[0])
            ind = np.where((Cell.lev.astype(int)==lev) & (xind_root>=xstaind_root) & (xind_root<xendind_root+1) & (yind_root>=ystaind_root) & (yind_root<yendind_root+1) & (zind_root>=zstaind_root) & (zind_root<zendind_root+1))
            #ind = ind3
            numind = numind + len(ind[0])
            dx = Cell.dx[ind]
            if projection == 'xy':
                x = Cell.x[ind]
                y = Cell.y[ind]
                z = Cell.z[ind]
                bins = [int((3 + 2 * xwidind_root) * 2 ** (lev - minlev)),
                        int((3 + 2 * ywidind_root) * 2 ** (lev - minlev))]

            elif projection =='xz':
                x = Cell.x[ind]
                y = Cell.z[ind]
                z = Cell.y[ind]
                bins = [int((3 + 2 * xwidind_root) * 2 ** (lev - minlev)),
                        int((3 + 2 * zwidind_root) * 2 ** (lev - minlev))]

            else:
                raise ValueError('improper projection description')
            if var=='nH': #volume-weighted
                sumvar = Cell.nH[ind]*Cell.dx[ind]
                weight = Cell.dx[ind]

            elif var=='T': #mass-weighted
                sumvar = Cell.T[ind]*Cell.m[ind]
                weight = Cell.m[ind]

            elif var=='xHI':
                sumvar = Cell.xHI[ind] * Cell.dx[ind]
                weight = Cell.dx[ind]
            elif var=='xH2':
                sumvar = Cell.xH2[ind] * Cell.dx[ind]
                weight = Cell.dx[ind]

            elif var == 'nH2':
                sumvar = Cell.nH2[ind] * Cell.dx[ind]
                weight = Cell.dx[ind]

            print('indexing level %d, t=%3.2f (s), #=%d' % (lev, time.time() - start, len(ind[0])))
            numarr = np.histogram2d(x, y, bins=bins, range=histrange)[0]

            sumarr = np.histogram2d(x, y, bins=bins, weights=sumvar,range=histrange)[0]
            weiarr = np.histogram2d(x, y, bins=bins, weights=weight,range=histrange)[0]
            #sumstat = stats.binned_statistic_2d(x, y, sumvar, bins=[2**lev,2**lev], statistic='sum')
            #weistat = stats.binned_statistic_2d(x, y, weight, bins=[2 ** lev, 2 ** lev], statistic='sum')
            print('complete level %d binning, t=%3.2f (s)' % (lev, time.time() - start))
            start = time.time()
            #sumarr = sumstat.statistic
            #weiarr = weistat.statistic
            """
            if n==0:


                sumarr2 = sumarr
                weiarr2 = weiarr

            else:
                #print(sumarr2)
                #print(sumarr2.shape)

                for i in range(2):
                    sumarr2 = np.repeat(sumarr2,2,axis=i)
                    weiarr2 = np.repeat(weiarr2,2,axis=i)

                sumarr2 = sumarr2 + sumarr
                weiarr2 = weiarr2 + weiarr
            """

            if n==0:
                sumarr2 = sumarr
                weiarr2 = weiarr

            else:
                sumarr2 = rescale(sumarr2, 2, mode='constant', order=0, multichannel=False, anti_aliasing=False)
                weiarr2 = rescale(weiarr2, 2, mode='constant', order=0, multichannel=False, anti_aliasing=False)
                sumarr2 = sumarr2 + sumarr
                weiarr2 = weiarr2 + weiarr

            print('complete level %d increasing size, t=%3.2f (s)' % (lev, time.time() - start))

            start = time.time()

            sumvol = sumvol + np.sum(dx**3)
        #print(np.min(sumarr2), np.max(sumarr2), np.min(weiarr2), np.max(weiarr2))
        #print(np.where(sumarr2==0), np.where(weiarr2==0))
        if projection=='xy':

            xstacut = int(self.xcenter - self.xwid2 - (xstaind_root) * 2 ** (maxlev - minlev))
            xendcut = int(self.xcenter + self.xwid2 - (xstaind_root) * 2 ** (maxlev - minlev))

            ystacut = int(self.ycenter - self.ywid2 - (ystaind_root) * 2 ** (maxlev - minlev))
            yendcut = int(self.ycenter + self.ywid2 - (ystaind_root) * 2 ** (maxlev - minlev))

        if projection =='xz':
            self.xcen2 = self.xcenter
            self.zcen2 = self.zcenter
            xstacut = int(self.xcenter - self.xwid2 - (xstaind_root) * 2 ** (maxlev - minlev))
            xendcut = int(self.xcenter + self.xwid2 - (xstaind_root) * 2 ** (maxlev - minlev))

            ystacut = int(self.zcenter - self.zwid2 - (zstaind_root) * 2 ** (maxlev - minlev))
            yendcut = int(self.zcenter + self.zwid2 - (zstaind_root) * 2 ** (maxlev - minlev))
        # crop
        self.xmin = int(self.xcenter-self.xwid2)*mindx
        self.xmax = int(self.xcenter+self.xwid2)*mindx
        self.ymin = int(self.ycenter - self.ywid2) * mindx
        self.ymax = int(self.ycenter + self.ywid2) * mindx
        self.zmin = int(self.zcenter - self.zwid2) * mindx
        self.zmax = int(self.zcenter + self.zwid2) * mindx

        #print(xstacut,xendcut,ystacut, yendcut)
        sumarr2 = sumarr2[xstacut:xendcut,ystacut:yendcut]
        weiarr2 = weiarr2[xstacut:xendcut,ystacut:yendcut]
        print(np.min(sumarr2),np.max(sumarr2), np.min(weiarr2),np.max(weiarr2))
        self.avarr = np.log10(sumarr2/weiarr2)
        #print(self.avarr)
        #print(self.avarr.shape)
        self.projection = projection

    def projectionPlot(self, ax, cm, ticks,cbar,ruler,corr,text, label):
        start = time.time()

        if self.projection == 'xy':
            im = ax.imshow(np.rot90(self.avarr), cmap=cm,
                           extent=[-self.xwid / 1000, self.xwid / 1000, -self.ywid / 1000,
                                   self.ywid / 1000], vmin=np.min(ticks), vmax=np.max(ticks), aspect='equal')
            ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
            ax.set_ylim(-self.ywid / 1000, self.ywid / 1000)
        elif self.projection =='xz':
            im = ax.imshow(np.rot90(self.avarr), cmap=cm,
                           extent=[-self.xwid / 1000, self.xwid / 1000, -self.zwid / 1000,
                                   self.zwid / 1000], vmin=np.min(ticks), vmax=np.max(ticks), aspect='equal')
            ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
            ax.set_ylim(-self.zwid / 1000, self.zwid / 1000)

        if cbar==True:
            cbaxes = inset_axes(ax, width="100%", height="100%", loc=3, bbox_to_anchor=(0.05,0.05,0.25,0.02),bbox_transform=ax.transAxes)
            cbar = plt.colorbar(im, cax=cbaxes, ticks=ticks, orientation='horizontal', cmap=cm)
            cbar.set_label('log(' + self.var + ')', color='w', labelpad=-50, fontsize=15)
            cbar.ax.xaxis.set_tick_params(color='w')
            cbar.ax.xaxis.set_ticks_position('bottom')
            plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')

        """
        you have to insert appropriate number for below 'rectangles' 
        this is ruler which indicates the size of projected image
        ex) if you want to draw 5 kpc width projected image and want to insert ruler with 3 kpc size, then 
        replace 5 kpc into 3 kpc and you have to multiply 3/5 instead of 5/14 in width. 
        """
        if ruler==True:
            if self.projection == 'xy':

                rectangles = {
                    '1 kpc': patches.Rectangle(xy=(0.25 * self.xwid / 1000, -0.88 * self.ywid / 1000),
                                               width=(2 * self.ywid / 1000) * 1/8,
                                               height=0.01 * (2 * self.ywid / 1000)*corr, facecolor='white')}
                for r in rectangles:
                    ax.add_artist(rectangles[r])
                    rx, ry = rectangles[r].get_xy()
                    cx = rx + rectangles[r].get_width() / 2.0
                    cy = ry + rectangles[r].get_height() / 2.0

                    ax.annotate(r, (cx, cy + 0.02 * (2 * self.ywid / 1000)*corr), color='w', weight='bold',
                                fontsize=15, ha='center', va='center')
            if self.projection == 'xz':
                rectangles = {
                    '5 kpc': patches.Rectangle(xy=(0.25 * self.xwid / 1000, -0.88 * self.zwid / 1000),
                                               width=(2 * self.xwid / 1000) * 5 / 14,
                                               height=0.01 * (2 * self.zwid / 1000) * corr, facecolor='white')}
                for r in rectangles:
                    ax.add_artist(rectangles[r])
                    rx, ry = rectangles[r].get_xy()
                    cx = rx + rectangles[r].get_width() / 2.0
                    cy = ry + rectangles[r].get_height() / 2.0

                    ax.annotate(r, (cx, cy + 0.02 * (2 * self.zwid / 1000) * corr), color='w', weight='bold',
                                fontsize=15, ha='center', va='center')
        if text==True:
            if self.projection=='xy':
                ax.text(-self.xwid/1000*0.9,self.ywid/1000*0.8,label,color='w',fontsize=40)
            if self.projection=='xz':
                ax.text(-self.xwid/1000*0.9,self.zwid/1000*0.8,label,color='w',fontsize=40)
        return im

    def star_plot(self, Part, ax, ind):

        start=time.time()

        print('star plotting...')

        ex_xcenter = self.xcenter*self.mindx
        ex_ycenter = self.ycenter*self.mindx
        ex_zcenter = self.zcenter*self.mindx

        sxplot = (Part.xp[0] - ex_xcenter)/1000
        syplot = (Part.xp[1] - ex_ycenter)/1000
        szplot = (Part.xp[2] - ex_zcenter)/1000
        if self.projection == 'xy':
            cax1 = ax.scatter(sxplot[ind], syplot[ind], c='cyan', s=1, alpha=0.8)
            ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
            ax.set_ylim(-self.ywid / 1000, self.ywid / 1000)

        if self.projection == 'xz':
            cax1 = ax.scatter(sxplot[ind], szplot[ind], c='w', s=1, alpha=0.8)
            ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
            ax.set_ylim(-self.zwid / 1000, self.zwid / 1000)

        print('plotting stars finished , t = %.2f [sec]' %(time.time()-start))
        return cax1


    def star_plot3(self, Part   , ax,binsize,cm,ticks,vmin,vmax, cbar, ruler,corr):

        start = time.time()

        print('star plotting...')

        ex_xcenter = self.xcenter * self.mindx
        ex_ycenter = self.ycenter * self.mindx
        ex_zcenter = self.zcenter * self.mindx

        sxplot = (Part.xp[0] - ex_xcenter) / 1000
        syplot = (Part.xp[1] - ex_ycenter) / 1000
        szplot = (Part.xp[2] - ex_zcenter) / 1000

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

        fwhm = cosmo.kpc_comoving_per_arcmin(0.1)
        # print('fwhm',fwhm)


        if self.projection == 'xy':
            x = sxplot
            y = syplot
            ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
            ax.set_ylim(-self.ywid / 1000, self.ywid / 1000)
            #histrange = [[self.xmin, self.xmax], [self.ymin, self.ymax]]
            histrange = [[-self.xwid / 1000, self.xwid / 1000], [-self.ywid / 1000, self.ywid / 1000]]
            bin = [(self.xmax - self.xmin) / binsize, (self.ymax - self.ymin) / binsize]
        elif self.projection =='xz':
            x = sxplot
            y = szplot
            ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
            ax.set_ylim(-self.zwid / 1000, self.zwid / 1000)
            histrange = [[-self.xwid / 1000, self.xwid / 1000], [-self.zwid / 1000, self.zwid / 1000]]
            bin = [(self.xmax - self.xmin) / binsize, (self.zmax - self.zmin) / binsize]

        sfrd= np.histogram2d(x, y, weights=Part.mp0 / (binsize/1000)**2, range=histrange, bins=bin)[0]

        sfrd_gauss = gaussian_filter(sfrd, sigma=2)

        if self.projection =='xy':
            im = ax.imshow(np.log10(np.rot90(sfrd_gauss+1.)),  cmap=cm,extent=[-self.xwid / 1000, self.xwid / 1000, -self.ywid / 1000,
                                   self.ywid / 1000],interpolation='none', aspect='equal',vmin=vmin,vmax=vmax)
        elif self.projection =='xz':
            im = ax.imshow(np.log10(np.rot90(sfrd_gauss+1.)), cmap=cm,extent=[-self.xwid / 1000, self.xwid / 1000, -self.zwid / 1000,
                                          self.zwid / 1000], interpolation='none', aspect='equal',vmin=vmin,vmax=vmax)
        if cbar==True:
            cbaxes = inset_axes(ax, width="100%", height="100%", loc=3, bbox_to_anchor=(0.05, 0.05, 0.25, 0.02),
                                bbox_transform=ax.transAxes)
            cbar = plt.colorbar(im, cax=cbaxes, ticks=ticks, orientation='horizontal', cmap=cm)
            cbar.set_label('$log(\Sigma_*) (M_\odot\cdot kpc^{-2}$)', color='w', labelpad=-50, fontsize=15)
            cbar.ax.xaxis.set_tick_params(color='w')
            cbar.ax.xaxis.set_ticks_position('bottom')
            plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')

        if ruler==True:
            if self.projection =='xy':

                rectangles = {
                    '1 kpc': patches.Rectangle(xy=(0.25 * self.xwid / 1000, -0.88 * self.xwid / 1000),
                                               width=(2 * self.xwid / 1000) * 1/8,
                                               height=0.01 * (2 * self.ywid / 1000)*corr, facecolor='white')}
                for r in rectangles:
                    ax.add_artist(rectangles[r])
                    rx, ry = rectangles[r].get_xy()
                    cx = rx + rectangles[r].get_width() / 2.0
                    cy = ry + rectangles[r].get_height() / 2.0

                    ax.annotate(r, (cx, cy + 0.02 * (2 * self.ywid / 1000)*corr), color='w', weight='bold',
                                fontsize=15, ha='center', va='center')
            if self.projection =='xz':
                rectangles = {
                    '1 kpc': patches.Rectangle(xy=(0.25 * self.xwid / 1000, -0.88 * self.zwid / 1000),
                                               width=(2 * self.xwid / 1000) * 1/8,
                                               height=0.01 * (2 * self.zwid / 1000)*corr, facecolor='white')}
                for r in rectangles:
                    ax.add_artist(rectangles[r])
                    rx, ry = rectangles[r].get_xy()
                    cx = rx + rectangles[r].get_width() / 2.0
                    cy = ry + rectangles[r].get_height() / 2.0

                    ax.annotate(r, (cx, cy + 0.02 * (2 * self.zwid / 1000)*corr), color='w', weight='bold',
                                fontsize=15, ha='center', va='center')

    def clump_plot(self, Clump, ax):
        # for appropriate description of size of clump, dpi = 144, figsize * size of axis = size
        # clump finding
        start = time.time()
        print('finding gas clumps...')

        ex_xcenter = self.xcenter * self.mindx
        ex_ycenter = self.ycenter * self.mindx
        ex_zcenter = self.zcenter * self.mindx

        cxplot = (Clump.xclump - ex_xcenter) / 1000
        cyplot = (Clump.yclump - ex_ycenter) / 1000
        czplot = (Clump.zclump - ex_zcenter) / 1000

        cax1 = ax.scatter(cxplot, cyplot, edgecolor='k', marker='o',
                          s=(Clump.rclump * ax.get_window_extent().width / (2 * self.xwid))** 2, linewidths=1, facecolors='none')
        ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
        ax.set_ylim(-self.ywid / 1000, self.ywid / 1000)

        return cax1
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



def main(read, ini, end, Cell, label):
    num = end - ini +1
    for n in range(num):
        nout = n + ini
        if not os.path.isfile(read + '/SAVE/part_%05d.sav' % (nout)):
            print(read + '/SAVE/part_%05d.sav' % (nout))
            continue
        Part1 = Part(read, nout)
        Cell1 = Cell(read, nout, Part1)
        #xcen, ycen, zcen = CoM_Main(Part1)
        #xcen=xcen[-1];ycen=ycen[-1];zcen=zcen[-1]
        xcen = np.sum(Part1.mp0 * Part1.xp[0])/np.sum(Part1.mp0)
        ycen = np.sum(Part1.mp0 * Part1.xp[1]) / np.sum(Part1.mp0)
        zcen = np.sum(Part1.mp0 * Part1.xp[2]) / np.sum(Part1.mp0)
        fig = plt.figure(figsize=(5*2.5,4*2.5))
        ax1 = fig.add_axes([0.1,0,0.4,0.5])
        ax3 = fig.add_axes([0.5,0,0.4,0.5])

        ax2 = fig.add_axes([0.1,0.6,0.8,0.14])
        ax4 = fig.add_axes([0.1,0.85,0.8,0.14])


        ax2.set_xlabel('r (pc)')
        ax2.set_ylabel("$\Sigma_* (M_\odot/yr/kpc^2)$")
        ax4.set_xlabel('r (pc)')
        ax4.set_ylabel("$\Sigma_{H_2} (M_\odot/pc^2)$")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        cm1 = plt.get_cmap('inferno')
        cm2 = plt.get_cmap('pink')

        a= new_projection(Cell1, xcen, ycen, zcen, 4000,4000,4000,'nH','xy')
        a.projectionPlot(ax1, cm1,[-3,-2,-1,0,1,2],True,True,1,True,label)
        ind = np.where(Part1.starage<10)

        a.star_plot(Part1,ax1,ind)
        rstar = get2dr(Part1.xp[0], Part1.xp[1],xcen,ycen)
        rcell = get2dr(Cell1.x, Cell1.y,xcen,ycen)

        zdistpart = np.abs(Part1.xp[2] - zcen)
        zdistcell = np.abs(Cell1.z - zcen)

        sfrdarr = np.array([])
        sdarr = np.array([])

        b= new_projection(Cell1, xcen, ycen, zcen, 4000,4000,4000,'nH2','xy')
        b.projectionPlot(ax3, cm2,[-3,-2,-1,0,1,2],True,True,1,True,label)

        for i in range(8):
            rgrid = np.linspace(0,4000,num=9)
            indstar = np.where((rstar < rgrid[i+1])&(rstar>=rgrid[i])&(Part1.starage>=0)&(Part1.starage<10)&(zdistpart<8000))
            area = np.pi * (rgrid[i + 1] ** 2 - rgrid[i] ** 2)
            sfrd = np.sum(Part1.mp0[indstar]) / 10 / area  # solar mass /yr/kpc^2


            indcell = np.where((rcell < rgrid[i+1])&(rcell>=rgrid[i])&(zdistcell<8000))
            sd = np.sum(Cell1.mH2[indcell])/area
            rr = np.linspace(500,4000,num=8)
            circle = plt.Circle((0,0),rr[i]/1000,edgecolor='w',fill=False)
            circle2 = plt.Circle((0,0),rr[i]/1000,edgecolor='w',fill=False)

            ax1.add_artist(circle)
            ax3.add_artist(circle2)

            sfrdarr = np.append(sfrdarr, sfrd)
            sdarr = np.append(sdarr, sd)


        x = np.linspace(0,4000,num=9)
        xx = (x[1:] +x[:-1])/2

        ax2.plot(xx,np.log10(sfrdarr),c='k')
        ax4.plot(xx,np.log10(sdarr),c='k')
        plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/ks/%s_kscheck_%05d.png'%(label,nout))
        plt.close()

#main(read_new_01Zsun,300,Cell,'G9_Zlow')
main(read_new_gasrich,150,300,Cellfromdat,'G9_Zlow_gas5')

main(read_new_1Zsun,150,480,Cell,'G9_Zhigh')
#main(read_new_1Zsun_highSN_new,220,Cell,'G9_Zhigh_SN5')
#main(read_new_03Zsun_highSN,220,Cell,'G9_Zmid_SN5')
#main(read_new_01Zsun_05pc,220,Cell,'G9_Zlow_HR')
