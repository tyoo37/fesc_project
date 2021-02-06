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
        print('cell1')
        self.x = celldata.cell.x[0] * Part.boxpc
        self.y = celldata.cell.y[0] * Part.boxpc
        self.z = celldata.cell.z[0] * Part.boxpc
        self.dx = celldata.cell.dx[0] * Part.boxpc
        self.mindx = np.min(celldata.cell.dx[0])
        self.m = celldata.cell[0][4][0] *Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l *Part.unit_l *(celldata.cell.dx[0]*Part.boxlen)**3
        self.vx = celldata.cell[0][4][1] * Part.unit_l /4.70430e14
        self.vy = celldata.cell[0][4][2] * Part.unit_l /4.70430e14
        self.vz = celldata.cell[0][4][3] * Part.unit_l /4.70430e14
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


class GasrichCell():
    def __init__(self, dir, nout, Part):
        self.dir = dir
        self.nout = nout
        print(nout)
        celldata = readsav(self.dir + '/SAVE/cell_%05d.sav' % (self.nout))
        self.nH = celldata.cell.nh[0] * 30.996344
        self.x = celldata.cell.x[0] * Part.boxpc
        self.y = celldata.cell.y[0] * Part.boxpc
        self.z = celldata.cell.z[0] * Part.boxpc
        self.dx = celldata.cell.dx[0] * Part.boxpc
        self.mindx = np.min(celldata.cell.dx[0])
        self.m = celldata.cell.nh[0] *Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l *Part.unit_l *(celldata.cell.dx[0]*Part.boxlen)**3
#        self.vx = celldata.cell.vx[0]* Part.unit_l /4.70430e14
 #       self.vy = celldata.cell.vy[0]* Part.unit_l /4.70430e14
  #      self.vz = celldata.cell.vz[0]* Part.unit_l /4.70430e14

class GenerateArray():

    def __init__(self, Cell, xcenter, ycenter, zcenter, wid, height,depth, orientation):



        pc = 3.08e18
        mindx = np.min(Cell.dx)

        maxgrid = int(np.log2(np.max(Cell.dx) / mindx))

        xind = Cell.x / mindx - 0.5
        yind = Cell.y / mindx - 0.5
        zind = Cell.z / mindx - 0.5


        if orientation == 'xy':
            xwid = wid
            ywid = height
            zwid = depth

        elif orientation == 'yz' :
            xwid = depth
            ywid = wid
            zwid = height
        elif orientation =='xz':
            xwid = wid
            ywid = depth
            zwid = height
        print(xwid, ywid, zwid)
        self.xfwd = 2 * int(xwid)
        self.yfwd = 2 * int(ywid)
        self.zfwd = 2 * int(zwid)

        xini = int(xcenter/mindx) - xwid + 1
        yini = int(ycenter/mindx) - ywid + 1
        zini = int(zcenter/mindx) - zwid + 1
        xfin = int(xcenter/mindx) + xwid
        yfin = int(ycenter/mindx) + ywid
        zfin = int(zcenter/mindx) + zwid
        # print(max(celldata.cell[0][4][0]))

        self.leafcell = np.zeros((self.xfwd, self.yfwd, self.zfwd))

        ind_1 = np.where((Cell.dx == mindx) & (xind >= xini) & (xind <= xfin)
                         & (yind >= yini) & (yind <= yfin) & (zind >= zini) & (zind <= zfin))[0]

        self.leafcell[xind[ind_1].astype(int) - int(xini), yind[ind_1].astype(int) - int(yini), zind[ind_1].astype(int) - int(zini)] = ind_1.astype(int)

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
        self.orientation = orientation


    def projectionPlot(self, Cell, ax, cm, field, vmin, vmax):

        if field == 'nH':
            var = Cell.nH
            if self.orientation == 'xy':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=2) / self.zfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                                extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000, -self.ywid * self.mindx / 1000,
                                        self.ywid * 9.1 / 1000], vmin=-3, vmax=2)

            elif self.orientation == 'yz':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=0) / self.xfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                                extent=[-self.ywid * self.mindx / 1000, self.ywid * self.mindx / 1000, -self.zwid * self.mindx / 1000,
                                        self.zwid * self.mindx / 1000], vmin=-3, vmax=2)

            elif self.orientation == 'zx' or 'xz':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=1) / self.yfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                                extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000, -self.zwid * self.mindx / 1000,
                                        self.zwid * self.mindx / 1000], vmin=-3, vmax=2)
        return cax

    def star_plot(self, Part,  ax):
        print('star plotting...')
        sxind = Part.xp[0] / self.mindx - self.xcenter
        syind = Part.xp[1] / self.mindx - self.ycenter
        szind = Part.xp[2] / self.mindx - self.zcenter
        sind = np.where(
            (sxind >= 3 - self.xwid) & (sxind < self.xwid - 3) & (syind >= 3 - self.ywid) & (syind < self.ywid - 3) & (szind - self.zwid >= 3) & (
                        szind < self.zwid - 3))[
            0]
        # sind = np.where((sxind >= 0) & (sxind < xfwd) & (syind >= 0) & (syind < yfwd) & (szind >= 0) & (
        # szind < zfwd))[0]
        sxind = sxind[sind]
        syind = syind[sind]
        szind = szind[sind]

        sxplot = sxind * self.mindx
        syplot = syind * self.mindx
        szplot = szind * self.mindx
        if self.orientation=='xy':

            cax1 = ax.scatter(sxplot/1000, syplot/1000,  c='grey', s=10,alpha=0.7)
        if self.orientation=='yz':

            cax1 = ax.scatter(syplot/1000, szplot/1000,  c='grey', s=10,alpha=0.7)
        if self.orientation=='xz':

            cax1 = ax.scatter(sxplot/1000, szplot/1000,  c='grey', s=10,alpha=0.7)
        return cax1

def CoM(Part1,Cell1, type, inir):


    if type =='part' or type=='star':
        x = Part1.xp[0]
        y = Part1.xp[1]
        z = Part1.xp[2]
        m = Part1.mp0
        #print('part')
    if type =='cell' or type=='gas':
        x = Cell1.x
        y = Cell1.y
        z = Cell1.z
        m = Cell1.m
        #print('cell')
    xy = np.sqrt((x - Part1.boxpc / 2) ** 2 + (y - Part1.boxpc / 2) ** 2)
    for j in range(10):
        index = np.where(xy < inir * (0.95 ** j))
        xcenter = np.sum(x[index] * m[index]) / np.sum(m[index])
        ycenter = np.sum(y[index] * m[index]) / np.sum(m[index])
        #print('xcenter', xcenter-Part1.boxpc/2, 'ycenter', ycenter-Part1.boxpc/2)
        xy = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)

    return xcenter, ycenter

def CoM2(Part,Cell , type, inir):

    if type == 'part' or  type=='star':
        x = Part.xp[0]
        y = Part.xp[1]
        z = Part.xp[2]
        m = Part.mp0
        print('part')
    if type == 'cell' or type=='gas':
        x = Cell.x
        y = Cell.y
        z = Cell.z
        m = Cell.m
        dx = Cell.dx
        nH = Cell.nH
        print('cell')
    rr = np.sqrt((x - Part.boxpc / 2) ** 2 + (y - Part.boxpc / 2) ** 2)
    centerindex = np.where((abs(z - Part.boxpc / 2) < 10000) & (rr < 5000))[0]

    xcenter = np.sum(x[centerindex] * m[centerindex]) / np.sum(m[centerindex])
    ycenter = np.sum(y[centerindex] * m[centerindex]) / np.sum(m[centerindex])
    #zcenter = np.sum(z * nH * dx ** 3) / np.sum(nH *dx ** 3)

    return xcenter, ycenter

def scaleheight(Part1,Cell1, xcenter, ycenter, zcenter,xyrange,zrange, field):

    if field == 'gas':
        cutindex = np.where((get2dr(Cell1.x,Cell1.y,xcenter,ycenter)<xyrange) & (
                    np.abs(Cell1.z - zcenter) < zrange))
        z = Cell1.z[cutindex]
        dx = Cell1.dx[cutindex]
        nH = Cell1.nH[cutindex]
        #zcenter = np.sum(z * nH * dx ** 3) / np.sum(nH * dx ** 3)

        H = np.sqrt(np.sum((z - zcenter) ** 2 * nH * (dx) ** 3) / np.sum(nH * (dx) ** 3))

    if field =='star':
        cutindex = np.where((get2dr(Part1.xp[0], Part1.xp[1], xcenter, ycenter) < xyrange) & (
                np.abs(Part1.xp[2] - zcenter) < zrange))

        H = np.sqrt(np.sum((Part1.xp[2][cutindex]-zcenter)**2*Part1.mp0[cutindex])/np.sum(Part1.mp0[cutindex]))



    return H

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
    for i in range(10):

        ind = np.where((get2dr(Part1.xp[0],Part1.xp[1],xcen[i],ycen[i])<hmr[i])&(np.abs(Part1.xp[2]-zcen[i])<2000))
        xcen[i+1]=np.sum(Part1.xp[0][ind]*Part1.mp0[ind])/np.sum(Part1.mp0[ind])
        ycen[i+1]=np.sum(Part1.xp[1][ind]*Part1.mp0[ind])/np.sum(Part1.mp0[ind])
        zcen[i+1]=np.sum(Part1.xp[2][ind]*Part1.mp0[ind])/np.sum(Part1.mp0[ind])
        for j in range(len(rgrid)):
            mass = getmass_zlim(Part1.mp0, get2dr(Part1.xp[0], Part1.xp[1], xcen[i+1], ycen[i+1]), rgrid[j], Part1.xp[2],
                                zcen[i+1], 2000)
            if mass > np.sum(Part1.mp0) / 2:
                hmr[i+1] = rgrid[j]
                break
    print(xcen, ycen, zcen)
    return xcen, ycen, zcen
def halfmassrad(Part,Cell,xcen,ycen,zcen,rbin, zrange):
    totstarmass = np.sum(Part.mp0)
    totgasmass = np.sum(Cell.m)
    cellrr = get2dr(Cell.x,Cell.y, xcen,ycen)
    cellzdist = np.abs(Cell.z-zcen)
    partrr = get2dr(Part.xp[0],Part.xp[1],xcen,ycen)
    partzdist = np.abs(Part.xp[2]-zcen)
    starhalfmassrad=0; cellhalfmassrad=0


    for i in range(len(rbin)):
        partind = np.where((partrr<rbin[i])&(partzdist<zrange))
        if np.sum(Part.mp0[partind])>totstarmass/2:
            starhalfmassrad = rbin[i]
            break

    for i in range(len(rbin)):
        cellind = np.where((cellrr<rbin[i])&(cellzdist<zrange))
        if np.sum(Cell.m[cellind])>totgasmass/2:
            cellhalfmassrad = rbin[i]
            break
    if cellhalfmassrad ==0:
        raise ValueError('what the')
    return starhalfmassrad, cellhalfmassrad

def main(inisnap, endsnap, read, Part, Cell, zrange,skip,load,diskmass):
    if load==False:
        numsnap = endsnap - inisnap + 1
        Harr1 = []
        Harr2 = []
        timearr =np.array([])
        outflowarr=np.array([])
        for n in range(numsnap):
            start = time.time()
            skip3 = 0

            nout = inisnap + n
            for j in range(len(skip)):
                if nout==skip[j]:
                    skip3=1
                    continue
            if skip3==1:
                continue
            print(nout)
            if not os.path.isfile(read + '/SAVE/part_%05d.sav' % (nout)):
                print(read + '/SAVE/part_%05d.sav' % (nout))
                continue

            if Cell==Cellfromdat:
                if not os.path.isfile(read + '/dat/cell_%05d.dat' % (nout)):
                    print(read + '/SAVE/cell_%05d.dat' % (nout))
                    continue
            else:
                if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
                    print(read + '/SAVE/cell_%05d.sav' % (nout))
                    continue


            if read==read_new_gasrich and nout==166:
                continue
            Part1 = Part(read, nout)
            Cell1 = Cell(read, nout, Part1)
            time2 = time.time()
            print('reading finished, %3.2f (s)'%(time2-start))
            #xcenter, ycenter, zcenter = CoM_main(Part1, Cell1,diskmass)
            xcenter,ycenter,zcenter=CoM_Main(Part1)
            xcenter=xcenter[-1];ycenter=ycenter[-1];zcenter=zcenter[-1]
            simpler = get2dr(Part1.xp[0],Part1.xp[1],xcenter,ycenter)
            xcen,ycen,zcen = simpleCoM(Part1.xp[0],Part1.xp[1],Part1.xp[2],Part1.mp0,simpler,3000)
            time3 = time.time()
            print('find CoM, %3.2f (s)'%(time3-time2))
            starhalfmassr, gashalfmassr = halfmassrad(Part1,Cell1,xcenter,ycenter,zcenter,np.linspace(0,7000,num=140),zrange)
            time4 = time.time()
            print('finding half-radius, %3.2f (s)'%(time4-time3))
            H1 = scaleheight(Part1, Cell1, xcenter, ycenter, zcenter,gashalfmassr,  zrange, 'gas')
            H2 = scaleheight(Part1, Cell1, xcen, ycen, zcen,starhalfmassr,  zrange, 'star')
            time5 = time.time()
            print('calculating scale height, %3.2f (s)'%(time5-time4))
            Harr1.append(H1)
            Harr2.append(H2)

            timearr = np.append(timearr, Part1.snaptime)

        Harr1 = np.array(Harr1)
        Harr2 = np.array(Harr2)

        np.savetxt(read+'timearr1.dat',timearr,delimiter=' ',newline='\n')
        np.savetxt(read+'Harr1.dat',Harr1,delimiter=' ',newline='\n')
        np.savetxt(read+'Harr2.dat',Harr2,delimiter=' ',newline='\n')

    else:
        timearr = np.loadtxt(read+'timearr1.dat')
        Harr1 = np.loadtxt(read+'Harr1.dat')
        Harr2 = np.loadtxt(read+'Harr2.dat')


    return timearr, Harr1, Harr2

def plot1(load):

    skip1 = [165]
    time3, thick3, thick33 = main(10, 303, read_new_gasrich, Part, Cellfromdat,6000,skip1,load,1.75e09*5)
    #time3, thick3, thick33 = main(50, 460, read_new_1Zsun, Part, Cell, 2000, [], load)

    #time4, thick4, thick44 = main(16, 393, read_new_1Zsun_highSN_new, Part, Cell, 2000,[],load,1.75e09)
    time1, thick1, thick11 = main(10, 480, read_new_01Zsun, Part, Cell, 2000,[],load,1.75e09)

    #time2, thick2, thick22 = main(50, 480, read_new_1Zsun, Part, Cell, 2000,[])


    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 20
    plt.rcParams['lines.linewidth']=3
    fig = plt.figure(figsize=(9, 9))

    ax1 = fig.add_axes([0.16, 0.15, 0.8, 0.8])

    ax1.plot(time1, thick1, c='k', label='G9_Zlow')
    #ax1.plot(time2, thick2, c='r', label='G9_1Zsun')
    ax1.plot(time3, thick3, c='b', label='G9_Zlow_gas5')
    #ax1.plot(time4, thick4, c='orange', label='G9_1Zsun_SNboost')

    ax1.plot(time1, thick11, c='k', ls='dashed')
    #ax1.plot(time2, thick22, c='r', ls='dashed')
    ax1.plot(time3, thick33, c='b', ls='dashed')
    #ax1.plot(time4, thick44, c='orange', ls='dashed')

    ax1.legend(loc='upper right',frameon=False)
    ax1.set_xlabel('time (Myr)')
    ax1.set_ylabel('scale height (pc)')

    plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig10.png')

    plt.show()
Harr = np.array([])
timearr = np.array([])
outflowarr = np.array([])
"""
def main2(Part, Cell, read, nout, initr, xyrange, zrange, zvalue,Harr, outflowarr, timearr, fescarr):


    Part1 = Part(read, nout)
    Cell1 = Cell(read, nout, Part1)
    xcenter, ycenter = CoM2(Part1, Cell1, 'gas', initr)
    H, outflow, zcenter = scaleheight(Cell1, xcenter, ycenter, xyrange, zrange, zvalue)
    Harr = np.append(Harr, H)
    outflowarr = np.append(outflowarr, outflow / 100000)
    timearr = np.append(timearr, Part1.snaptime)
    fescarr = np.append(fescarr,Part1.fesc)

    return Harr, outflowarr, fescarr, timearr,xcenter, ycenter, zcenter

def projectionPlot(Part, Cell, read, nout, xcenter1, ycenter1, zcenter1, wid, height, depth, orientation, field, label, H, ax1):

    Part1 = Part(read, nout)
    Cell1 = Cell(read, nout, Part1)
    cm1 = plt.get_cmap('rainbow')

    a = GenerateArray(Cell1, xcenter1, ycenter1, zcenter1, wid, height, depth, orientation)
    ss1 = a.projectionPlot(Cell1, ax1, cm1, field, -3, 2)

    a.star_plot(Part1, ax1)
    xx = np.linspace(a.xwid*(-1)*a.mindx/1000,a.xwid*a.mindx/1000,3)
    yy = np.ones(3)*H
    ax1.plot(xx,yy/1000, c='k',ls='dashed',lw=3)
    ax1.text(a.xwid * (-0.8) * a.mindx / 1000, a.zwid * (-0.8) * a.mindx / 1000,
             '%s\ntime = %3.2f(Myr)' % (label,Part1.snaptime), color='khaki', fontsize=13)

    return ss1


def plot2(inisnap, endsnap, read1, read2):

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 20

    numsnap = endsnap - inisnap + 1

    for n in range(numsnap):
        nout = inisnap + n
        if not os.path.isfile(read1 + '/SAVE/cell_%05d.sav' % (nout)):
            print(read1 + '/SAVE/cell_%05d.sav' % (nout))
            continue
        if not os.path.isfile(read2 + '/SAVE/cell_%05d.sav' % (nout)):
            print(read2 + '/SAVE/cell_%05d.sav' % (nout))
            continue

        fig = plt.figure(figsize=(12, 9), dpi=144)

        ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.2])
        ax2 = fig.add_axes([0.1, 0.4, 0.2, 0.2])
        ax5 = fig.add_axes([0.1, 0.7, 0.2, 0.2])
        ax3 = fig.add_axes([0.4, 0.1, 0.25, 0.75])
        ax4 = fig.add_axes([0.65, 0.1, 0.25, 0.75])
        cax1 = fig.add_axes([0.9, 0.1, 0.02, 0.75])

        if n==0:

            H1, outflow1, fesc1, time1, xcenter1, ycenter1, zcenter1 = main2(Part, Cell,read1,nout,2000, 1500, 3000,500,np.array([]),np.array([]),np.array([]),np.array([]))
            H2, outflow2, fesc2, time2, xcenter2, ycenter2, zcenter2 = main2(Part, GasrichCell,read2,nout,5000, 3000, 6000,500,np.array([]),np.array([]),np.array([]),np.array([]))
        else:
            H1, outflow1, fesc1, time1, xcenter1, ycenter1, zcenter1 = main2(Part, Cell, read1, nout, 2000, 1500, 3000, 500,
                                                                      H1,outflow1,time1,fesc1)
            H2, outflow2, fesc2, time2, xcenter2, ycenter2, zcenter2 = main2(Part, GasrichCell, read2, nout, 5000, 3000, 6000,
                                                                      500, H2,outflow2,time2, fesc2)

        ax1.plot(time1,H1,c='k',label='ref')
        ax1.plot(time2,H2,c='gold',label='gas_rich')
        ax1.scatter(time1[-1],H1[-1],c='cyan',s=5)
        ax1.scatter(time2[-1],H2[-1],c='cyan',s=5)
        print(time1[-1], H1[-1])
        ax2.plot(time1, outflow1, c='k', label='ref')
        ax2.plot(time2, outflow2, c='gold', label='gas_rich')
        ax2.scatter(time1[-1], outflow1[-1], c='cyan', s=5)
        ax2.scatter(time2[-1], outflow2[-1], c='cyan', s=5)

        ax5.plot(time1, fesc1, c='k', label='ref')
        ax5.plot(time2, fesc2, c='gold', label='gas_rich')
        ax5.scatter(time1[-1], fesc1[-1], c='cyan', s=5)
        ax5.scatter(time2[-1], fesc2[-1], c='cyan', s=5)

        ss1 = projectionPlot(Part, Cell,read1, nout,xcenter1, ycenter1, zcenter1, 150, 450, 300, 'xz', 'nH', 'ref', H1[-1],ax3 )
        ss2 = projectionPlot(Part, GasrichCell,read2, nout,xcenter2, ycenter2, zcenter2, 150, 450, 300, 'xz', 'nH', 'gas_rich',H2[-1],ax4 )
        ss2 = projectionPlot(Part, Cellfromdat,read2, nout,xcenter2, ycenter2, zcenter2, 150, 450, 300, 'xz', 'nH', 'gas_rich',H2[-1],ax4 )

        cm1 = plt.get_cmap('rainbow')
        cbar = plt.colorbar(ss1, cax=cax1, cmap=cm1)
        cbar.set_label('log(nH)')
        ax1.legend()
        ax2.legend()
        ax5.legend()
        ax1.set_xlabel('time(Myr)')
        ax1.set_ylabel('scale height (pc)')
        ax2.set_xlabel('time(Myr)')
        ax2.set_ylabel('outflow(km/s)')
        ax3.set_xlabel('X(kpc)')
        ax4.set_xlabel('X(kpc)')
        ax3.set_ylabel('Y(kpc)')
        ax4.set_yticks([])
        ax1.set_xlim(90,200)
        ax2.set_xlim(90,200)
        ax5.set_xlim(90,200)
        ax1.set_ylim(200,1300)
        ax2.set_ylim(0,100)
        ax5.set_ylim(0,1)



        plt.savefig('/Volumes/THYoo/kisti/scale_height/compare%05d.png'%nout)
        plt.close()
"""
plot1(True)
#plot2(100,200,read_new_01Zsun,read_new_gasrich)