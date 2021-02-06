import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import os.path
import matplotlib.colors as colors
from multiprocessing import Pool, Process, Queue
from matplotlib.lines import Line2D

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
        #self.vx = celldata.cell[0][4][1] * Part.unit_l / 4.70430e14
        #self.vy = celldata.cell[0][4][2] * Part.unit_l / 4.70430e14
        self.den = celldata.cell[0][4][0] *Part.unit_d
        self.dx = celldata.cell.dx[0] * Part.boxpc
        self.mindx = np.min(celldata.cell.dx[0])
        self.nHI = self.nH * celldata.cell[0][4][7]
        nHII = self.nH * celldata.cell[0][4][8]
        nH2 = self.nH * (1 - celldata.cell[0][4][7] - celldata.cell[0][4][8])/2
        YY= 0.24/(1-0.24)/4
        nHeII = self.nH * YY*celldata.cell[0][4][9]
        nHeIII = self.nH * YY*celldata.cell[0][4][10]
        nHeI = self.nH * YY*(1 - celldata.cell[0][4][9] - celldata.cell[0][4][10])
        ne = nHII + nHeII + nHeIII *2
        ntot = self.nHI + nHII + nHeI + nHeII + nHeIII + ne + nH2
        mu = celldata.cell[0][4][0] * Part.unit_d / 1.66e-24 / ntot
        self.m = celldata.cell[0][4][0] *Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l *Part.unit_l *(celldata.cell.dx[0]*Part.boxlen)**3

        self.T = celldata.cell[0][4][5]/celldata.cell[0][4][0] * 517534.72 * mu
        kB = 1.38e-16
        mproton = 1.66e-24
        self.P = self.den * self.T /mu /mproton
        self.vz = celldata.cell[0][4][3] * Part.unit_l / 4.70430e14
        lam = 315614 / self.T
        f = 1+(lam/2.74)**0.407
        alpha_B = 2.753e-14 * lam**1.5 /f**2.242
        self.rec_rate = alpha_B *ne *nHII

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

def halfmassrad(Part,xcen,ycen,zcen,rbin, zrange):
    totstarmass = np.sum(Part.mp0)
    #cellrr = get2dr(Cell.x,Cell.y, xcen,ycen)
    #cellzdist = np.abs(Cell.z-zcen)
    partrr = get2dr(Part.xp[0],Part.xp[1],xcen,ycen)
    partzdist = np.abs(Part.xp[2]-zcen)
    starhalfmassrad=0; #cellhalfmassrad=0


    for i in range(len(rbin)):
        partind = np.where((partrr<rbin[i])&(partzdist<zrange))
        if np.sum(Part.mp0[partind])>totstarmass/2:
            starhalfmassrad = rbin[i]
            break

    #for i in range(len(rbin)):
    #    cellind = np.where((cellrr<rbin[i])&(cellzdist<zrange))
    #    if np.sum(Cell.m[cellind])>diskmass/2:
    #        cellhalfmassrad = rbin[i]
    #        break

    #if cellhalfmassrad ==0:
    #    raise ValueError('what the')
    #print(starhalfmassrad, cellhalfmassrad)
    return starhalfmassrad
"""
def decompose(Cell, xcenter, ycenter, zcenter, xy, zfwd,  positive):
    pc = 3.08e18
    mindx = np.min(Cell.dx)

    maxgrid = int(np.log2(np.max(Cell.dx) / mindx))

    xind = Cell.x / mindx - 0.5
    yind = Cell.y / mindx - 0.5
    zind = Cell.z / mindx - 0.5

    xwid = xy
    ywid = xy
    zwid = int(zfwd/2)
    # print(xwid, ywid, zwid)
    xfwd = 2 * int(xwid)
    yfwd = 2 * int(ywid)
    zfwd = 2 * int(zwid)

    xini = int(xcenter / mindx) - xwid + 1
    yini = int(ycenter / mindx) - ywid + 1
    zini = int(zcenter / mindx) - zwid + 1
    xfin = int(xcenter / mindx) + xwid
    yfin = int(ycenter / mindx) + ywid
    zfin = int(zcenter / mindx) + zwid

    leafcell = np.zeros((xfwd, yfwd, zfwd))

    ind_1 = np.where((Cell.dx == mindx) & (xind >= xini) & (xind <= xfin)
                     & (yind >= yini) & (yind <= yfin) & (zind >= zini) & (zind <= zfin))[0]

    leafcell[xind[ind_1].astype(int) - int(xini), yind[ind_1].astype(int) - int(yini), zind[ind_1].astype(int) - int(zini)] = ind_1.astype(int)

    #print('leaf cells are allocated (n=%d)' % len(ind_1))

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
        #print(len(ind), len(ind) * (2 ** (n + 1)) ** 3)
        for a in range(2 ** (n + 1)):
            for b in range(2 ** (n + 1)):
                for c in range(2 ** (n + 1)):
                    xx = xind[ind] - xini + mul[a]
                    yy = yind[ind] - yini + mul[b]
                    zz = zind[ind] - zini + mul[c]
                    xyzind = np.where(
                        (xx >= 0) & (xx <= xfwd - 1) & (yy >= 0) & (yy <= yfwd - 1) & (zz >= 0) & (zz <= zfwd - 1))[0]
                    leafcell[xx[xyzind].astype(int), yy[xyzind].astype(int), zz[xyzind].astype(int)] = ind[xyzind]
                    ##    print('zerocell')
                    nnn = nnn + len(xyzind)
        nn = nn + nnn
        #print('level %d grids are allocated(n = %d)' % (n + 2, nnn))
        if len(ind_1) + nn == xfwd * yfwd * zfwd:
            break
    #nonzero = len(np.where(leafcell != 0)[0])
    #print('total allocated cells are = ', len(ind_1) + nn)
    #print('total box cells are = ', xfwd * yfwd * zfwd)
    #print('total non zero cell in the box are = ', nonzero)

    if len(ind_1) + nn != xfwd * yfwd * zfwd:
        raise ValueError("allocation failure")
    #print(np.min(Cell.nHI[leafcell[:,:,:].astype(int)]),np.max(Cell.nHI[leafcell[:,:,:].astype(int)]))
    if positive==True:
        edgeind = leafcell[:,:,-1].astype(int)
        outflowind = np.where(Cell.vz[edgeind.ravel()] > 0)

    else:
        edgeind = leafcell[:,:,0].astype(int)
        outflowind = np.where(Cell.vz[edgeind.ravel()] < 0)

    frag = Cell.dx**3/mindx**3


    outflow = np.sum(Cell.vz[edgeind.ravel()[outflowind]]*Cell.m[edgeind.ravel()[outflowind]]/frag[edgeind.ravel()[outflowind]]/(mindx*3.08e18)*365*24*3600)
    leafcell2 =leafcell.ravel()
    #volfrac = np.zeros(len(th))

   
    for l in range(len(th)):
        lowind = np.where(Cell.nHI[leafcell2.astype(int)]<th[l])[0]
        volfrac[l] = len(lowind)/(xfwd*yfwd*zfwd)
        #print(len(lowind),xfwd*yfwd*zfwd,len(leafcell2))
  

    HIden = np.mean(Cell.nHI[leafcell2.astype(int)])
    gasden = np.mean(Cell.nH[leafcell2.astype(int)])
    print(outflow)

    recrate = np.sum(Cell.rec_rate[leafcell2.astype(int)]*Cell.dx[leafcell2.astype(int)]**3)/np.sum(Cell.dx[leafcell2.astype(int)]**3)





    return recrate, HIden, gasden, outflow
"""
def decompose_new(Cell, xcenter, ycenter, zcenter, xy, zfwd,  positive):

    ind_inner = np.where((Cell.z+Cell.dx/2<zcenter+zfwd/2)&(Cell.z-Cell.dx/2>zcenter-zfwd/2)&(get2dr(Cell.x,Cell.y,xcenter,ycenter)<xy))
    ind_up = np.where((Cell.z+Cell.dx/2>zcenter+zfwd/2)&(Cell.z-Cell.dx/2<zcenter+zfwd/2)&(get2dr(Cell.x,Cell.y,xcenter,ycenter)<xy))
    ind_down = np.where((Cell.z+Cell.dx/2>zcenter-zfwd/2)&(Cell.z-Cell.dx/2<zcenter-zfwd/2)&(get2dr(Cell.x,Cell.y,xcenter,ycenter)<xy))

    marginvol_up = (zcenter+zfwd/2-(Cell.z-Cell.dx/2))*Cell.dx**2
    marginvol_down = (Cell.z+Cell.dx/2-(zcenter-zfwd/2))*Cell.dx**2

   # print(ind_down)

    #print(Cell.dx[ind_inner]**3, marginvol_up[ind_up],np.sum(marginvol_down[ind_down]))
    if len(ind_inner[0])>0:
        nHI_inner = Cell.nHI[ind_inner]
        nH_inner = Cell.nH[ind_inner]
        dx_inner = Cell.dx[ind_inner]
        recrate_inner = Cell.rec_rate[ind_inner]
    else:
        nHI_inner=0
        nH_inner=0
        dx_inner=0
        recrate_inner=0

    if len(ind_up[0])>0:
        vol_up = marginvol_up[ind_up]
        nHI_up = Cell.nHI[ind_up]
        nH_up = Cell.nH[ind_up]
        recrate_up = Cell.rec_rate[ind_up]
        vz_up = Cell.vz[ind_up]
    else:
        vol_up=0
        nHI_up=0
        nH_up=0
        recrate_up=0

    if len(ind_down[0])>0:
        vol_down = marginvol_down[ind_down]
        nHI_down = Cell.nHI[ind_down]
        nH_down=Cell.nH[ind_down]
        recrate_down=Cell.rec_rate[ind_down]
    else:
        vol_down=0
        nHI_down=0
        nH_down=0
        recrate_down=0

    volsum = np.sum(dx_inner**3)+np.sum(vol_up)+np.sum(vol_down)

    if volsum >0:
        HIden = (np.sum(nHI_inner*dx_inner**3)+np.sum(nHI_up*vol_up)+np.sum(nHI_down*vol_down))/volsum
        gasden = (np.sum(nH_inner*dx_inner**3)+np.sum(nH_up*vol_up)+np.sum(nH_down*vol_down))/volsum
        recrate = (np.sum(recrate_inner*dx_inner**3)+np.sum(recrate_up*vol_up)+np.sum(recrate_down*vol_down))/volsum
    else:
        HIden=0;gasden=0;recrate=0

    mindx = np.min(Cell.dx)
    if positive ==True:
        if len(ind_up[0])>0:
            of_ind= np.where(Cell.vz[ind_up]>0)

            outflow = np.sum(Cell.vz[ind_up][of_ind] * Cell.den[ind_up][of_ind] * (Cell.dx[ind_up][of_ind]*3.08e18)**2 * 365 * 24 * 3600/1.989e33)
        else:
            outflow=0
    else:
        if len(ind_down[0])>0:

            of_ind= np.where(Cell.vz[ind_down]<0)

            outflow = np.sum(Cell.vz[ind_down][of_ind] * Cell.den[ind_down][of_ind] * (Cell.dx[ind_down][of_ind]*3.08e18)**2 * 365 * 24 * 3600/1.989e33)
        else:
            outflow=0
    #print(recrate, HIden, gasden, outflow)

    return recrate, HIden, gasden, outflow

def jonghab(ax1,ax2,ax3,ax4,read, inisnap, endsnap, zbin, zselect, zval, load, color, label): # zbin indicates the center of each z bin, not an edge.
    zstep = np.abs(zbin[1] - zbin[0])

    if load==False:
        numsnap = endsnap - inisnap + 1
        n_bin = len(zselect)

        fescarr = np.array([])
        timearr = np.array([])
        for i in range(numsnap):
            nout = i + inisnap
            print(nout)
            if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
                print(read + '/SAVE/cell_%05d.sav' % (nout))
                continue
            if not os.path.isfile(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout)):
                print(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout))
                continue
            Part1 = Part(read, nout)
            Cell1 = Cell(read, nout, Part1)
            Fesc1 = Fesc_new8(read, nout)

            xcenter, ycenter, zcenter = CoM_main(Part1, Cell1)
            starhalfmassrad = halfmassrad(Part1, xcenter, ycenter, zcenter,
                                                          np.linspace(100, 7000, num=70), zval)

            zcenter2ind = np.where(np.abs(Cell1.z - zcenter) < zval)
            zcenter2 = np.sum(Cell1.z[zcenter2ind] * Cell1.nH[zcenter2ind] * Cell1.dx[zcenter2ind] ** 3) / np.sum(
                Cell1.nH[zcenter2ind] * Cell1.dx[zcenter2ind] ** 3)
          #  print(starhalfmassrad)
            xy = int(starhalfmassrad/np.min(Cell1.dx))
            recratearr = np.zeros(int(n_bin))
            HIdenarr = np.zeros(int(n_bin))
            gasdenarr = np.zeros(int(n_bin))
            outflowarr = np.zeros(int(n_bin))

            for j in range(len(zselect)):
                zz = zcenter2 + zbin[zselect[j]]
                #print(xy, int(zstep/np.min(Cell1.dx)))

                recrate, HIden, gasden, outflow = decompose_new(Cell1,xcenter,ycenter,zz, xy, zstep,True)

                recratearr[j]=recrate
                HIdenarr[j]=HIden
                gasdenarr[j]=gasden
                outflowarr[j]=outflow
                  #  print(j)
                zz = zcenter2 - zbin[zselect[j]]

                recrate, HIden, gasden, outflow = decompose_new(Cell1,xcenter,ycenter,zz, xy, zstep,False)


                recratearr[j] = recratearr[j] + recrate
                HIdenarr[j] = HIdenarr[j] + HIden
                gasdenarr[j] = gasdenarr[j] + gasden
                outflowarr[j] = outflowarr[j] - outflow #negative outflow
                 #   print(7-j)
                #starind = np.where((Part1.xp[2]>=zz-zbinwidth/2)&(Part1.xp[2]<zz+zbinwidth/2)&(get2dr(Part1.xp[0],Part1.xp[1],xcenter,ycenter)<2000))
                #starden = np.sum(Part1.mp0[starind[0]])/(2000**2*np.pi*zbinwidth)
                #stardenarr[j]=stardenarr[j]+starden
            print(recratearr)
            recratearr = recratearr / 2
            HIdenarr = HIdenarr / 2
            gasdenarr = gasdenarr / 2
            outflowarr = outflowarr / 2

            if i ==0:
                recratearr2 = recratearr
                HIdenarr2 = HIdenarr
                gasdenarr2 = gasdenarr
                outflowarr2 = outflowarr
            else:
                recratearr2 = np.vstack((recratearr2, recratearr))
                HIdenarr2 = np.vstack((HIdenarr2, HIdenarr))
                gasdenarr2 = np.vstack((gasdenarr2, gasdenarr))
                outflowarr2 = np.vstack((outflowarr2, outflowarr))
            timearr=np.append(timearr,Part1.snaptime)
            fescarr=np.append(fescarr,Fesc1.fesc)

        np.savetxt(read+'gasdenarr2.dat',gasdenarr2,newline='\n',delimiter=' ')
        np.savetxt(read+'HIdenarr2.dat',HIdenarr2, newline='\n',delimiter=' ')
        np.savetxt(read+'outflowarr2.dat',outflowarr2, newline='\n',delimiter=' ')
        np.savetxt(read+'recratearr2.dat',recratearr2, newline='\n',delimiter=' ')
        #np.savetxt(read+'stardenarr2.dat',stardenarr2, newline='\n',delimiter=' ')
        np.savetxt(read+'timearr_jonghab.dat',timearr, newline='\n',delimiter=' ')
        np.savetxt(read+'fescarr_jonghab.dat',fescarr, newline='\n',delimiter=' ')
    else:

        gasdenarr2 = np.loadtxt(read+'gasdenarr2.dat')
        HIdenarr2 = np.loadtxt(read+'HIdenarr2.dat')
        outflowarr2 = np.loadtxt(read+'outflowarr2.dat')
        recratearr2 = np.loadtxt(read+'recratearr2.dat')
        timearr = np.loadtxt(read+'timearr_jonghab.dat')
        fescarr = np.loadtxt(read+'fescarr_jonghab.dat')
    print(outflowarr2)
    print(outflowarr2.size)

    xx = zbin[zselect]
    outflowarr3 = np.mean(outflowarr2, axis=0)
    HIdenarr3 = np.mean(HIdenarr2, axis=0)
    gasdenarr3 = np.mean(gasdenarr2, axis=0)
    recratearr3 = np.mean(recratearr2,axis=0)

    ax2.plot(xx[:-1], gasdenarr3[:-1], color=color, marker='s')
    ax3.plot(xx[:-1], HIdenarr3[:-1], color=color, ls='dashed',marker='s')
    ax1.plot(xx[:-1], outflowarr3[:-1], color=color,  label=label,marker='s')
    # ax4.plot(zz, stardenarr2[::-1], color=color, ls='solid', label=label,marker='s')

    ax4.plot(xx[:-1], recratearr3[:-1], color=color,marker='s')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax4.set_xscale('log')
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    ax1.set_xlabel('$d_{star}$')
    ax1.set_ylabel('$dM/dt$ $(M_\odot/yr)$')
    ax2.set_ylabel('nH $(cm^{-3})$')
    ax3.set_ylabel('nHI $(cm^{-3})$')
    ax4.set_ylabel(r'$dN_{HI}/dt$ $(cm^{-3}\cdot s^{-1})$')


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 13

fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_axes([0.18, 0.15, 0.8, 0.2])
ax2 = fig.add_axes([0.18, 0.35, 0.8, 0.2])
ax3 = fig.add_axes([0.18, 0.55, 0.8, 0.2])
ax4 = fig.add_axes([0.18, 0.75, 0.8, 0.2])

#zbin = np.linspace(-4000, 4000, num=16)

#jonghab(ax1,ax2,ax3,read_new_01Zsun,130,331,th,zbin,'olive','G9_01Zsun')
markerset=['o','s','p','x','D','*']
rec1 = jonghab(ax1,ax2,ax3,ax4,read_new_1Zsun,130,481,np.linspace(100,8000,num=80),[0,4,9,19,39,-1], 4000, True, 'lightseagreen','G9_highZ')
rec2 = jonghab(ax1,ax2,ax3,ax4,read_new_1Zsun_highSN_new,30,380,np.linspace(100,8000,num=80),[0,4,9,19,39,-1], 4000, True, 'r','G9_highZ_SN5')

#print(rec1/rec2)

ax3.set_yscale('log')
ax1.set_yscale('log')
ax4.set_yscale('log')

# ax1.legend()
#ax1.set_yscale('log')
ax2.set_yscale('log')

#ax3.set_yscale('log')

#legend_elements1 = [Line2D([0], [0], color='k',  label='nH'),

#                   Line2D([0], [0], color='k', label='nHI',ls='dashed')]
"""
legend_elements2 = [Line2D([0], [0], color='k',  label='$nHI_{th}$=0.001'),
                   Line2D([0], [0], color='k', label='$nHI_{th}$=0.01',
                         ls='dashed'),Line2D([0], [0], color='k', label='$nHI_{th}$=0.1',
                         ls='dotted')]
"""
#ax1.legend()
#ax2.legend(handles=legend_elements1)
#ax3.legend()
#ax4.legend()
#ax4.set_yscale('log')
#ax4.set_xticks([])
#ax4.set_ylabel(r'$\rho_* (M_\odot/pc^3)$')
plt.show()
#plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/figa2.pdf')
#jonghab(ax1,ax2,ax3,read_new_gasrich,30,331,th,zbin,'b','G9_01Zsun_gasrich',False)
#ax1.legend()
#plt.show()
