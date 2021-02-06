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
    x2, y2, z2 = CoM_pre(Part1,Cell1,rgrid1,1.75e9,x1,y1,z1,True)
    #print(x2,y2,z2)
    return x2, y2, z2
def scaleheight(Cell, xcenter, ycenter, xyrange, zrange,zvalue):
    zcenter = np.sum(Cell.z * Cell.nH * Cell.dx ** 3) / np.sum(Cell.nH * Cell.dx ** 3)
    cutindex = np.where((np.abs(Cell.x - xcenter) < xyrange) & (np.abs(Cell.y - ycenter) < xyrange) & (
                np.abs(Cell.z - zcenter) < zrange))
    x = Cell.x[cutindex]
    y = Cell.y[cutindex]
    z = Cell.z[cutindex]
    dx = Cell.dx[cutindex]
    nH = Cell.nH[cutindex]

    vz = Cell.vz[cutindex]
    den = Cell.den[cutindex]

    zcenter = np.sum(z*nH*dx**3)/np.sum(nH*dx**3)

    # outflow
    """
    upp = np.where((z - zcenter > 0) & (vz > 0) & (z - dx < zvalue) & (z + dx >= zvalue))
    low = np.where((z - zcenter < 0) & (vz < 0) & (z - dx < zvalue) & (z + dx >= zvalue))

    outflow_up = np.sum(vz[upp] * den[upp] * dx[upp] ** 2)
    outflow_lo = np.sum(vz[low] * den[low] * dx[low] ** 2)

    outflow = (outflow_up + np.abs(outflow_lo)) / 2.989e33 * 365 * 3600 * 24
    """


    return np.sqrt(np.sum((z-zcenter)**2*nH*(dx)**3)/np.sum(nH*(dx)**3)), zcenter

def MultiFesc2(ax1, dir, inisnap, endsnap, rjusarr, Part, Cell, color, label, new, initr, xyrange, zrange, zvalue, load):
        numsnap = endsnap - inisnap + 1

        if load == False:
            fescarr = np.zeros(len(rjusarr))
            fescarr2= np.zeros(len(rjusarr))
            fescarr3= np.zeros(len(rjusarr))

            print(rjusarr[:-1])
            bin = (rjusarr[:-1]+rjusarr[1:])/2
            numsum=0
            for i in range(numsnap):
                nout = inisnap + i
                print(nout)
                if not os.path.isfile(dir + 'SAVE/part_%05d.sav' % nout):
                    print(dir + 'SAVE/part_%05d.sav' % nout)
                    continue
                if not os.path.isfile(dir + 'clump3/clump_%05d.txt' % nout):
                    print(dir + 'clump3/clump_%05d.txt' % nout)
                    continue

                Part1 = Part(dir, nout)
                Cell1 = Cell(dir, nout, Part1)
                """
                Clump1 = Clump(dir, nout, Part1)
                nclump = len(Clump1.massclump)
                clumpindexarr = np.array([])
                for n in range(nclump):
                    rr = np.sqrt((Clump1.xclump[n] - Part1.xp[0]) ** 2 + (Clump1.yclump[n] - Part1.xp[1]) ** 2 + (
                            Clump1.zclump[n] - Part1.xp[2]) ** 2)
                    clumpindex = np.where((rr < Clump1.rclump[n]))[0]
                    clumpindexarr = np.append(clumpindexarr, clumpindex)
                noclumpindexarr = np.delete(np.arange(len(Part1.starage)),clumpindexarr)
                
            
                """
                #xcenter, ycenter = CoM2(Part1, Cell1, 'gas', initr)
                xcenter, ycenter, zcenter = CoM_main(Part1, Cell1)
                """
                H,  zcenter = scaleheight(Cell1, xcenter, ycenter, xyrange, zrange, zvalue)
                ind = np.where(np.abs(Part1.xp[2]-zcenter)<H)
                ind2 = np.where(np.abs(Part1.xp[2]-zcenter)>H)

                print(ind)
                """
                for j in range(len(rjusarr)):
                    if j > 0:
                        prev = fescD

                    break2=0


                    for k in range(len(rjusarr)):
                        if new==False:
                            filepath='ray_nside4_%3.2f/ray_%05d.dat' % (rjusarr[k], nout)
                        if new==True:
                            filepath='rjus_%3.2f/ray_%05d.dat' % (rjusarr[k], nout)

                        if not os.path.isfile(dir + filepath):
                            print(dir + filepath)
                            break2=1
                            break

                    if break2==1:
                        break

                    if new == False:
                        filepath = 'ray_nside4_%3.2f/ray_%05d.dat' % (rjusarr[j], nout)
                    else:
                        filepath = 'rjus_%3.2f/ray_%05d.dat' % (rjusarr[j], nout)

                    dat = FortranFile(dir + filepath, 'r')

                    if new==False:
                        npart, nwave2 = dat.read_ints()
                        wave = dat.read_reals(dtype=np.double)
                        sed_intr = dat.read_reals(dtype=np.double)
                        sed_attH = dat.read_reals(dtype=np.double)
                        sed_attD = dat.read_reals(dtype=np.double)
                        npixel = dat.read_ints()
                        tp = dat.read_reals(dtype='float32')
                        fescH = dat.read_reals(dtype='float32')
                        fescD = dat.read_reals(dtype='float32')

                        photonr = dat.read_reals(dtype=np.double)
                    else:

                        npart, nwave2, version = dat.read_ints()
                        wave = dat.read_reals(dtype=np.double)
                        sed_intr = dat.read_reals(dtype=np.double)
                        sed_attHHe = dat.read_reals(dtype=np.double)
                        sed_attHHeD = dat.read_reals(dtype=np.double)
                        sed_attHHI = dat.read_reals(dtype=np.double)
                        sed_attHH2 = dat.read_reals(dtype=np.double)
                        sed_attHHe = dat.read_reals(dtype=np.double)
                        sed_attD = dat.read_reals(dtype=np.double)

                        npixel = dat.read_ints()
                        tp = dat.read_reals(dtype='float32')
                        fescH = dat.read_reals(dtype='float32')
                        fescD = dat.read_reals(dtype='float32')
                        photonr = dat.read_reals(dtype=np.double)

                    if len(photonr)!=len(Part1.starage):
                        print(len(photonr))
                        print(len(Part1.starage))
                        continue
                    #clumpindexarr = np.unique(clumpindexarr.astype(int))
                    #noclumpindexarr = np.unique(noclumpindexarr.astype(int))

                    fesc = np.sum(fescD * photonr) / np.sum(photonr)
                    #fesc1 = np.sum(fescD[ind] * photonr[ind]) / np.sum(photonr[ind])
                    #fesc2 = np.sum(fescD[ind2] * photonr[ind2]) / np.sum(photonr[ind2])

                    if np.isnan(fesc)==False:
                        fescarr[j] = fescarr[j] + fesc
                    #if np.isnan(fesc1) == False:
                    #    fescarr2[j] = fescarr2[j] + fesc1
                    #if np.isnan(fesc2) ==False:
                    #    fescarr3[j] = fescarr3[j] + fesc2

                    #fescnoclumparr[j] = fescnoclumparr[j] + fesc2



                if break2==0:
                    numsum = numsum + 1
            fescarr = fescarr / numsum
            #fescarr2 = fescarr2/ numsum
            #fescarr3 = fescarr3/ numsum

            np.savetxt(dir+'fescarr_multi.dat',fescarr,newline='\n', delimiter=' ')
            #np.savetxt(dir+'fescarr_multi2.dat',fescarr2,newline='\n', delimiter=' ')
            #np.savetxt(dir+'fescarr_multi3.dat',fescarr3,newline='\n', delimiter=' ')

        else:
            fescarr = np.loadtxt(dir+'fescarr_multi.dat')
            #fescarr2 = np.loadtxt(dir+'fescarr_multi2.dat')
            #fescarr3 = np.loadtxt(dir+'fescarr_multi3.dat')
        print(fescarr)
        #print(fescarr2)
        #print(fescarr3)

        #fescclumparr = fescclumparr / numsum
        #fescnoclumparr = fescnoclumparr / numsum

        ax1.plot(rjusarr, fescarr, color=color, marker='o',label=label,lw=3)
        #print(fescarr,fescarr2,fescarr3)
        #ax1.plot(rjusarr, fescarr2, color=color, marker='x',ls='dashed',lw=2)


        #ax1.plot(rjusarr, fescarr3, color=color, marker='^', ls='dotted', lw=2)


class Fesc_rjus():
    def __init__(self,read,nout,rjus):

        dat = FortranFile(read+'ray_nside4_%3.2f/ray_%05d.dat' % (rjus, nout), 'r')
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



rjusarr = np.array([0.04, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 4.00])



plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20

fig = plt.figure(figsize=(9,7))
ax1 = fig.add_axes([0.15,0.1,0.8,0.8])
#ax2 = fig.add_axes([0.1,0.3,0.8,0.2])
#ax3 = fig.add_axes([0.1,0.5,0.8,0.2])
#ax4 = fig.add_axes([0.1,0.7,0.8,0.2])

MultiFesc2(ax1,read_new_1Zsun_highSN_new, 46, 225, rjusarr, Part, Cell, 'r', 'G9_1Zsun_SNboost',True,2000,1500,3000,500,True)
MultiFesc2(ax1,read_new_01Zsun, 100, 200, rjusarr, Part, Cell, 'gray', 'G9_01Zsun',,2000,1500,3000,500,True)
MultiFesc2(ax1,read_new_gasrich, 100, 158, rjusarr, Part, Cellfromdat,'b', 'G9_01Zsun_gasrich',False,5000, 3000, 6000,500,True)
#MultiFesc2(ax1,read_new_1Zsun, 100, 267, rjusarr, Part, Cell, 'lightseagreen', 'G9_1Zsun',True,2000,1500,3000,500,True)

#arr3,arr4= Makefescarr2(rjusarr, read_new_gasrich, 101, 150, Part, Cell, 100)

#print('a')
#arr11= Makefescarr2(rjusarr, read_new_01Zsun, 101, 150, Part, Cell, 250)
#print('b')
#arr21= Makefescarr2(rjusarr, read_new_gasrich, 101, 150, Part, GasrichCell, 250)
#print('c')
#ax1.plot(rjusarr, arr2, c=col2, label=lab2, marker='o')
#ax1.plot(rjusarr, arr3, c='orange', label='gas_rich (d<0.5kpc)', marker='o', ls='dashed')
#ax1.plot(np.log10(rjusarr), np.log10(arr4), c='cyan', label='gas_rich (d>0.3kpc)',marker='o')
#ax1.plot(rjusarr, arr11, c='purple', label='ref(z<250)', marker='x')
#ax1.plot(rjusarr, arr21, c='orange', label='gas_rich(z<250)', marker='x')
#ax2.plot(rjusarr, arr1-arr2,c='purple', marker='o')
#print(arr1-arr2)
#ax2.plot(rjusarr, np.zeros(len(rjusarr)),c='k',ls='dashed')

ax1.set_xlabel('$r;$ distance from stars (kpc)')
ax1.set_ylabel('$f_{esc}(r)$')
#ax2.set_ylabel('$f_{esc}(<r)$')
#ax3.set_ylabel('$f_{esc}(<r)$')
#ax4.set_ylabel('$f_{esc}(<r)$')

ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax1.set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
ax1.set_xticks([0.1,0.5,1,2,4])
ax1.set_ylim(0.007,1)
#ax2.set_xscale('log')
#ax2.set_yscale('log')
#ax3.set_yscale('log')
#ax4.set_yscale('log')

#ax3.set_xscale('log')
#ax4.set_xscale('log')

#ax2.set_xticks([]clump_pop_fesc_scatter.py
#ax3.set_xticks([])
#ax4.set_xticks([])
#plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig9.pdf')
plt.show()