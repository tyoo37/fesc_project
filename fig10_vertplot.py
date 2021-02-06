import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import os.path
from multiprocessing import Process
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
        self.xHI =celldata.cell[0][4][7]
        self.nHI = self.nH * self.xHI
        self.den = celldata.cell[0][4][0] *Part.unit_d
        nHII = self.nH * celldata.cell[0][4][8]
        nH2 = self.nH * (1 - celldata.cell[0][4][7] - celldata.cell[0][4][8]) / 2
        YY = 0.24 / (1 - 0.24) / 4
        nHeII = self.nH * YY * celldata.cell[0][4][9]
        nHeIII = self.nH * YY * celldata.cell[0][4][10]
        nHeI = self.nH * YY * (1 - celldata.cell[0][4][9] - celldata.cell[0][4][10])
        ne = nHII + nHeII + nHeIII *2
        ntot = self.nHI + nHII + nHeI + nHeII + nHeIII + ne + nH2
        mu = celldata.cell[0][4][0] * Part.unit_d / 1.66e-24 / ntot
        self.T = celldata.cell[0][4][5]/celldata.cell[0][4][0] * 517534.72 * mu

        lam = 315614 / self.T
        f = 1 + (lam / 2.74) ** 0.407
        alpha_B = 2.753e-14 * lam ** 1.5 / f ** 2.242
        self.rec_rate = alpha_B * ne * nHII

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
        self.xHI = var[:,7]
        xHII = var[:,8]
        self.mHIH2 = self.m * (1-xHII)
        self.nHI = self.nH * self.xHI
        self.vx = var[:,1] * Part.unit_l /4.70430e14
        self.vy = var[:,2] * Part.unit_l /4.70430e14
        self.vz = var[:,3] * Part.unit_l /4.70430e14
        self.den = var[:,0] * Part.unit_d

        nHII = self.nH * var[:,8]
        nH2 = self.nH * (1 - var[:,7] - var[:,8]) / 2
        YY = 0.24 / (1 - 0.24) / 4
        nHeII = self.nH * YY * var[:,9]
        nHeIII = self.nH * YY * var[:,10]
        nHeI = self.nH * YY * (1 - var[:,9] - var[:,10])
        ne = nHII + nHeII + nHeIII * 2
        ntot = self.nHI + nHII + nHeI + nHeII + nHeIII + ne + nH2
        mu = var[:,0] * Part.unit_d / 1.66e-24 / ntot
        self.T = var[:,5] / var[:,0] * 517534.72 * mu

        lam = 315614 / self.T
        f = 1 + (lam / 2.74) ** 0.407
        alpha_B = 2.753e-14 * lam ** 1.5 / f ** 2.242
        self.rec_rate = alpha_B * ne * nHII

def getr(x, y, z, xcenter, ycenter, zcenter):
    return np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2 + (z - zcenter) ** 2)

def get2dr(x, y, xcen, ycen):
    return np.sqrt((x - xcen) ** 2 + (y - ycen) ** 2)

def getmass(marr, rarr, r):
    ind = np.where(rarr < r)
    return np.sum(marr[ind])
def getmass_zlim(marr, rarr, r,z,zcen,zlim):
    ind = np.where((rarr < r)&(np.abs(z-zcen)<zlim))
    return np.sum(marr[ind])
def mxsum(marr, xarr, ind):
    return np.sum(marr[ind] * xarr[ind])

def msum(marr, ind):
    return np.sum(marr[ind])

def simpleCoM(x, y, z, marr, rarr, r):
    ind = np.where(rarr < r)
    xx = np.sum(x[ind] * marr[ind]) / np.sum(marr[ind])
    yy = np.sum(y[ind] * marr[ind]) / np.sum(marr[ind])
    zz = np.sum(z[ind] * marr[ind]) / np.sum(marr[ind])

    return xx, yy, zz

# half-mass CoM
def CoM_first(Cell1, xcen,ycen,zcen,rbound,zbound):
    #rstar = getr(Part1.xp[0], Part1.xp[1], Part1.xp[2], xcen, ycen, zcen)
    #rpart = getr(Part1.dmxp[0], Part1.dmxp[1], Part1.dmxp[2], xcen, ycen, zcen)
    rcell = getr(Cell1.x, Cell1.y, Cell1.z, xcen, ycen, zcen)

    #starind = np.where(rstar<rbound)
    #partind = np.where(rpart<rbound)
    cellind = np.where(rcell<rbound)

    #masssum = msum(Part1.mp0,starind)+msum(Cell1.m,cellind)+msum(Part1.dmm,partind)
    return simpleCoM(Cell1.x[cellind],Cell1.y[cellind],Cell1.z[cellind],Cell.m[cellind],rcell[cellind],rbound)


def CoM_pre(Part1, Cell1, rgrid, totmass, xcen, ycen, zcen, gasonly,zlim):
    rstar = getr(Part1.xp[0], Part1.xp[1], Part1.xp[2], xcen, ycen, zcen)
    rpart = getr(Part1.dmxp[0], Part1.dmxp[1], Part1.dmxp[2], xcen, ycen, zcen)
    rcell = getr(Cell1.x, Cell1.y, Cell1.z, xcen, ycen, zcen)
    if gasonly == False:

        for i in range(len(rgrid)):
            mstar = getmass(Part1.mp0, rstar, rgrid[i])
            mpart = getmass(Part1.dmm, rpart, rgrid[i])
            mcell = getmass(Cell1.m, rcell, rgrid[i])
            summass = mstar + mpart + mcell
            if summass > totmass / 2:
                rrr = rgrid[i]
                break


        indstar = np.where(rstar < rrr)
        indpart = np.where(rpart < rrr)
        indcell = np.where(rcell < rrr)

        totalmx = mxsum(Part1.xp[0], Part1.mp0, indstar) + mxsum(Part1.dmxp[0], Part1.dmm, indpart) + mxsum(
            Cell1.x,
            Cell1.m,
            indcell)
        totalmy = mxsum(Part1.xp[1], Part1.mp0, indstar) + mxsum(Part1.dmxp[1], Part1.dmm, indpart) + mxsum(
            Cell1.y,
            Cell1.m,
            indcell)
        totalmz = mxsum(Part1.xp[2], Part1.mp0, indstar) + mxsum(Part1.dmxp[2], Part1.dmm, indpart) + mxsum(
            Cell1.z,
            Cell1.m,
            indcell)
        totalm = msum(Part1.mp0, indstar) + msum(Part1.dmm, indpart) + msum(Cell1.m, indcell)

    else:
        for i in range(len(rgrid)):
            mcell = getmass_zlim(Cell1.m, rcell, rgrid[i],Cell1.z,zcen,zlim)
            if mcell > totmass / 2:
                rrr = rgrid[i]
                break
        indcell = np.where((rcell < rrr)&(np.abs(Cell1.z-zcen)<zlim))

        totalmx = mxsum(Cell1.x, Cell1.m, indcell)
        totalmy = mxsum(Cell1.y, Cell1.m, indcell)
        totalmz = mxsum(Cell1.z, Cell1.m, indcell)

        totalm = msum(Cell1.m, indcell)
    xx = totalmx / totalm
    yy = totalmy / totalm
    zz = totalmz / totalm

    return xx, yy, zz, rrr

def CoM_main(Part1, Cell1, diskmass,zlim):

    rgrid1 = np.linspace(1000, 100000, num=100)
    rgrid2 = np.linspace(100,8000,80)
    boxcen = Part1.boxpc / 2
    x1, y1, z1 = CoM_first(Part1, Cell1,boxcen, boxcen, boxcen,100000,10000)
    for i in range(10):
        x1, y1, z1,hmr = CoM_pre(Part1, Cell1, rgrid1, diskmass, x1, y1, z1, True,zlim)
    # print(x2,y2,z2)
    return x1, y1, z1,hmr

def CoM_Main(Part1):
    start = time.time()

    xcen = np.sum(Part1.xp[0] * Part1.mp0) / np.sum(Part1.mp0)
    ycen = np.sum(Part1.xp[1] * Part1.mp0) / np.sum(Part1.mp0)
    zcen = np.sum(Part1.xp[2] * Part1.mp0) / np.sum(Part1.mp0)

    """
    rgrid1 = np.linspace(100,5000, num=50)
    rgrid2 = np.linspace(400,8000, num=39)
    
    for i in range(len(rgrid1)):
        mass = getmass_zlim(Part1.mp0,get2dr(Part1.xp[0],Part1.xp[1],xcen,ycen),rgrid1[i],Part1.xp[2],zcen,2000)
        if mass > np.sum(Part1.mp0)/2:
            hmr = rgrid1[i]
            break
    for j in range(len(rgrid2)):
        mass2 = getmass_zlim(Cell1.m,get2dr(Cell1.x,Cell1.y,xcen,ycen),rgrid2[j],Cell1.z,zcen,8000)
        if mass2 > diskmass/2:
            hmr2= rgrid2[j]
            break

    print('COM finished, t=%3.2f [s]'%(time.time()-start))
    """
    return xcen, ycen, zcen

def decompose_new(Cell, xcenter, ycenter, zcenter, xy, zfwd):

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
        xHI_inner = Cell.xHI[ind_inner]
        recreate_inner = Cell.rec_rate[ind_inner]
        T_inner = Cell.T[ind_inner]


    else:
        nHI_inner=0
        nH_inner=0
        dx_inner=0
        xHI_inner=0
        recreate_inner=0
        T_inner=0

    if len(ind_up[0])>0:
        vol_up = marginvol_up[ind_up]
        nHI_up = Cell.nHI[ind_up]
        nH_up = Cell.nH[ind_up]
        xHI_up = Cell.xHI[ind_up]
        recreate_up = Cell.rec_rate[ind_up]
        T_up = Cell.T[ind_up]

    else:
        vol_up=0
        nHI_up=0
        nH_up=0
        xHI_up=0
        recreate_up = 0
        T_up =0
    if len(ind_down[0])>0:
        vol_down = marginvol_down[ind_down]
        nHI_down = Cell.nHI[ind_down]
        nH_down=Cell.nH[ind_down]
        xHI_down=Cell.xHI[ind_down]
        recreate_down = Cell.rec_rate[ind_down]
        T_down = Cell.T[ind_down]
    else:
        vol_down=0
        nHI_down=0
        nH_down=0
        xHI_down=0
        recreate_down = 0
        T_down = 0

    volsum = np.sum(dx_inner**3)+np.sum(vol_up)+np.sum(vol_down)
    masssum = np.sum(dx_inner**3*nH_inner)+np.sum(vol_up*nH_up)+np.sum(vol_down*nH_down)

    if volsum >0:
        avHIden = (np.sum(nHI_inner*dx_inner**3)+np.sum(nHI_up*vol_up)+np.sum(nHI_down*vol_down))/volsum
        avHIden2 = (np.sum(nHI_inner**2*dx_inner**3)+np.sum(nHI_up**2*vol_up)+np.sum(nHI_down**2*vol_down))/volsum

        avgasden = (np.sum(nH_inner*dx_inner**3)+np.sum(nH_up*vol_up)+np.sum(nH_down*vol_down))/volsum
        avxHI = (np.sum(xHI_inner*dx_inner**3*nH_inner)+np.sum(xHI_up*vol_up*nH_up)+np.sum(xHI_down*vol_down*nH_down))/masssum
        avrec = (np.sum(recreate_inner*dx_inner**3)+np.sum(recreate_up*vol_up)+np.sum(recreate_down*vol_down))/volsum
        avT = (np.sum(T_inner*dx_inner**3*nH_inner)+np.sum(T_up*vol_up*nH_up)+np.sum(T_down*vol_down*nH_down))/masssum
    else:
        HIden=0;gasden=0;xHI=0

    cf = avHIden2 / avHIden**2


    return  avHIden, avgasden, avxHI, volsum, avrec,cf, avT

def getstarhmr(Part,xcen,ycen,zcen,rbin, zrange):
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

def getoutflow(Cell, zz,up, xcen, ycen, rad):
    if up == True:
        ind=np.where((Cell.z+Cell.dx/2>zz)&(Cell.z-Cell.dx/2<zz)&(Cell.vz>0)&(get2dr(Cell.x,Cell.y,xcen,ycen)<rad))
    else:
        ind = np.where((Cell.z + Cell.dx / 2 > zz) & (Cell.z - Cell.dx / 2 < zz) & (Cell.vz < 0) & (get2dr(Cell.x,Cell.y,xcen,ycen)<rad))
    outflow = np.sum(Cell.den[ind] * np.abs(Cell.vz[ind]) * (Cell.dx[ind] *3.08e18)**2/1.989e33*365*24*3600)


    return outflow

def stellarden(Part, zz, zbin,hmr,xcen,ycen):

    r2d =get2dr(Part.xp[0],Part.xp[1],xcen,ycen)
    ind = np.where((Part.xp[2]>zz-zbin/2)&(Part.xp[2]<zz+zbin/2)&(r2d<hmr))
    vol = np.pi*hmr**2*zbin
    return np.sum(Part.mp0[ind])/vol

def jonghab(Cell,read, inisnap, endsnap, zbin, zselect, zval, load, color, label,ls='solid'): # zbin indicates the center of each z bin, not an edge.
    zstep = np.abs(zbin[1] - zbin[0])
    #gascomarr = np.loadtxt(read+'gascomarr.dat')

    #xcenarr = gascomarr[:,0]
    #ycenarr = gascomarr[:,1]
    #zcenarr = gascomarr[:,2]

    #centimearr = gascomarr[:,4]
    #print(xcenarr, ycenarr, zcenarr, centimearr)
    n_bin = len(zselect)
    numsnap = endsnap - inisnap + 1

    if load==False:

        fescarr = np.array([])
        timearr = np.array([])

        arr = np.loadtxt(read + 'sh_hmr.dat')
        noutarr = arr[:, 0]


        for i in range(numsnap):
            nout = i + inisnap
            print(nout)

            if not os.path.isfile(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout)):
                print(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout))
                continue
            if not os.path.isfile(read+'/SAVE/part_%05d.sav'%(nout)):
                continue
            Part1 = Part(read, nout)
            if Cell==Cellfromdat:
                if not os.path.isfile(read + '/dat/cell_%05d.dat' % (nout)):
                    print(read + '/dat/cell_%05d.dat' % (nout))
                    continue
            else:
                if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
                    print(read + '/SAVE/cell_%05d.sav' % (nout))
                    continue
            Cell1 = Cell(read, nout, Part1)
            xcenter,ycenter,zcenter= CoM_Main(Part1)
            indnout = np.where(noutarr == nout)
            gassh = arr[indnout, 5]
            starhmr = arr[indnout, 2]

            starhmr = np.asscalar(starhmr)





            """
            timind = np.where(centimearr==Part1.snaptime)
            xcenter = np.asscalar(xcenarr[timind])
            ycenter = np.asscalar(ycenarr[timind])
            zcenter = np.asscalar(zcenarr[timind])
    
            #xcenter, ycenter, zcenter, gashmr = CoM_main(Part1, Cell1)
            starhmr= getstarhmr(Part1, xcenter, ycenter, zcenter,
                                                          np.linspace(100, 7000, num=70), zval)
            """
           # zcenter2ind = np.where(np.abs(Cell1.z - zcenter) < zval)
           # zcenter2 = np.sum(Cell1.z[zcenter2ind] * Cell1.nH[zcenter2ind] * Cell1.dx[zcenter2ind] ** 3) / np.sum(
           #     Cell1.nH[zcenter2ind] * Cell1.dx[zcenter2ind] ** 3)
          #  print(starhalfmassrad)
            zcenter2 = zcenter
            HIdenarr = np.zeros(int(n_bin)*2)
            gasdenarr = np.zeros(int(n_bin)*2)
            xHIarr = np.zeros(int(n_bin)*2)
            volsum2 = np.zeros(int(n_bin)*2)
            outflowarr = np.zeros(int(n_bin)*2)
            sdenarr = np.zeros(int(n_bin)*2)
            outflowarr_gas = np.zeros(int(n_bin)*2)
            avrecarr = np.zeros(int(n_bin)*2)
            cfarr = np.zeros(int(n_bin)*2)
            avTarr = np.zeros(int(n_bin)*2)

            for j in range(len(zselect)):
                zz = zcenter2 + zbin[zselect[j]]
                #print(xy, int(zstep/np.min(Cell1.dx)))

                HIden, gasden, xHI, volsum, avrec, cf, avT = decompose_new(Cell1,xcenter,ycenter,zz, starhmr*2, zstep)

                HIdenarr[int(n_bin)+j]=HIden
                gasdenarr[int(n_bin)+j]=gasden
                xHIarr[int(n_bin)+j]=xHI
                volsum2[int(n_bin)+j] = volsum
                outflow =getoutflow(Cell1,zz,True, xcenter,ycenter,starhmr*2)
                #outflow_gas = getoutflow(Cell1,zz,True, xcenter, ycenter, hmr2)
                outflowarr[int(n_bin)+j] = outflow
                #outflowarr_gas[int(n_bin)+j] = outflow_gas
                sdenarr[int(n_bin)+j] = stellarden(Part1,zz,zstep,starhmr*2,xcenter,ycenter)
                avrecarr[int(n_bin)+j] = avrec
                cfarr[int(n_bin)+j]=cf
                avTarr[int(n_bin)+j]=avT
                  #  print(j)
                zz = zcenter2 - zbin[zselect[j]]

                HIden, gasden, xHI, volsum, avrec, cf, avT = decompose_new(Cell1,xcenter,ycenter,zz, starhmr*2, zstep)
                volsum2[int(n_bin)-1-j] = volsum

                HIdenarr[int(n_bin)-1-j] =  HIden
                gasdenarr[int(n_bin)-1-j] =  gasden
                xHIarr[int(n_bin)-1-j]= xHI
                outflow = getoutflow(Cell1,zz,False, xcenter, ycenter, starhmr*2)
                outflowarr[int(n_bin)-1-j] = outflow
                avrecarr[int(n_bin) -1-j] = avrec
                cfarr[int(n_bin) -1- j] = cf
                avTarr[int(n_bin) -1- j] = avT
                #outflow_gas = getoutflow(Cell1, zz,False,xcenter,ycenter, hmr2)
                #outflowarr_gas[int(n_bin)-1-j] = outflow_gas
                sdenarr[int(n_bin)-1-j] = stellarden(Part1,zz,zstep,starhmr*2,xcenter,ycenter)

                 #   print(7-j)
                #starind = np.where((Part1.xp[2]>=zz-zbinwidth/2)&(Part1.xp[2]<zz+zbinwidth/2)&(get2dr(Part1.xp[0],Part1.xp[1],xcenter,ycenter)<2000))
                #starden = np.sum(Part1.mp0[starind[0]])/(2000**2*np.pi*zbinwidth)
                #stardenarr[j]=stardenarr[j]+starden

            #HIdenarr = HIdenarr/ volsum2
            #gasdenarr = gasdenarr / volsum2
            #xHIarr = xHIarr / volsum2

            if i ==0:
                HIdenarr2 = HIdenarr
                gasdenarr2 = gasdenarr
                xHIarr2 = xHIarr
                outflowarr2 = outflowarr
                sdenarr2 = sdenarr
               # outflowarr_gas2 = outflowarr_gas
                avrecarr2 = avrecarr
                cfarr2 = cfarr
                avTarr2 = avTarr

            else:
                HIdenarr2 = np.vstack((HIdenarr2, HIdenarr))
                gasdenarr2 = np.vstack((gasdenarr2, gasdenarr))
                xHIarr2 = np.vstack((xHIarr2, xHIarr))
                outflowarr2 = np.vstack((outflowarr2,outflowarr))
                sdenarr2 = np.vstack((sdenarr2,sdenarr))
                avrecarr2 = np.vstack((avrecarr2,avrecarr))
                cfarr2 = np.vstack((cfarr2,cfarr))
                avTarr2 = np.vstack((avTarr2, avTarr))
               # outflowarr_gas2 = np.vstack((outflowarr_gas2,outflowarr_gas))
            timearr=np.append(timearr,Part1.snaptime)

            np.savetxt(read+'gasdenarr2.dat',gasdenarr2,newline='\n',delimiter=' ')
            np.savetxt(read+'HIdenarr2.dat',HIdenarr2, newline='\n',delimiter=' ')
            np.savetxt(read+'xHIarr2.dat',xHIarr2, newline='\n', delimiter=' ')
            np.savetxt(read+'sdenarr2.dat',sdenarr2, newline='\n',delimiter=' ')
            np.savetxt(read+'outflowarr2.dat',outflowarr2,newline='\n',delimiter=' ')
            #nreaap.savetxt(read+'stardenarr2.dat',stardenarr2, newline='\n',delimiter=' ')
            np.savetxt(read+'timearr_jonghab.dat',timearr, newline='\n',delimiter=' ')
            #np.savetxt(read+'outflowarr_gas2.dat',outflowarr_gas2,newline='\n',delimiter=' ')
            np.savetxt(read+'cfarr2.dat',cfarr2,newline='\n',delimiter=' ')
            np.savetxt(read+'avrecarr2.dat',avrecarr2,newline='\n',delimiter=' ')
            np.savetxt(read+'avTarr2.dat',avTarr2,newline='\n',delimiter=' ')

    else:

        gasdenarr2 = np.loadtxt(read+'gasdenarr2.dat')
        HIdenarr2 = np.loadtxt(read+'HIdenarr2.dat')
        xHIarr2 = np.loadtxt(read+'xHIarr2.dat')
        timearr = np.loadtxt(read+'timearr_jonghab.dat')
        outflowarr2= np.loadtxt(read+'outflowarr2.dat')
        sdenarr2 = np.loadtxt(read+'sdenarr2.dat')
        #outflowarr_gas2 = np.loadtxt(read+'outflowarr_gas2.dat')
        cfarr2 = np.loadtxt(read+'cfarr2.dat')
        avrecarr2 = np.loadtxt(read+'avrecarr2.dat')
        avTarr2 = np.loadtxt(read+'avTarr2.dat')
    #xx = np.zeros(2*n_bin)

    #xx[len(zselect):] = zbin[zselect]
    #xx[:len(zselect)] = -np.flip(zbin[zselect],0)
    #print(xx)
    xx = zbin[zselect]
    gasden = np.zeros(len(xx))
    HIden = np.zeros(len(xx))
    xHI = np.zeros(len(xx))
    outflow = np.zeros(len(xx))
    cf = np.zeros(len(xx))
    T = np.zeros(len(xx))
    sden = np.zeros(len(xx))

    for i in range(len(xx)):
        gasden[len(xx)-i-1] = (np.median(gasdenarr2,axis=0)[i]+np.median(gasdenarr2,axis=0)[len(xx)*2-i-1])/2
        HIden[len(xx)-i-1] = (np.median(HIdenarr2,axis=0)[i]+np.median(HIdenarr2,axis=0)[len(xx)*2-i-1])/2
        xHI[len(xx)-i-1] = (np.median(xHIarr2,axis=0)[i]+np.median(xHIarr2,axis=0)[len(xx)*2-i-1])/2
        outflow[len(xx)-i-1] = (np.median(outflowarr2,axis=0)[i]+np.median(outflowarr2,axis=0)[len(xx)*2-i-1])/2
        cf[len(xx)-i-1] = (np.median(cfarr2,axis=0)[i]+np.median(cfarr2,axis=0)[len(xx)*2-i-1])/2
        T[len(xx)-i-1] = (np.median(avTarr2,axis=0)[i]+np.median(avTarr2,axis=0)[len(xx)*2-i-1])/2
        sden[len(xx)-i-1] = (np.median(sdenarr2,axis=0)[i]+np.median(sdenarr2,axis=0)[len(xx)*2-i-1])/2

    ax2.plot(xx,gasden, color=color, label=label, marker='D',ls=ls)
    ax3.plot(xx, HIden, color=color, marker='D',ls=ls)
    ax5.plot(xx, sden, color=color, marker='D',ls=ls)
    ax6.plot(xx, outflow,color=color, marker='D',ls=ls)
    #ax4.plot(xx,np.median(sdenarr2,axis=0),color=color, marker='D')
    #ax6.plot(xx, np.median(avrecarr2,axis=0),color=color, marker='D')
    #ax6.plot(xx, cf,color=color, marker='D')
    #ax7.plot(xx, T,color=color, marker='D')
    print(HIden)


    ax2.set_xlabel('|z| (pc)')

    ax6.set_xlabel('|z| (pc)')

        #ax1.plot(xx, np.mean(gasdenarr2, axis=0), color=color, label=label,ls='dashed')
        #ax2.plot(xx, np.mean(HIdenarr2, axis=0), color=color,ls='dashed')
        #ax3.plot(xx, np.mean(xHIarr2, axis=0), color=color,ls='dashed')



    #ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    #ax4.set_yscale('log')
    ax5.set_yscale('log')
    ax6.set_yscale('log')
    #ax7.set_yscale('log')
    #ax8.set_yscale('log')

   # ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    #ax4.set_xscale('log')
    ax5.set_xscale('log')
    ax6.set_xscale('log')
    #ax7.set_xscale('log')
    #ax8.set_xscale('log')

   # ax4.set_ylabel('$dM/dt$ $(M_\odot/yr)$')
    ax2.set_ylabel('nH $(cm^{-3})$')
    ax3.set_ylabel('nHI $(cm^{-3})$')
    ax5.set_ylabel(r'$\rho_* (M_\odot pc^{-2})$')
    #ax4.set_ylabel(r'$\rho_* (M_\odot/pc^2)$')
    ax6.set_ylabel('$dM_{out}/dt$ $(M_\odot/yr)$')
    #ax6.set_ylabel('$dnHI/dt (cm^{-3}\cdot s^{-1}$')
    #ax6.set_ylabel(r'$C\equiv\langle n^2 \rangle / \langle n \rangle^2$')
    #ax7.set_ylabel('T')

    #ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax3.xaxis.set_major_formatter(plt.NullFormatter())
    ax5.xaxis.set_major_formatter(plt.NullFormatter())
    #ax6.xaxis.set_major_formatter(plt.NullFormatter())



    dat = np.loadtxt(read+'sh_hmr.dat')
    timearr = dat[:,0]
    gassh = dat[:,5]
    starsh = dat[:,4]

    ax1.plot(timearr, gassh, color=color, label=label,ls=ls)
    #ax1.plot(timearr, starsh, color=color,  ls='dashed')
    """
    dat2 = np.loadtxt(read+'outflowarr_box_of.dat')

    outbox10 = np.append(dat2[:229,1],dat2[230:,1])
    timeout = np.append(dat2[:229,-1],dat2[230:,-1])


    ax11 = ax1.twinx()
    ax11.plot(timeout, outbox10, color=color, ls='dashed')

    ax11.set_ylim(1e-4,0.5)
    ax11.set_yticks([1e-4,1e-3,1e-2,1e-1,1])
    ax11.set_yscale('log')
    ax1.set_ylabel('H (pc)')
    ax1.set_xlabel('time (Myr)')
    if read==read_new_01Zsun:
        ax11.set_ylabel('$dm_{out}/dt$ $(M_\odot/yr)$')
    else:
        ax11.set_yticks([])
    """
    ax2.set_yticks([1e-5, 1e-3, 1e-1, 1e1])
    ax3.set_yticks([1e-5,1e-3,1e-1, 1e1])
    #ax4.set_yticks([1e-6,1e-4,1e-2,1])
    ax5.set_yticks([1e-6,1e-4,1e-2,1])
    ax6.set_yticks([1e-3,1e-1,10,1000])

    #ax7.set_yticks([1e3,1e4,1e5,1e6])

    ax2.set_xticks([1e2,1e3,1e4])
    ax3.set_xticks([1e2,1e3,1e4])
    #ax4.set_xticks([1e2,1e3,1e4])
    ax5.set_xticks([1e2,1e3,1e4])
    ax6.set_xticks([1e2,1e3,1e4])
    ax6.set_ylim(5e-4, 1e3)

    #ax7.set_xticks([1e2,1e3,1e4])

    ax1.legend(frameon=False,fontsize=13)

    ax3.set_ylim(1e-5,10)


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] =15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['figure.titlesize'] = 13

fig = plt.figure(figsize=(8,8))

ax1 = fig.add_axes([0.13,0.69,0.75,0.27])

ax5 = fig.add_axes([0.13,0.36,0.34,0.23])
ax2 = fig.add_axes([0.13,0.1,0.34,0.23])

#ax4 = fig.add_axes([0.12,0.05,0.35,0.17])

ax3 = fig.add_axes([0.54,0.36,0.34,0.23])
ax6 = fig.add_axes([0.54,0.1,0.34,0.23])
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()
ax6.yaxis.set_label_position("right")
ax6.yaxis.tick_right()
#ax7 = fig.add_axes([0.62,0.05,0.35,0.17])

jonghab(Cell,read_new_01Zsun,150,300,np.linspace(125,8125,num=32), [0,1,2,3,7,15,31],8000,True,'k','G9_Zlow_c001')
jonghab(Cell,read_new_1Zsun,150,300,np.linspace(125,8125,num=32), [0,1,2,3,7,15,31],8000,True,'firebrick','G9_Zhigh')
jonghab(Cellfromdat,read_new_gasrich,150,300,np.linspace(125,8125,num=32), [0,1,2,3,7,15,31],8000,True,'dodgerblue','G9_Zlow_gas5')
#jonghab(Cell,read_new_1Zsun_highSN_new,70,220,np.linspace(125,8125,num=32), [0,1,2,3,7,15,31],8000,True,'magenta','G9_Zhigh_SN5')
jonghab(Cell,'/Volumes/Gemini/G9_Zlow/',150,300,np.linspace(125,8125,num=32), [0,1,2,3,7,15,31],8000,True,'k','G9_Zlow_c01',ls='dashed')


"""
if __name__ == '__main__':
    reads = [read_new_01Zsun, read_new_gasrich]
    Cells = [Cell, Cellfromdat]
    inis = [150, 150]
    # inis = [3,3,3,3,3]
    ends = [300, 300]
    colors = ['k', 'b']
    labels = ['G9_Zlow', 'G9_Zlow_gas5']
    loads = [True,True]
    procs = []
    for cell, read, ini, end, load, color, label in zip(Cells, reads, inis, ends, loads, colors, labels,
                                                                  ):
        proc = Process(target=jonghab, args=(
        cell, read, ini, end, np.linspace(125, 8125, num=32), [0, 1, 2, 3, 7, 15, 31], 8000, load, color, label
        ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
"""
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/vert_plot_thesis.pdf')
plt.show()