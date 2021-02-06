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

def CoM_Main(Part1, Cell1, diskmass):
    xcen = np.zeros(11)
    ycen = np.zeros(11)
    zcen = np.zeros(11)
    hmr = np.zeros(11)
    rgrid = np.linspace(100, 8000, num=80)
    rgrid2 = np.linspace(600, 8000, num=38)

    xcen = np.sum(Part1.mp0 * Part1.xp[0]) / np.sum(Part1.mp0)
    ycen= np.sum(Part1.mp0 * Part1.xp[1]) / np.sum(Part1.mp0)
    zcen = np.sum(Part1.mp0 * Part1.xp[2]) / np.sum(Part1.mp0)
    for j in range(len(rgrid)):
        mass = getmass_zlim(Part1.mp0, get2dr(Part1.xp[0], Part1.xp[1], xcen, ycen), rgrid[j], Part1.xp[2],
                            zcen, 2000)
        if mass > np.sum(Part1.mp0) / 2:
            hmr= rgrid[j]
            break
    for j in range(len(rgrid2)):
        mass = getmass_zlim(Cell1.m, get2dr(Cell1.x, Cell1.y, xcen, ycen), rgrid2[j], Cell1.z,
                            zcen, 8000)
        if mass > np.sum(Part1.mp0) / 2:
            hmr2 = rgrid2[j]
            break

    return xcen, ycen, zcen, hmr, hmr2


def getoutflow(Cell, zz,up, xcen, ycen, rad):
    if up == True:
        ind=np.where((Cell.z+Cell.dx/2>zz)&(Cell.x-Cell.dx/2<zz)&(Cell.vz>0)&(get2dr(Cell.x,Cell.y,xcen,ycen)<rad))
    else:
        ind = np.where((Cell.x + Cell.dx / 2 > zz) & (Cell.x - Cell.dx / 2 < zz) & (Cell.vz < 0) & (get2dr(Cell.x,Cell.y,xcen,ycen)<rad))
    outflow = np.sum(Cell.den[ind] * np.abs(Cell.vz[ind]) * (Cell.dx[ind] *3.08e18)**2/1.989e33*365*24*3600)


    return outflow


def getinflow(Cell, zz,up, xcen, ycen, rad):
    if up == True:
        ind=np.where((Cell.z+Cell.dx/2>zz)&(Cell.z-Cell.dx/2<zz)&(Cell.vz<0)&(get2dr(Cell.x,Cell.y,xcen,ycen)<rad))
    else:
        ind = np.where((Cell.z + Cell.dx / 2 > zz) & (Cell.z- Cell.dx / 2 < zz) & (Cell.vz > 0) & (get2dr(Cell.x,Cell.y,xcen,ycen)<rad))
    outflow = np.sum(Cell.den[ind] * np.abs(Cell.vz[ind]) * (Cell.dx[ind] *3.08e18)**2/1.989e33*365*24*3600)


    return outflow
def plot_in_scatter(ax, x, y,color,label):
    xbin = [np.min(x), np.percentile(x,20), np.percentile(x,40), np.percentile(x,60), np.percentile(x,80),np.max(x)]
    avy = np.zeros(len(xbin)-1)
    upp = np.zeros(len(xbin)-1)
    low = np.zeros(len(xbin)-1)
    for j in range(len(xbin)-1):
        ind = np.where((x>=xbin[j])&(x<xbin[j+1]))
        avy[j] = np.median(y[ind])
        upp[j] = np.abs(avy[j]-np.percentile(y[ind],75))
        low[j] = np.abs(avy[j]-np.percentile(y[ind],25))

    xxbin = [np.percentile(x,10),np.percentile(x,30),np.percentile(x,50),np.percentile(x,70),np.percentile(x,90)]

    ax.errorbar(xxbin, avy,yerr=(low,upp), marker='s', color='w',lw=4,markersize=7)

    ax.errorbar(xxbin, avy, yerr=(low,upp), marker='s', color=color, lw=2,label=label,markersize=5)

def main(read, Cell, Fesc, inisnap, endsnap,load,diskmass,zbin, axlist, axlist2, color, label,loadcom):
    numsnap = endsnap - inisnap + 1
    if load == False:
        sfrdarr = np.array([])
        # timearr = np.array([])
        fescarr = np.array([])
        skip=0
        #outflowarr2 = np.array([])
        #outflow_gasarr2 = np.array([])
        for n in range(numsnap):
            nout = n + inisnap
            print(nout)
            if read == read_new_1Zsun:
                if nout == 26 or nout == 27 or nout == 28 or nout == 29:
                    continue
            if Cell == Cellfromdat:
                if nout == 165 or nout == 166:
                    continue
                if not os.path.isfile(read + '/dat/cell_%05d.dat' % (nout)):
                    print(read + '/dat/cell_%05d.dat' % (nout))
                    continue
            else:
                if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
                    print(read + '/SAVE/cell_%05d.sav' % (nout))
                    continue
            if Fesc == Fesc_new8:
                if not os.path.isfile(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout)):
                    print(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout))
                    continue
            if Fesc == Fesc_new:
                if not os.path.isfile(read + '/ray_nside4_laursen/ray_%05d.dat' % (nout)):
                    print(read + '/ray_nside4_laursen/ray_%05d.dat' % (nout))
                    continue
            Part1 = Part(read, nout)
            Cell1 = Cell(read, nout, Part1)
            Fesc1 = Fesc(read, nout)
            if np.isnan(Fesc1.fesc) == True:
                continue
            if Fesc1.fesc == 0:
                continue

            if loadcom ==False:
                xcen, ycen, zcen, hmr, hmr2 = CoM_Main(Part1,Cell1,diskmass)

            else:
                cenarr = np.loadtxt(read + 'cenarr_whole.dat')
                xcenter = cenarr[:, 0]
                ycenter = cenarr[:, 1]
                zcenter = cenarr[:, 2]
                noutarr = cenarr[:, 3]

                hmrarr = np.loadtxt(read + 'hmrarr_whole.dat')
                gashmrarr = np.loadtxt(read + 'gashmrarr_whole.dat')

                ind = np.where(noutarr == nout)

                xcen = xcenter[ind]
                ycen = ycenter[ind]
                zcen = zcenter[ind]
                hmr = hmrarr[ind]
                hmr2 = gashmrarr[ind]

            outflowarr = np.zeros(len(zbin))
            outflow_gasarr= np.zeros(len(zbin))
            for j in range(len(zbin)):
                zz = zcen + zbin[j]
                outflow = getoutflow(Cell1, zz, True, xcen, ycen, hmr)
                outflow_gas = getoutflow(Cell1, zz, True, xcen, ycen, hmr2)

                zz = zcen-zbin[j]
                outflow2 = getoutflow(Cell1, zz, True, xcen, ycen, hmr)
                outflow_gas2 = getoutflow(Cell1, zz, True, xcen, ycen, hmr2)

                outflowarr[j] = (outflow + outflow2)/2
                outflow_gasarr[j] = (outflow_gas + outflow_gas2)/2

            if len(outflowarr)>0 and skip ==0:
                outflowarr2 = outflowarr
                outflow_gasarr2 = outflow_gasarr
                skip=1
            else:
                outflowarr2 = np.vstack((outflowarr2, outflowarr))
                outflow_gasarr2 = np.vstack((outflow_gasarr2, outflow_gasarr))

            fescarr = np.append(fescarr, Fesc1.fesc)
            #timearr = np.append(timearr, Part1.snaptime)

        #np.savetxt(read+'timearr_of.dat',timearr, newline='\n',delimiter=' ')
        np.savetxt(read+'fescarr_of.dat',fescarr, newline='\n',delimiter=' ')
        np.savetxt(read+'outflowarr_of.dat',outflowarr2,newline='\n',delimiter=' ')
        np.savetxt(read+'outflowarr_gas_of.dat',outflow_gasarr2, newline='\n',delimiter=' ')

    else:

        fescarr = np.loadtxt(read+'fescarr_of.dat')
#        timearr = np.loadtxt(read+'fescarr_of.dat')
        outflowarr2 = np.loadtxt(read+'outflowarr_of.dat')
        outflow_gasarr2 = np.loadtxt(read+'outflowarr_gas_of.dat')


    for j in range(len(zbin)):
        axlist[j].scatter(outflowarr2[:,j], fescarr, color=color,label=label,s=0.5)
        axlist2[j].scatter(outflow_gasarr2[:,j], fescarr, color=color, label=label,s=0.5)
        plot_in_scatter(axlist[j],outflowarr2[:,j],fescarr,color,label)
        plot_in_scatter(axlist2[j],outflow_gasarr2[:,j],fescarr,color,label)

        axlist[j].set_ylabel('$f_{esc}$')
        axlist2[j].set_ylabel('$f_{esc}$')
        axlist[j].set_yscale('log')
        axlist[j].set_xscale('log')
        axlist2[j].set_yscale('log')
        axlist2[j].set_xscale('log')
        axlist[j].set_ylim(1e-4,1)
        axlist2[j].set_ylim(1e-4,1)

        #axlist[j].set_xlim(1e-4,50)
        #axlist2[j].set_xlim(1e-4,50)
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
fig = plt.figure(figsize=(15,8))
ax1 = fig.add_axes([0.1,0.1,0.25,0.25])
ax2 = fig.add_axes([0.4,0.1,0.25,0.25])
ax3 = fig.add_axes([0.7,0.1,0.25,0.25])
ax4 = fig.add_axes([0.1,0.5,0.25,0.25])
ax5 = fig.add_axes([0.4,0.5,0.25,0.25])
ax6 = fig.add_axes([0.7,0.5,0.25,0.25])

zbin = [500, 1000, 2000]
axlist = [ax1, ax2, ax3]
axlist2 = [ax4, ax5, ax6]
main(read_new_01Zsun,Cell,Fesc_new8,150,480,True,1.75e09,zbin,axlist,axlist2,'k','G9_Zlow',False)
main(read_new_1Zsun,Cell,Fesc_new8,150,480,True,1.75e09,zbin,axlist,axlist2,'r','G9_Zhigh',False)
main(read_new_1Zsun_highSN_new,Cell,Fesc_new8,70,380,True,1.75e09,zbin,axlist,axlist2,'magenta','G9_Zhigh_SN5',False)
#main(read_new_01Zsun_05pc,Cell,Fesc_new8,3,480,True,1.75e09,zbin,axlist,axlist2,'lightseagreen','G9_Zlow_HR',False)
main(read_new_03Zsun_highSN,Cell,Fesc_new8,70,380,True,1.75e09,zbin,axlist,axlist2,'orange','G9_Zmid_SN5',False)
main(read_new_gasrich,Cellfromdat,Fesc_new8,150,300,True,1.75e09*5,zbin,axlist,axlist2,'b','G9_Zlow_gas5',False)
ax2.yaxis.set_major_formatter(plt.NullFormatter())
ax3.yaxis.set_major_formatter(plt.NullFormatter())
ax5.yaxis.set_major_formatter(plt.NullFormatter())
ax6.yaxis.set_major_formatter(plt.NullFormatter())

ax1.set_xlabel('$(dm_{out, 500pc}/dt)_{starhmr} (M_\odot/yr)$')
ax2.set_xlabel('$(dm_{out, 1000pc}/dt)_{starhmr} (M_\odot/yr)$')
ax3.set_xlabel('$(dm_{out, 2000pc}/dt)_{starhmr} (M_\odot/yr)$')
ax4.set_xlabel('$(dm_{out, 500pc}/dt)_{gashmr} (M_\odot/yr)$')
ax5.set_xlabel('$(dm_{out, 1000pc}/dt)_{gashmr} (M_\odot/yr)$')
ax6.set_xlabel('$(dm_{out, 2000pc}/dt)_{gashmr} (M_\odot/yr)$')
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/outflowfesc.png')
plt.show()






