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
class Cellfromdat():
    def __init__(self, dir, nout, Part):
        celldata = FortranFile(dir + 'dat/cell_%05d.dat' % nout, 'r')
        nlines,xx,nvarh=celldata.read_ints(dtype=np.int32)
        #print(nlines)
        #print(nvarh)
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
        sed_attHI = dat2.read_reals(dtype=np.double)
        sed_attH2 = dat2.read_reals(dtype=np.double)
        sed_attHe= dat2.read_reals(dtype=np.double)
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
def CoM_Main2(Part1):
    xcen = np.sum(Part1.mp0*Part1.xp[0])/np.sum(Part1.mp0)
    ycen = np.sum(Part1.mp0*Part1.xp[1])/np.sum(Part1.mp0)
    zcen = np.sum(Part1.mp0*Part1.xp[1])/np.sum(Part1.mp0)
    return xcen, ycen, zcen
def sfrd_hmr(Part,Cell,xcen,ycen,zcen,rbin, zrange):
    totstarmass = np.sum(Part.mp0)
    cellrr = get2dr(Cell.x,Cell.y, xcen,ycen)
    cellzdist = np.abs(Cell.z-zcen)
    partrr = get2dr(Part.xp[0],Part.xp[1],xcen,ycen)
    partzdist = np.abs(Part.xp[2]-zcen)
    for i in range(len(rbin)):
        #cellind = np.where((cellrr>=rbin[i])&(cellrr<rbin[i+1])&(cellzdist<zrange))
        partind = np.where((partrr<rbin[i])&(partzdist<zrange))
        if np.sum(Part.mp0[partind])>totstarmass/2:
            halfmassrad = rbin[i]
            break

#    halfmassind_cell = np.where((cellrr<halfmassrad)&(cellzdist<zrange))
    halfmassind_part = np.where((partrr<halfmassrad)&(partzdist<zrange)&(Part.starage<10))


    area = np.pi * (halfmassrad)**2
    #sd = np.sum(Cell.mHIH2[halfmassind_cell])/area
    sfrd = np.sum(Part.mp0[halfmassind_part])/area/10
    print(np.sum(Part.mp0[halfmassind_part]),halfmassrad)


    return sfrd,halfmassrad

def plot_scatter(ax, ax2, Cell, Fesc, read, inisnap, endsnap, color, label, loadcom, load):
    numsnap = endsnap - inisnap + 1
    if loadcom == True:
        arr = np.loadtxt(read + 'sh_hmr.dat')
    if load==False:
        sfrdarr=np.array([])
        #timearr = np.array([])
        fescarr = np.array([])
        for n in range(numsnap):
            nout = n + inisnap
            print(nout)
            if read == read_new_1Zsun:
                if nout == 26 or nout == 27 or nout == 28 or nout == 29:
                    continue
            if Cell==Cellfromdat:
                if nout==165 or nout==166 or nout==239:
                    continue
                if not os.path.isfile(read + '/dat/cell_%05d.dat' % (nout)):
                    print(read + '/dat/cell_%05d.dat' % (nout))
                    continue
            else:
                if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
                    print(read + '/SAVE/cell_%05d.sav' % (nout))
                    continue
            if Fesc==Fesc_new8:
                if not os.path.isfile(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout)):
                    print(read + '/ray_nside8_laursen/ray_%05d.dat' % (nout))
                    continue
            if Fesc==Fesc_new:
                if not os.path.isfile(read + '/ray_nside4_laursen/ray_%05d.dat' % (nout)):
                    print(read + '/ray_nside4_laursen/ray_%05d.dat' % (nout))
                    continue
            Part1 = Part(read, nout)
            Cell1 = Cell(read, nout, Part1)
            Fesc1 = Fesc(read, nout)
            if np.isnan(Fesc1.fesc)==True:
                continue
            if Fesc1.fesc==0:
                continue
            #xcen, ycen, zcen = CoM_Main(Part1)
            #xcen=xcen[-1];ycen=ycen[-1];zcen=zcen[-1]
            xcen, ycen, zcen = CoM_Main2(Part1)

            if loadcom==True:
                ind = np.where(arr[:,0]==nout)
                hmr = arr[ind[0],2]
                partrr = get2dr(Part1.xp[0], Part1.xp[1], xcen, ycen)
                partzdist = np.abs(Part1.xp[2] - zcen)
                halfmassind_part = np.where((partrr < hmr) & (partzdist < 4000) & (Part1.starage < 10))
                area = np.pi * (hmr) ** 2
                # sd = np.sum(Cell.mHIH2[halfmassind_cell])/area
                sfrd = np.sum(Part1.mp0[halfmassind_part]) / area / 10
            else:
                sfrd,hmr = sfrd_hmr(Part1,Cell1,xcen,ycen,zcen,np.linspace(0,8000,num=160),3000)
            #print(nout)
            print(sfrd,hmr)

            sfrdarr = np.append(sfrdarr, sfrd)
            #timearr = np.append(timearr, Part1.snaptime)
            fescarr = np.append(fescarr, Fesc1.fesc)

        np.savetxt(read+'sfrd_fesc.dat',sfrdarr,newline='\n',delimiter=' ')
        np.savetxt(read+'fesc_sfrd.dat',fescarr,newline='\n',delimiter=' ')
    else:
        sfrdarr = np.loadtxt(read+'sfrd_fesc.dat')
        fescarr = np.loadtxt(read+'fesc_sfrd.dat')


    ax.scatter(sfrdarr, fescarr,color=color,s=2)

    plot_in_scatter(ax, sfrdarr,fescarr,color,label)

    outflow = np.loadtxt(read+'outflowarr_box_of.dat')
    fescof = np.loadtxt(read+'fescarr_of.dat')
    outflowbox = outflow[:,1]
    ax2.scatter(outflowbox, fescof, color=color, s=2)
    plot_in_scatter(ax2, outflowbox, fescof, color, label)


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

    ax.errorbar(xxbin, avy,yerr=(low,upp), marker='s', color='w',lw=6)

    ax.errorbar(xxbin, avy, yerr=(low,upp), marker='s', color=color, lw=3,label=label)


fig = plt.figure(figsize=(8, 9), dpi=144)

ax1 = fig.add_axes([0.15, 0.59, 0.8, 0.4])
ax2 = fig.add_axes([0.15,0.09,0.8,0.4])
plot_scatter(ax1,ax2,Cell,Fesc_new8,read_new_01Zsun,150,480,'k','G9_Zlow',True,True)
plot_scatter(ax1,ax2,Cell,Fesc_new8,read_new_1Zsun,150,480,'firebrick','G9_Zhigh',True,True)
plot_scatter(ax1,ax2,Cellfromdat,Fesc_new8,read_new_gasrich,150,300,'dodgerblue','G9_Zlow_gas5',True,True)
plot_scatter(ax1,ax2,Cellfromdat,Fesc_new8,read_new_03Zsun_highSN,70,380,'orange','G9_Zmid_SN5',True,True)
plot_scatter(ax1,ax2,Cell,Fesc_new8,read_new_1Zsun_highSN_new,70,380,'magenta','G9_Zhigh_SN5',True,True)

ax1.set_ylabel('$log(f_{esc})$')
ax1.set_xlabel('$log(\Sigma_{SFR})$ $(M_\odot yr^{-1} kpc^{-2} )$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(frameon=False,fontsize=16)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel('$log(f_{esc})$')
ax2.set_xlabel('$log(dM_{out}/dt)$ $(M_\odot yr^{-1})$')
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/sfrd_fesc_fin_thesis2.pdf')
plt.show()