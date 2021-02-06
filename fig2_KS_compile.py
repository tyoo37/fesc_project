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
        self.mHIH2 = self.m*(1-xHII) * 0.76
        self.mH2 = self.m * (1-xHII-xHI) * 0.76

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
        self.mHIH2 = self.m * (1-xHII)
        self.mH2 = self.m * (1-xHII-xHI)

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

def halfmassrad_KS(Part,Cell,xcen,ycen,zcen,halfmassrad, zrange):
    totstarmass = np.sum(Part.mp0)
    cellrr = get2dr(Cell.x,Cell.y, xcen,ycen)
    cellzdist = np.abs(Cell.z-zcen)
    partrr = get2dr(Part.xp[0],Part.xp[1],xcen,ycen)
    partzdist = np.abs(Part.xp[2]-zcen)

    halfmassind_cell = np.where((cellrr<halfmassrad)&(cellzdist<zrange))
    halfmassind_part = np.where((partrr<halfmassrad)&(partzdist<zrange)&(Part.starage<10))


    area = np.pi * (halfmassrad)**2
    sd = np.sum(Cell.mH2[halfmassind_cell])/area
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
    """
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


    """
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
def scatter_kennicutt(ax,ax2,filename,colnum,color,sfrind,Hind, H2ind,marker,label):
    arr = fits_to_arr(colnum,filename)
    SFR = 10**arr[:,sfrind].astype(float)
    HI = 10**arr[:,Hind].astype(float)* 1.67e-24  * 3.08e18 /1.98e33 *3.08e18
    H2 = 10**arr[:,H2ind].astype(float) * 1.67e-24 * 2 * 3.08e18 /1.98e33 *3.08e18
    ax.scatter(HI+H2, SFR, color=color,marker=marker,label=label,alpha=0.5)

    ax2.scatter(H2, SFR, color=color,marker=marker,label=label,alpha=0.5)
def scatter_tacconi(ax,filename,colnum,color,sfrind,Hind,marker,label):
    arr = fits_to_arr(colnum,filename)
    SFR = 10**arr[:,sfrind].astype(float)
    H = 10**arr[:,Hind].astype(float)

    ax.scatter(H, SFR, color=color,marker=marker,label=label,alpha=0.5)

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

    HIH2=H2
    #HIH2 = HI+H2


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

def gauss_cont(ax1,ax2, filename1,filename2,colnum1,colnum2):

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

    HIH22 = 10 ** arr2[:, 2].astype(float)
    SFR2 = arr2[:, 4].astype(float) / 1e5
    #index2 = np.where((HIH22 != 1.) & (SFR2 != 1.))
    #HIH2 = np.append(HIH2, HIH22[index2])
    #SFR2 = np.append(SFR, SFR2[index2])

    print(np.min(HIH2),np.max(HIH2),np.min(SFR),np.max(SFR))

    xx, yy = np.mgrid[np.min(np.log10(HIH2))-0.5:np.max(np.log10(HIH2))+0.5:200j, np.min(np.log10(SFR))-0.5:np.max(np.log10(SFR))+0.5:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([np.log10(HIH2), np.log10(SFR)])
    #ax.scatter(HIH2,SFR,s=1)
    #ax.scatter(HIH22,SFR2,s=1)

    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    sigma_1, sigma_2, sigma_3 = sigma3(f)

    ss1 = ax1.contour(np.power(10,xx),np.power(10,yy),f,levels=[sigma_1,sigma_2,sigma_3],colors='grey',linewidths=2,origin='lower',alpha=0.4)

    xx, yy = np.mgrid[np.min(np.log10(HIH22)) - 0.5:np.max(np.log10(HIH22)) + 0.5:200j,
             np.min(np.log10(SFR2)) - 0.5:np.max(np.log10(SFR2)) + 0.5:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([np.log10(HIH22), np.log10(SFR2)])
    # ax.scatter(HIH2,SFR,s=1)
    # ax.scatter(HIH22,SFR2,s=1)

    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    sigma_1, sigma_2, sigma_3 = sigma3(f)

    ss1 = ax1.contour(np.power(10, xx), np.power(10, yy), f, levels=[sigma_1, sigma_2, sigma_3], colors='grey',
                      linewidths=2, origin='lower', alpha=0.4)


    xx, yy = np.mgrid[np.min(np.log10(H2)) - 0.5:np.max(np.log10(H2)) + 0.5:200j,
             np.min(np.log10(SFR)) - 0.5:np.max(np.log10(SFR)) + 0.5:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([np.log10(H2), np.log10(SFR)])
    # ax.scatter(HIH2,SFR,s=1)
    # ax.scatter(HIH22,SFR2,s=1)

    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    sigma_1, sigma_2, sigma_3 = sigma3(f)

    ss2 = ax2.contour(np.power(10, xx), np.power(10, yy), f, levels=[sigma_1, sigma_2, sigma_3], colors='grey',
                     linewidths=2, origin='lower', alpha=0.4)

    #ss1 = ax.contour(xx,yy,f,colors='darkgoldenrod',linewidths=0.5)
    """
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
    """
    labels = ['Bigiel+10']
    ss1.collections[0].set_label(labels[0])
    ss2.collections[0].set_label(labels[0])

    #ss1.collections[-1].set_label(labels[1])
   # ss2.collections[0].set_label(labels[1])
    #ss2.collections[-1].set_label(labels[3])



def KS(ax,ax2, read, Part1, Cell1, saveload,label,color,nout):

    if saveload==False:
        sfrdarr= np.array([])

        sdarr= np.array([]);sdarr2=np.array([])
        diskgasmass = 1.75e9 # solarmass
        rgrid = np.linspace(0, 5000, num=11)
        print(rgrid)
        rgrid2 = np.linspace(0,5000,num=100)
        #xcen, ycen, zcen = CoM_Main(Part1)
        #xcen=xcen[-1]
        #ycen=ycen[-1]
        #zcen=zcen[-1]
        xcen = np.sum(Part1.mp0*Part1.xp[0])/np.sum(Part1.mp0)
        ycen = np.sum(Part1.mp0*Part1.xp[1])/np.sum(Part1.mp0)

        zcen = np.sum(Part1.mp0*Part1.xp[2])/np.sum(Part1.mp0)

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
            sd = np.sum(Cell1.mHIH2[indcell])/area
            sd2 = np.sum(Cell1.mH2[indcell])/area #solar mass/pc^2
            sfrd = np.sum(Part1.mp0[indstar])/10/area #solar mass /yr/kpc^2
            sdarr=np.append(sdarr,sd)
            sdarr2=np.append(sdarr2,sd2)

            sfrdarr=np.append(sfrdarr,sfrd)

        arr = np.loadtxt(read + 'sh_hmr.dat')
        noutarr = arr[:, 0]
        indnout = np.where(noutarr == nout)
        gassh = arr[indnout, 5]
        starhmr = arr[indnout, 2]

        starhmr = np.asscalar(starhmr)
        #halfmass_sd, halfmass_sfrd,halfmassrad = halfmassrad_KS(Part1,Cell1,xcen,ycen,zcen,starhmr,8000)
        #print('hmr',halfmassrad)
        #sdarr = np.append(sdarr, halfmass_sd)
        #sfrdarr = np.append(sfrdarr, halfmass_sfrd)
        print(sdarr, sfrdarr)
        np.savetxt(read+'sd.dat',sdarr,delimiter=' ',newline='\n')
        np.savetxt(read+'sfrd.dat',sfrdarr,delimiter=' ',newline='\n')

        #halfmassradind = int(np.min(np.where(rgrid>halfmassrad)[0]))
        for i in range(len(rgrid)-1):
            if i ==3:
                ax.scatter(sdarr[i], sfrdarr[i], color=color, s=200*0.8**i,label=label)
            else:
                ax.scatter(sdarr[i], sfrdarr[i], color=color, s=200*0.8**i)
        for i in range(len(rgrid)-1):
            if i ==3:
                ax2.scatter(sdarr2[i], sfrdarr[i], color=color, s=200*0.8**i,label=label)
            else:
                ax2.scatter(sdarr2[i], sfrdarr[i], color=color, s=200*0.8**i)
        #ax.scatter(sdarr[:halfmassradind], sfrdarr[:halfmassradind], color =color,s=100)
        #ax.scatter(sdarr[halfmassradind:-1], sfrdarr[halfmassradind:-1], color =color,s=20)
        #ax.scatter(sdarr[-1],sfrdarr[-1],color=color,marker='*',s=400,label=label)


    else:

        sdarr=np.loadtxt(read+'sd.dat')
        sfrdarr=np.loadtxt(read+'sfrd.dat')


        ax.scatter(sdarr, sfrdarr,  color=color, s=np.linspace(25, 5, len(sdarr)))

        ax.scatter(sdarr2, sfrdarr,  color=color, s=np.linspace(25, 5, len(sdarr)))

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

def main(ax, ax2,read, Cell,label,color, saveload, noutdirect):
    if noutdirect==0:
        nout=findsnap(read, 200,250, 200)
    else:
        nout=noutdirect
    Part1=Part(read,nout)
    Cell1=Cell(read,nout,Part1)

    KS(ax,ax2,read, Part1,Cell1,saveload,label,color,nout)



def emp(x):
    return 1.5e-4*x**1.4
fig = plt.figure(figsize=(8,14),dpi=72)
ax2=fig.add_axes([0.15,0.1,0.8,0.4])
ax1=fig.add_axes([0.15,0.59,0.8,0.4])

#ax1.plot([0.03,200],[emp(0.03),emp(200)],ls='dashed')
"""
ax1.plot([0.1,1e5],[emp(0.1),emp(1e5)],ls='dotted',c='grey')
ax1.annotate('$\Sigma_*\propto \Sigma_{gas}^{1.4}$',xy=(1e3,2*emp(1e3)),xycoords='data',rotation=np.degrees(np.arctan2(np.log10(emp(1e5))-np.log10(emp(0.1)),6)))
ax1.plot([0.1,1e5],[0.1/1e4,1e5/1e4],ls='dotted',c='grey')
ax1.annotate('1%',xy=(1,1.1*1/1e4),xycoords='data',rotation=40)
ax1.plot([0.1,1e5],[0.1/1e3,1e5/1e3],ls='dotted',c='grey')
ax1.annotate('10%',xy=(1,1.1*1/1e3),xycoords='data',rotation=40)
ax1.plot([0.1,1e5],[0.1/1e2,1e5/1e2],ls='dotted',c='grey')
ax1.annotate('100%',xy=(1,1.1*1/1e2),xycoords='data',rotation=40)
"""

cmap1 = plt.get_cmap('Purples')
cmap2 = plt.get_cmap('Greys')
gauss_cont(ax1,ax2, '/Users/taehwau/projects/KS/bigiel10_inner.fit', '/Users/taehwau/projects/KS/bigiel10_outer.fit',8,6)

#hist2d(ax1,'/Users/taehwau/projects/KS/bigiel10_inner.fit','/Users/taehwau/projects/KS/bigiel10_outer.fit',8,6,cmap1,'Bigiel+10 ')
#hist2d(ax1,'/Users/taehwau/projects/KS/bigiel10_outer.fit',6,cmap2,'Bigiel+10 (outer disk)')
scatter_kennicutt(ax1,ax2,'/Users/taehwau/projects/KS/kennicutt07_total.fit',4,'grey',0,1,2,'x', 'Kennicutt+07 (M51)')
scatter_tacconi(ax2,'/Users/taehwau/projects/KS/tacconi13_2.fit',2,'grey',1,0,'^', 'Tacconi+13')#scatter2(ax1,'/Users/taehwau/projects/KS/tacconi13.fit',5,'grey','grey',4,3,'^', 's','Tacconi+13 (z<2)','Tacconi+13 (z>2)')

main(ax1,ax2,read_new_01Zsun,Cell,'G9_Zlow','k',False,300)
main(ax1,ax2, read_new_1Zsun, Cell,'G9_Zhigh','firebrick',False,300)
main(ax1,ax2,read_new_gasrich,Cellfromdat,'G9_Zlow_gas5','dodgerblue',False,300)
main(ax1,ax2,read_new_1Zsun_highSN_new,Cell,'G9_Zhigh_SN5','magenta',False,220)
main(ax1,ax2,read_new_03Zsun_highSN,Cell,'G9_Zmid_SN5','orange',False,220)
main(ax1,ax2,read_new_01Zsun_05pc,Cell,'G9_Zlow_HR','lightseagreen',False,220)
#main(ax1,read_new_01Zsun,Cell,'G9_Zlow','k',False,300,1.75e9)
#main(ax1,read_new_01Zsun_highSN,Cell,'G9_Zlow_SN5','olive',False,300,1.75e9)

ax1.set_xlabel(r'$\Sigma_{HI+H2}$ $(M_\odot/pc^2)$',fontsize=25)
ax1.set_ylabel(r'$\Sigma_*$ $(M_\odot/yr/kpc^2)$',fontsize=25)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1e-1,1e4)
ax1.set_ylim(1e-5,1e1)
ax1.legend(frameon=False,loc='lower right')

ax2.set_xlabel(r'$\Sigma_{H2}$ $(M_\odot/pc^2)$',fontsize=25)
ax2.set_ylabel(r'$\Sigma_*$ $(M_\odot/yr/kpc^2)$',fontsize=25)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1e-1,1e4)
ax2.set_ylim(1e-5,1e1)
ax2.legend(frameon=False, loc='lower right')
#plt.show()
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/ksH2_compile.pdf')
plt.show()
