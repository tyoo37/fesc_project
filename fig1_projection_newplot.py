import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import os.path
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import FlatLambdaCDM
from astropy.visualization import ZScaleInterval,ImageNormalize
import matplotlib.pyplot as plt
from skimage.transform import rescale
#from skimage import data, color
from scipy import stats
from scipy.ndimage import gaussian_filter

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

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] =15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['figure.titlesize'] = 13
#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#plt.switch_backend('agg')
class Part():
    def __init__(self, dir, nout):
        self.dir = dir
        self.nout = nout
        partdata = readsav(self.dir + '/SAVE/part_%05d.sav' % (self.nout))
        self.snaptime = partdata.info.time*4.70430e14/365/3600/24/1e6
        pc = 3.08e18
        self.boxpc = partdata.info.boxlen * partdata.info.unit_l / pc
        self.boxlen = partdata.info.boxlen
        xp = partdata.star.xp[0]
        self.xp = xp * partdata.info.unit_l / 3.08e18
        self.unit_l = partdata.info.unit_l
        self.unit_d = partdata.info.unit_d
        self.starage = self.snaptime - partdata.star.tp[0]*4.70430e14/365/3600/24/1e6
        self.starid = np.abs(partdata.star.id[0])
        self.mp0 = partdata.star.mp0[0] * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l

        tp = (partdata.info.time-partdata.star.tp[0]) * 4.70430e14 / 365. /24./3600/1e6
        sfrindex = np.where((tp >= 0) & (tp < 10))[0]
        self.SFR = np.sum(self.mp0[sfrindex]) / 1e7
        self.dmxp = partdata.part.xp[0] * partdata.info.unit_l / 3.08e18
        dmm = partdata.part.mp[0]

        self.dmm = dmm * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l * partdata.info.unit_l
        dmindex = np.where(dmm > 2000)
        self.dmxpx = self.dmxp[0][dmindex]
        self.dmxpy = self.dmxp[1][dmindex]
        self.dmxpz = self.dmxp[2][dmindex]


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

class Fesc_indSED():
    def __init__(self, dir, nout):
        dat2 = FortranFile(dir + 'ray_indSED/ray_%05d.dat' % (nout), 'r')
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
        idp = dat2.read_ints()
        self.rflux = dat2.read_reals(dtype=np.double)
        self.gflux = dat2.read_reals(dtype=np.double)
        self.bflux = dat2.read_reals(dtype=np.double)

        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
        self.fesc2 = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)

        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)


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
        self.xHI =celldata.cell[0][4][7]
        self.xHII=celldata.cell[0][4][8]
        self.xH2 = (1 - celldata.cell[0][4][7] - celldata.cell[0][4][8])/2
        self.lev = np.round(np.log2(1 / celldata.cell.dx[0]), 0).astype(int)
        print('minmax', np.min(self.lev),np.max(self.lev))

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
        self.xHI = var[:,7]
        self.xH2=(1 - var[:,7] - var[:,8])/2
        xHII = var[:,8]
        self.mHIH2 = self.m * (1-xHII)
        self.lev = np.round(np.log2(1/dxc),0)
        self.minlev = round(np.log2(1/np.max(dxc)))
        self.maxlev = round(np.log2(1/np.min(dxc)))

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

class Clump():
    def __init__(self, dir, nout, Part):
        self.dir = dir
        self.nout = nout
        unit_d = Part.unit_d
        unit_l = Part.unit_l
        if dir==read_new_01Zsun_lya0ff_re:
            clumpdata = np.loadtxt(self.dir + '/dat/clump_%05d.txt' % (self.nout),
                                   dtype=np.double)
        else:
            clumpdata = np.loadtxt(self.dir + '/clump3/clump_%05d.txt' % (self.nout),
                               dtype=np.double)

        self.xclump = clumpdata[:, 4] * Part.unit_l / 3.08e18
        self.yclump = clumpdata[:, 5] * Part.unit_l / 3.08e18
        self.zclump = clumpdata[:, 6] * Part.unit_l / 3.08e18

        self.massclump = clumpdata[:,
                    10] * unit_d * unit_l / 1.989e33 * unit_l * unit_l
        self.rclump = (clumpdata[:, 10] / clumpdata[:, 9] / 4 / np.pi * 3) ** (0.333333) * unit_l / 3.08e18
        self.nclump = len(self.xclump)


def minmax(var):
    return np.min(var), np.max(var)

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
                sumvar = Cell.T[ind]*Cell.m[ind]/Cell.dx[ind]**2
                weight = Cell.m[ind]

            elif var=='xHI':
                sumvar = Cell.xHI[ind] * Cell.dx[ind]
                weight = Cell.dx[ind]
            elif var=='xH2':
                sumvar = Cell.xH2[ind] * Cell.dx[ind]
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
                    '5 kpc': patches.Rectangle(xy=(0.25 * self.xwid / 1000, -0.88 * self.ywid / 1000),
                                               width=(2 * self.ywid / 1000) * 5 / 14,
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

    def star_plot(self, Part, ax):

        start=time.time()

        print('star plotting...')

        ex_xcenter = self.xcenter*self.mindx
        ex_ycenter = self.ycenter*self.mindx
        ex_zcenter = self.zcenter*self.mindx

        sxplot = (Part.xp[0] - ex_xcenter)/1000
        syplot = (Part.xp[1] - ex_ycenter)/1000
        szplot = (Part.xp[2] - ex_zcenter)/1000
        if self.projection == 'xy':
            cax1 = ax.scatter(sxplot, syplot, c='grey', s=0.1, alpha=0.3)
            ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
            ax.set_ylim(-self.ywid / 1000, self.ywid / 1000)

        if self.projection == 'xz':
            cax1 = ax.scatter(sxplot, szplot, c='grey', s=0.1, alpha=0.3)
            ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
            ax.set_ylim(-self.zwid / 1000, self.zwid / 1000)

        print('plotting stars finished , t = %.2f [sec]' %(time.time()-start))
        return cax1


    def star_plot3(self, Part, ax,binsize,cm,ticks,vmin,vmax, cbar, ruler,corr):

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
                    '5 kpc': patches.Rectangle(xy=(0.25 * self.xwid / 1000, -0.88 * self.xwid / 1000),
                                               width=(2 * self.xwid / 1000) * 5 / 14,
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
                    '5 kpc': patches.Rectangle(xy=(0.25 * self.xwid / 1000, -0.88 * self.zwid / 1000),
                                               width=(2 * self.xwid / 1000) * 5/ 14,
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

def zscale(arr):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(arr)

    return vmin, vmax

def CoM_check_plot(Part1, Cell1, wid, height, depth, xcen,ycen,zcen):
    fig = plt.figure(figsize=(8, 8),dpi=144)

    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    cm1 = plt.get_cmap('rainbow')

    a =  new_projection(Cell1, xcen,ycen,zcen, wid, height, depth,'nH','xy')
    ss1 = a.projectionPlot(Cell1, ax1, cm1)
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
    x2, y2, z2 = CoM_pre(Part1,Cell1,rgrid1,diskmass,x1,y1,z1,True)
    x3, y3, z3 = CoM_pre(Part1,Cell1,rgrid1,diskmass,x2,y2,z2,True)

    #print(x2,y2,z2)
    return x3, y3, z3
def CoM_Main(Part1, diskmass):
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
    return xcen, ycen, zcen, hmr



def plot(ax1, ax2, read, nout, Part, Cell, xwid, ywid, zwid, label,diskmass):

    start = time.time()
    Part1 = Part(read, nout)
    Cell1 = Cell(read, nout, Part1)
    print('reading finished , t = %.2f [sec]' % (time.time() - start))

    xcen, ycen, zcen = CoM_main(Part1, Cell1,diskmass)
    print('finish to find CoM, t= %.2f [sec]'%(time.time()-start))
    a = new_projection( Cell1,  xcen, ycen, zcen,xwid, ywid, zwid, 'nH','xy')
    cm = plt.get_cmap('inferno')
    ss = a.projectionPlot(ax1, cm)
    ax1.set_xlim(-a.xwid/1000,a.xwid/1000)
    ax1.set_ylim(-a.xwid/1000,a.xwid/1000)
    ax2.set_xlim(-a.xwid / 1000, a.xwid / 1000)
    ax2.set_ylim(-a.xwid / 1000, a.xwid / 1000)
    ax1.text(-a.xwid/1000*0.8,a.xwid/1000*0.8,label,color='white')
    cc= a.star_plot(Part1, ax2)
    # aa= a.clump_plot(Clump1, ax1)

    # a.set_facecolor('none')
    #ax.set_xlabel('X(kpc)')
    #ax.set_ylabel('Y(kpc)')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    return ss

def main():
    fig = plt.figure(figsize=(15,12))
    ax1 = fig.add_axes([0.05,0.1,0.2,0.25])
    ax2 = fig.add_axes([0.25,0.1,0.2,0.25])
    ax3 = fig.add_axes([0.05,0.4,0.2,0.25])
    ax4 = fig.add_axes([0.25,0.4,0.2,0.25])
    ax5 = fig.add_axes([0.05,0.7,0.2,0.25])
    ax6 = fig.add_axes([0.25,0.7,0.2,0.25])

    ax7 = fig.add_axes([0.5,0.4,0.2,0.25])
    ax8 = fig.add_axes([0.7,0.4,0.2,0.25])
    ax9 = fig.add_axes([0.5,0.7,0.2,0.25])
    ax10 = fig.add_axes([0.7,0.7,0.2,0.25])
    ax11 = fig.add_axes([0.5,0.1,0.2,0.25])
    ax12 = fig.add_axes([0.7,0.1,0.2,0.25])

    ss = plot(ax11, ax12,read_new_01Zsun_05pc,380,Part,Cellfromdat,7000,7000,3000,'G9_01Zsun_5pc',1.75e9)


    ss = plot(ax5, ax6,read_new_01Zsun,480,Part,Cell,7000,7000,3000,'G9_01Zsun',1.75e9)
    ss = plot(ax7, ax8,read_new_1Zsun_highSN_new,380,Part,Cell,7000,7000,3000,'G9_1Zsun_SNboost',1.75e9)
    ss = plot(ax1, ax2,read_new_gasrich,250,Part,Cellfromdat,7000,7000,7000,'G9_01Zsun_gasrich',1.15e10)
    ss = plot(ax9, ax10,read_new_03Zsun_highSN,380,Part,Cell,7000,7000,3000,'G9_03Zsun_SNboost',1.75e9)
    ss = plot(ax3, ax4,read_new_1Zsun,480,Part,Cell,7000,7000,3000,'G9_1Zsun',1.75e9)


    #plt.savefig('/Volumes/THYoo/2019_thesis/projection(fig1).png' )
    plt.show()


def thr_fiducial(read,nout,Cell,savedirec,dpi,label):

    start = time.time()

    fig = plt.figure(figsize=(10*3, 2.5*3),dpi=dpi)

    ax1 = fig.add_axes([0, 0, 0.25, 1])
    ax3 = fig.add_axes([0.25, 0, 0.25, 1])
    ax2 = fig.add_axes([0.75, 0, 0.25, 1])
    ax4 = fig.add_axes([0.5,0,0.25,0.5])
    ax5 = fig.add_axes([0.5,0.5,0.25,0.5])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    ax5.set_xticks([])

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax5.set_yticks([])

    ax3.set_facecolor('black')



    Part1 = Part(read, nout)
    Cell1 = Cell(read, nout, Part1)
#    Fesc_indSED1 = Fesc_indSED(read, nout)
    print('reading finished , t = %.2f [sec]' % (time.time() - start))

    xcen, ycen, zcen = CoM_main(Part1, Cell1, 1.75e9)
    print('finish to find CoM, t= %.2f [sec]'%(time.time()-start))

    a = new_projection(Cell1, xcen, ycen, zcen, 7000,7000, 3500, 'nH', 'xy')
    cm1 = plt.get_cmap('inferno')
    cm2 = plt.get_cmap('viridis_r')
    cm3 = plt.get_cmap('bone')

    a.projectionPlot(ax1,cm1,[-3,-2,-1,0,1,2],True,True,1,True,label)
    b = new_projection(Cell1, xcen, ycen, zcen, 7000, 7000, 100, 'xHI', 'xy')
    b.projectionPlot(ax2,cm2,[-2, -1,0],True,True,1,False,label)
    filter = [[7950/4, 10050/4],[17550/4, 22260/4],[24160/4,31270/4]]
    a.star_plot3(Part1,ax3,a.mindx,cm3,[6,7,8],5.5,8.5,True,True,1)
    c = new_projection(Cell1, xcen, ycen, zcen, 7000, 7000, 3500, 'nH', 'xz')
    c.projectionPlot(ax5, cm1, [-3, -2, -1, 0, 1, 2],False,True,2,False,label)
    c.star_plot3(Part1,ax4,a.mindx,cm3,[6,7,8],5.5,8.5,False,True,1)
    plt.savefig(savedirec+'thr_fiducial%s_%05d.png'%(label,nout))
    #plt.show()
def thr_fiducial_mol(read,nout,Cell,savedirec,dpi,label):

    start = time.time()

    fig = plt.figure(figsize=(10*3, 2*3),dpi=dpi)

    ax1 = fig.add_axes([0, 0, 0.2, 1])
    ax3 = fig.add_axes([0.2, 0, 0.2, 1])
    ax2 = fig.add_axes([0.6, 0, 0.2, 1])
    ax4 = fig.add_axes([0.4,0,0.2,0.5])
    ax5 = fig.add_axes([0.4,0.5,0.2,0.5])
    ax6 = fig.add_axes([0.8,0,0.2,1])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    ax5.set_xticks([])
    ax6.set_xticks([])

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])
    ax3.set_facecolor('black')



    Part1 = Part(read, nout)
    Cell1 = Cell(read, nout, Part1)
    Fesc_indSED1 = Fesc_indSED(read, nout)
    print('reading finished , t = %.2f [sec]' % (time.time() - start))

    xcen, ycen, zcen = CoM_main(Part1, Cell1, 1.75e9)
    print('finish to find CoM, t= %.2f [sec]'%(time.time()-start))

    a = new_projection(Cell1, xcen, ycen, zcen, 7000,7000, 3500, 'nH', 'xy')
    cm1 = plt.get_cmap('inferno')
    cm2 = plt.get_cmap('viridis_r')
    cm3 = plt.get_cmap('bone')
    cm4 = plt.get_cmap('pink')
    a.projectionPlot(ax1,cm1,[-3,-2,-1,0,1,2],True,True,1,True,label)
    #b = new_projection(Cell1, xcen, ycen, zcen, 7000, 7000, 100, 'xHI', 'xy')
    b = new_projection(Cell1, xcen, ycen, zcen, 7000, 7000, 3500, 'xHI', 'xy')

    b.projectionPlot(ax2,cm2,[-2, -1,0],True,True,1,False,label)
    filter = [[7950/4, 10050/4],[17550/4, 22260/4],[24160/4,31270/4]]
    a.star_plot3(Part1,ax3,a.mindx,cm3,[6,7,8],5.5,8.5,True,True,1)
    c = new_projection(Cell1, xcen, ycen, zcen, 7000, 7000, 3500, 'nH', 'xz')
    c.projectionPlot(ax5, cm1, [-3, -2, -1, 0, 1, 2],False,True,2,False,label)
    c.star_plot3(Part1,ax4,a.mindx,cm3,[6,7,8],5.5,8.5,False,True,2)
    d = new_projection(Cell1, xcen, ycen, zcen, 7000, 7000, 3500, 'xH2', 'xy')

    d.projectionPlot(ax6, cm4, [-6,-4, -2, 0],True,True,1,False,label)
    arr = np.loadtxt(read + 'sh_hmr.dat')
    noutarr = arr[:, 0]

    indnout = np.where(noutarr == nout)
    hmr = arr[indnout, 2]
    hmr = np.asscalar(hmr)

    circle = plt.Circle((0, 0), hmr / 1000, edgecolor='w', fill=False)
    ax6.add_artist(circle)

    plt.savefig(savedirec+'thr_fiducial_H2_%s_%05d.png'%(label,nout))

#single(read_new_01Zsun,Cell,3,480,'G9_01Zsun')
#four_fiducial('/blackwhale/dbahck37/kisti/',124,Cell)
thr_fiducial(read_new_01Zsun,480,Cell,'/Volumes/THYoo/kisti/plot/2019thesis/',54,'G9_Zlow')
thr_fiducial(read_new_gasrich,300,Cellfromdat,'/Volumes/THYoo/kisti/plot/2019thesis/',72,'G9_Zlow_gas5')
thr_fiducial(read_new_03Zsun_highSN,380,Cell,'/Volumes/THYoo/kisti/plot/2019thesis/',72,'G9_Zmid_SN5')
thr_fiducial(read_new_01Zsun_05pc,380,Cell,'/Volumes/THYoo/kisti/plot/2019thesis/',72,'G9_Zlow_HR')
thr_fiducial(read_new_1Zsun,480,Cell,'/Volumes/THYoo/kisti/plot/2019thesis/',72,'G9_Zhigh')
thr_fiducial(read_new_1Zsun_highSN_new,380,Cell,'/Volumes/THYoo/kisti/plot/2019thesis/',72,'G9_Zhigh_SN5')


def main(read, Cell, inisnap, endsnap,savedirec):
    numsnap = endsnap - inisnap + 1
    for i in range(numsnap):
        nout = i + inisnap

        if not os.path.isfile(read + '/SAVE/part_%05d.sav' % (nout)):
            print(read + '/SAVE/part_%05d.sav' % (nout))
            continue
        if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
            print(read + '/SAVE/cell_%05d.sav' % (nout))
            continue
        thr_fiducial(read,nout,Cell,savedirec,'G9_Zlow')




