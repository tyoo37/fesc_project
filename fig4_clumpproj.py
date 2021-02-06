import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

read_old_01Zsun = '/Volumes/THYoo/RHD_10pc/'

read_old_002Zsun = '/Volumes/THYoo/RHD_10pc_lowZ/'

read_new_01Zsun = '/Volumes/THYoo/kisti/RHD_10pc_0.1Zsun/'

read_new_1Zsun = '/Volumes/THYoo/kisti/RHD_10pc_1Zsun/'

read_new_01Zsun_re = '/blackwhale/dbahck37/kisti/0.1Zsun/'

read_new_1Zsun_re = '/blackwhale/dbahck37/kisti/1Zsun/'

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
        sfrindex = np.where((tp >= 0) & (tp < 10))[0]
        self.SFR = np.sum(self.mp0[sfrindex]) / 1e7


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
        nHI = self.nH * celldata.cell[0][4][7]
        nHII = self.nH * celldata.cell[0][4][8]
        nH2 = self.nH * (1 - celldata.cell[0][4][7] - celldata.cell[0][4][8])/2
        YY= 0.24/(1-0.24)/4
        nHeII = self.nH * YY*celldata.cell[0][4][9]
        nHeIII = self.nH * YY*celldata.cell[0][4][10]
        nHeI = self.nH * YY*(1 - celldata.cell[0][4][9] - celldata.cell[0][4][10])
        ne = nHII + nHeII + nHeIII *2
        ntot = nHI + nHII + nHeI + nHeII + nHeIII + ne + nH2
        mu = celldata.cell[0][4][0] * Part.unit_d / 1.66e-24 / ntot

        self.T = celldata.cell[0][4][5]/celldata.cell[0][4][0] * 517534.72 * mu



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

    def __init__(self, Part, Cell, dir, nout, xwid ,ywid,zwid, xcenter,ycenter,zcenter):
        start = time.time()
        ywid=xwid
        mindx = np.min(Cell.dx)

        self.dir = dir
        self.nout = nout
        self.xwid=xwid/mindx
        self.ywid=ywid/mindx
        self.zwid=zwid/mindx


        print('reading finished , t = %.2f [sec]' %(time.time()-start))

        start= time.time()


        pc = 3.08e18


        maxgrid = int(np.log2(np.max(Cell.dx) / mindx))

        xind = Cell.x / mindx - 0.5
        yind = Cell.y / mindx - 0.5
        zind = Cell.z / mindx - 0.5

        #center = int(Part.boxpc / 2 / mindx)
        xcenter = int(xcenter/mindx)
        ycenter = int(ycenter/mindx)
        zcenter = int(zcenter/mindx)

        self.xfwd = 2 * int(self.xwid)
        self.yfwd = 2 * int(self.ywid)
        self.zfwd = 2 * int(self.zwid)

        xini = xcenter - xwid
        yini = ycenter - ywid
        zini = zcenter - zwid
        xfin = xcenter + xwid
        yfin = ycenter + ywid
        zfin = zcenter + zwid
        # print(max(self.Cell.cell[0][4][0]))

        self.leafcell = np.zeros((self.xfwd, self.yfwd, self.zfwd))

        ind_1 = np.where((Cell.dx == mindx) & (xind >= xini) & (xind < xfin)
                         & (yind >= yini) & (yind < yfin) & (zind >= zini) & (zind < zfin))[0]

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
                        ##   print('zerocell')
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



        print('Calculation for discomposing , t = %.2f [sec]' %(time.time()-start))


    def projectionPlot(self, Cell, ax, cm, direction, field):
        start=time.time()
        if field == 'nH':
            var = Cell.nH
            if direction == 'xy':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=2) / self.zfwd)

            if direction == 'yz':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=0) / self.xfwd)

            if direction == 'zx':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=1) / self.yfwd)
            cax = ax.imshow(np.rot90(plane), cmap=cm,
                            extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000, -self.ywid * self.mindx / 1000,
                                    self.ywid * self.mindx / 1000], vmin=-3, vmax=2, aspect='auto')

        if field == 'T':
            var = Cell.T
            if direction == 'xy':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)]*Cell.nH[self.leafcell[:, :, :].astype(int)], axis=2) / np.sum(Cell.nH[self.leafcell[:, :, :].astype(int)], axis=2)/self.zfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                            extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000, -self.ywid * self.mindx / 1000,
                                    self.ywid * self.mindx / 1000], vmin=1, vmax=7)
            if direction == 'yz':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)]*Cell.nH[self.leafcell[:, :, :].astype(int)], axis=0) / np.sum(Cell.nH[self.leafcell[:, :, :].astype(int)], axis=0)/self.zfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                                extent=[-self.ywid * 9.1 / 1000, self.ywid * 9.1 / 1000, -self.zwid * 9.1 / 1000,
                                        self.zwid * 9.1 / 1000], vmin=1, vmax=7)
            if direction == 'zx':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)]*Cell.nH[self.leafcell[:, :, :].astype(int)], axis=1) / np.sum(Cell.nH[self.leafcell[:, :, :].astype(int)], axis=1)/self.zfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                            extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx/ 1000, -self.ywid * self.mindx / 1000,
                                    self.ywid * self.mindx / 1000], vmin=1, vmax=7)

        print('projection finished , t = %.2f [sec]' %(time.time()-start))
        cbaxes = inset_axes(ax, width="30%", height="3%", loc=3)
        cbar = plt.colorbar(cax, cax=cbaxes, ticks=[-3, 2], orientation='horizontal', cmap=cm)
        cbar.set_label('nH', color='w', labelpad=-11, fontsize=10)
        cbar.ax.xaxis.set_tick_params(color='w')
        cbar.ax.xaxis.set_ticks_position('top')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')
        rectangles = {'500 pc': patches.Rectangle(xy=(1/6 * 2*self.xwid*self.mindx / 1000, -0.97 * self.xwid*self.mindx / 1000),
                                                 width=(2 * self.xwid*self.mindx / 1000) / 6.0,
                                                 height=0.01 * (2 * self.ywid *self.mindx/ 1000), facecolor='white')}
        for r in rectangles:
            ax.add_artist(rectangles[r])
            rx, ry = rectangles[r].get_xy()
            cx = rx + rectangles[r].get_width() / 2.0
            cy = ry + rectangles[r].get_height() / 2.0

            ax.annotate(r, (cx, cy+0.03 * (2 * self.ywid*self.mindx / 1000)), color='w', weight='bold',
                        fontsize=10, ha='center', va='center')
        return cax

    def star_plot(self, Part, ax, young):

        start=time.time()

        print('star plotting...')
        sxind = Part.xp[0] / self.mindx - self.xcenter + self.xwid
        syind = Part.xp[1] / self.mindx - self.ycenter + self.ywid
        szind = Part.xp[2] / self.mindx - self.zcenter + self.zwid
       # sind = np.where(
      #      (sxind >= 3) & (sxind < self.xfwd - 3) & (syind >= 3) & (syind < self.yfwd - 3) & (szind >= 3) & (szind < self.zfwd - 3))[
      #      0]
        # sind = np.where((sxind >= 0) & (sxind < xfwd) & (syind >= 0) & (syind < yfwd) & (szind >= 0) & (
        # szind < zfwd))[0]
        if young ==True:
            sind = np.where(Part.starage<10)
            sxind = sxind[sind]
            syind = syind[sind]
            szind = szind[sind]

        sxplot = (sxind - self.xwid) * self.mindx
        syplot = (syind - self.ywid) * self.mindx
        szplot = (szind - self.zwid) * self.mindx
        cax1 = ax.scatter(sxplot/1000, syplot/1000,  c='grey', s=1,alpha=0.7)

        ax.set_xlim(-self.xwid * self.mindx / 1000, self.xwid * self.mindx/ 1000)
        ax.set_ylim(-self.ywid * self.mindx / 1000, self.ywid * self.mindx/ 1000)

        print('plotting stars finished , t = %.2f [sec]' %(time.time()-start))
        return cax1


    def clump_plot(self,Clump,ax):
        #for appropriate description of size of clump, dpi = 144, figsize * size of axis = size
        size=5.6
        # clump finding
        start=time.time()
        print('finding gas clumps...')


        xclumpind = Clump.xclump / self.mindx - self.xcenter + self.xwid
        yclumpind = Clump.yclump / self.mindx - self.ycenter + self.ywid
        zclumpind = Clump.zclump / self.mindx - self.zcenter + self.zwid



        xclumpplot = (xclumpind - self.xwid) * self.mindx
        yclumpplot = (yclumpind - self.ywid) * self.mindx
        zclumpplot = (zclumpind - self.zwid) * self.mindx

        cax1 = ax.scatter(xclumpplot/1000, yclumpplot/1000 ,edgecolor='k', marker='o', s=(Clump.rclump*144*size/self.mindx/self.xfwd)**2,linewidths=1,facecolors='none')
        ax.set_xlim(-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000)
        ax.set_ylim(-self.ywid * self.mindx / 1000, self.ywid * self.mindx / 1000)
        return cax1

def plot(dir, inisnap, endsnap):
    numsnap = endsnap - inisnap + 1

    for i in range(numsnap):
        nout = i + inisnap
        fig = plt.figure(figsize=(7, 7))
        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        #cax = fig.add_axes([0.9, 0.1, 0.02, 0.24])

        Part1 = Part(dir, nout)
        Cell1 = Cell(dir, nout, Part1)
        Clump1 = Clump(dir, nout, Part1)
        cm = plt.get_cmap('inferno')
        boxpc = Part1.boxpc
        xcenter = boxpc/2
        ycenter = boxpc/2
        zcenter = np.sum(Cell1.z*Cell1.nH*Cell1.dx**3)/np.sum(Cell1.nH*Cell1.dx**3)
        a = GenerateArray(Part1, Cell1, dir, nout, 1500, 1500, 2500, xcenter, ycenter, zcenter)
        ss = a.projectionPlot(Cell1, ax1, cm, 'xy', 'nH')
        #ax1.set_xlim(a.xwid*(-9.1)/100,a.xwid*(-9.1)/100)
        #ax1.set_ylim(a.xwid*(-9.1)/100,a.xwid*(-9.1)/100)
        cc= a.star_plot(Part1, ax1, False)
        aa= a.clump_plot(Clump1, ax1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        #a.set_facecolor('none')
        #cbar = plt.colorbar(ss, cmap=cm)
       # cbar.set_label('nH')
        #ax1.set_xlabel('X(kpc)')
       # ax1.set_ylabel('Y(kpc)')
        plt.savefig('/Volumes/THYoo/plot/2019thesis/fig4_clump_proj.pdf')
        plt.show()


plot(read_new_01Zsun,127,127)
