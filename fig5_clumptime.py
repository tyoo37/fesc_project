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
        self.mp = partdata.star.mp[0] * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l

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

        self.T = celldata.cell[0][4][5]/celldata.cell[0][4][0] * 517534.72 * mu
        self.xHI = celldata.cell[0][4][7]

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

class YoungClumpFind():
    def __init__(self,Part,Clump,dir,snap):
        #generating star informaton belongs to clump
        self.dir=dir

        nmemberarr = np.array([])
        staridarr = np.array([])
        fescarr = np.array([])
        starmassarr=np.array([])
        nmembersort=np.array([])
        self.youngclumpindex=[]
        prevPart = Part(dir,snap-1)

        Part1 = Part(dir,snap)
        Clump1 = Clump(dir,snap,Part1)
        nclump = len(Clump1.xclump)

        for i in range(nclump):
            rr = np.sqrt((Clump1.xclump[i] - Part1.xp[0]) ** 2 + (Clump1.yclump[i] - Part1.xp[1]) ** 2 + (Clump1.zclump[i] - Part1.xp[2]) ** 2)
            youngindex = np.where((rr < Clump1.rclump[i]) & (Part1.starage < Part1.snaptime - prevPart.snaptime))
            preexistindex = np.where((rr<Clump1.rclump[i]) & (Part1.starage>=Part1.snaptime - prevPart.snaptime))
            if len(youngindex[0])>0 and len(preexistindex[0])==0:
                nmember = len(youngindex[0])
                nmemberarr = np.append(nmemberarr, nmember)


                starid = Part1.starid[youngindex]
                staridarr = np.append(staridarr, starid)

                starmass = np.sum(Part1.mp0[youngindex])
                starmassarr=np.append(starmassarr,starmass)
                self.youngclumpindex.append(i)


        self.nmember=nmemberarr
        self.starid=staridarr
        self.fesc=fescarr
        self.sf=starmassarr
        cum_nmember=np.zeros(len(self.nmember))
        for j in range(len(self.nmember)):
            cum_nmember[j]=np.sum(self.nmember[:j+1])
        self.cum_nmember=cum_nmember


class GenerateArray():

    def __init__(self, Part, Cell, dir, nout, xwid ,zwid, xcenter,ycenter,zcenter):
        start = time.time()
        ywid=xwid
        self.dir = dir
        self.nout = nout
        self.xwid=xwid
        self.ywid=xwid
        self.zwid=zwid

        print('reading finished , t = %.2f [sec]' %(time.time()-start))

        start= time.time()


        pc = 3.08e18

        mindx = np.min(Cell.dx)

        maxgrid = int(np.log2(np.max(Cell.dx) / mindx))

        xind = Cell.x / mindx - 0.5
        yind = Cell.y / mindx - 0.5
        zind = Cell.z / mindx - 0.5

        #center = int(Part.boxpc / 2 / mindx)
        xcenter = int(xcenter/mindx)
        ycenter = int(ycenter/mindx)
        zcenter = int(zcenter/mindx)

        self.xfwd = 2 * int(xwid)
        self.yfwd = 2 * int(ywid)
        self.zfwd = 2 * int(zwid)

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

        self.xwid = xwid
        self.ywid = ywid
        self.zwid = zwid

        print('Calculation for discomposing , t = %.2f [sec]' %(time.time()-start))


    def projectionPlot(self, Cell, ax, cm, direction, field, rullen, ruler,rulerswitch,vmin,vmax,rulercolor):
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
                            extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000, -self.ywid * self.mindx/ 1000,
                                    self.ywid * self.mindx/ 1000], vmin=vmin, vmax=vmax, aspect='auto')

        if field == 'nHI':
            var = Cell.nHI
            if direction == 'xy':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=2) / self.zfwd)

            if direction == 'yz':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=0) / self.xfwd)

            if direction == 'zx':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=1) / self.yfwd)
            print('minmax',np.min(plane),np.max(plane))
            cax = ax.imshow(np.rot90(plane), cmap=cm,
                            extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000,
                                    -self.ywid * self.mindx / 1000,
                                    self.ywid * self.mindx / 1000], vmin=vmin, vmax=vmax, aspect='auto')
        if field == 'xHI':
            var = Cell.xHI
            if direction == 'xy':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=2) / self.zfwd)

            if direction == 'yz':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=0) / self.xfwd)

            if direction == 'zx':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)], axis=1) / self.yfwd)
            print('minmax',np.min(plane),np.max(plane))
            cax = ax.imshow(np.rot90(plane), cmap=cm,
                            extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000,
                                    -self.ywid * self.mindx / 1000,
                                    self.ywid * self.mindx / 1000], vmin=vmin, vmax=vmax, aspect='auto')
        if field == 'T':
            var = Cell.T
            if direction == 'xy':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)]*Cell.nH[self.leafcell[:, :, :].astype(int)], axis=2) / np.sum(Cell.nH[self.leafcell[:, :, :].astype(int)], axis=2)/self.zfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                            extent=[-self.xwid * self.mindx / 1000, self.xwid * self.mindx / 1000, -self.ywid *self.mindx / 1000,
                                    self.ywid * self.mindx / 1000], vmin=vmin, vmax=vmax)
            if direction == 'yz':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)]*Cell.nH[self.leafcell[:, :, :].astype(int)], axis=0) / np.sum(Cell.nH[self.leafcell[:, :, :].astype(int)], axis=0)/self.zfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                                extent=[-self.ywid * 9.1 / 1000, self.ywid * 9.1 / 1000, -self.zwid * 9.1 / 1000,
                                        self.zwid * 9.1 / 1000], vmin=1, vmax=7)
            if direction == 'zx':
                plane = np.log10(np.sum(var[self.leafcell[:, :, :].astype(int)]*Cell.nH[self.leafcell[:, :, :].astype(int)], axis=1) / np.sum(Cell.nH[self.leafcell[:, :, :].astype(int)], axis=1)/self.zfwd)
                cax = ax.imshow(np.rot90(plane), cmap=cm,
                            extent=[-self.xwid * 9.1 / 1000, self.xwid * 9.1 / 1000, -self.ywid * 9.1 / 1000,
                                    self.ywid * 9.1 / 1000], vmin=1, vmax=7)

        print('projection finished , t = %.2f [sec]' %(time.time()-start))
        if rulerswitch==True:
            cbaxes = inset_axes(ax, width="35%", height="5%", loc=2)
            cbar = plt.colorbar(cax, cax=cbaxes, ticks=[vmin, vmax], orientation='horizontal', cmap=cm)
            cbar.set_label('log '+field, color=rulercolor, labelpad=-0.4, fontsize=10)

            cbar.ax.xaxis.set_tick_params(color=rulercolor,labelsize=10)
            cbar.ax.xaxis.set_ticks_position('bottom')
            plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')


            rectangles = {
                '%d pc'%ruler: patches.Rectangle(xy=(0.5 * self.xwid * self.mindx / 1000, -0.96 * self.xwid * self.mindx / 1000),
                                           width=(2 * self.xwid * self.mindx / 1000) *rullen,
                                           height=0.03 * (2 * self.ywid * self.mindx / 1000), facecolor=rulercolor)}
            for r in rectangles:
                ax.add_artist(rectangles[r])
                rx, ry = rectangles[r].get_xy()
                cx = rx + rectangles[r].get_width() / 2.0
                cy = ry + rectangles[r].get_height() / 2.0

                ax.annotate(r, (cx, cy + 0.05 * (2 * self.ywid * self.mindx / 1000)), color=rulercolor, weight='bold',
                            fontsize=10, ha='center', va='center')
        return cax

    def star_plot(self, Part, ax,index,color):

        start=time.time()

        print('star plotting...')
        sxind = Part.xp[0] / self.mindx - self.xcenter + self.xwid
        syind = Part.xp[1] / self.mindx - self.ycenter + self.ywid
        szind = Part.xp[2] / self.mindx - self.zcenter + self.zwid
        sind = np.where(
            (sxind >= 3) & (sxind < self.xfwd - 3) & (syind >= 3) & (syind < self.yfwd - 3) & (szind >= 3) & (szind < self.zfwd - 3))[
            0]
        # sind = np.where((sxind >= 0) & (sxind < xfwd) & (syind >= 0) & (syind < yfwd) & (szind >= 0) & (
        # szind < zfwd))[0]
        sxind = sxind[sind]
        syind = syind[sind]
        szind = szind[sind]

        sxplot = (sxind - self.xwid) * self.mindx
        syplot = (syind - self.ywid) * self.mindx
        szplot = (szind - self.zwid) * self.mindx
        #cax1 = ax.scatter(sxplot/1000, syplot/1000,  c='grey', s=1,alpha=0.7)

        sxind = Part.xp[0][index.astype(int)] / self.mindx - self.xcenter + self.xwid
        syind = Part.xp[1][index.astype(int)] / self.mindx - self.ycenter + self.ywid
        szind = Part.xp[2][index.astype(int)] / self.mindx - self.zcenter + self.zwid
        #sind = np.where(
        #    (sxind >= 1) & (sxind < self.xfwd - 1) & (syind >= 3) & (syind < self.yfwd - 3) & (szind >= 1) & (
        #                szind < self.zfwd - 1))[
        #    0]
        # sind = np.where((sxind >= 0) & (sxind < xfwd) & (syind >= 0) & (syind < yfwd) & (szind >= 0) & (
        # szind < zfwd))[0]
        #sxind = sxind[sind]
        #syind = syind[sind]
        #szind = szind[sind]

        sxplot = (sxind - self.xwid) * self.mindx
        syplot = (syind - self.ywid) * self.mindx
        szplot = (szind - self.zwid) * self.mindx
        cax2 = ax.scatter(sxplot/1000, syplot/1000,  c=color, marker='*',s=50,edgecolor='k',linewidth=0.5)
        ax.set_xlim(-self.xwid*self.mindx/1000,self.mindx*self.xwid/1000)
        ax.set_ylim(-self.ywid*self.mindx/1000,self.mindx*self.ywid/1000)

        print('plotting stars finished , t = %.2f [sec]' %(time.time()-start))
        return cax2


    def clump_plot(self,Clump,ax,member):
        #for appropriate description of size of clump, dpi = 144, figsize * size of axis = size
        size=1.5*9*0.12
        # clump finding
        start=time.time()
        print('finding gas clumps...')


        xclumpind = Clump.xclump / self.mindx - self.xcenter + self.xwid
        yclumpind = Clump.yclump / self.mindx - self.ycenter + self.ywid
        zclumpind = Clump.zclump / self.mindx - self.zcenter + self.zwid

        #clumpind = np.where(
        #    (xclumpind >= 0) & (xclumpind < self.xfwd ) & (yclumpind >= 4) & (yclumpind < self.yfwd - 4) & (zclumpind >= 1) & (
        #                zclumpind < self.zfwd - 1))[0]

        #xclumpind = xclumpind[clumpind]
        #yclumpind = yclumpind[clumpind]
        #zclumpind = zclumpind[clumpind]

        xclumpplot = (xclumpind - self.xwid) * self.mindx
        yclumpplot = (yclumpind - self.ywid) * self.mindx
        zclumpplot = (zclumpind - self.zwid) * self.mindx

        #cax1 = ax.scatter(xclumpplot/1000, yclumpplot/1000 ,edgecolor='k', marker='o', s=(Clump.rclump*144*size/self.mindx/self.xfwd)**2,linewidths=1,facecolors='none')
        cax1 = ax.scatter(xclumpplot/1000, yclumpplot/1000 ,edgecolor='k', marker='o', s=(Clump.rclump * ax.get_window_extent().width / (2 * self.xwid*self.mindx)) ** 2,linewidths=1,facecolors='none')
        xclumpplot=Clump.xclump[member.astype(int)] - self.mindx * self.xcenter
        yclumpplot=Clump.yclump[member.astype(int)] - self.mindx * self.ycenter

        #cax2 = ax.scatter(xclumpplot/1000, yclumpplot/1000 ,edgecolor='purple', marker='o', s=(Clump.rclump[member.astype(int)]*144*size/self.mindx/self.xfwd)**2,linewidths=3,facecolors='none')
        cax2 = ax.scatter(xclumpplot/1000, yclumpplot/1000 ,edgecolor='purple', marker='o', s=(Clump.rclump * ax.get_window_extent().width / (2 * self.xwid*self.mindx)) ** 2,linewidths=1,facecolors='none')

        ax.set_xlim(-self.xwid * self.mindx / 1000, self.mindx * self.xwid / 1000)
        ax.set_ylim(-self.ywid * self.mindx / 1000, self.mindx * self.ywid / 1000)

        return cax1,cax2


class Fesc_new8_dist():
    def __init__(self,dir,nout):
        dat2 = FortranFile(dir + 'ray_nside8_dist/ray_%05d.dat' % (nout), 'r')
        self.npart, nwave2, version = dat2.read_ints()
        wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attHHe = dat2.read_reals(dtype=np.double)
        sed_attHHeD = dat2.read_reals(dtype=np.double)
        sed_attHHI = dat2.read_reals(dtype=np.double)
        sed_attHH2 = dat2.read_reals(dtype=np.double)
        sed_attHHe = dat2.read_reals(dtype=np.double)
        sed_attD = dat2.read_reals(dtype=np.double)

        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        self.fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fescD = self.fescD.reshape(8,self.npart)
        self.fescH = self.fescH.reshape(8,self.npart)
def rjus_selected(read, nout, rjus, index):
    #Fesc_rjus1 = Fesc_rjus(read,nout,rjusval)
    Fesc_rjus1 = Fesc_new8_dist(read,nout)

    fesc =  np.sum(Fesc_rjus1.fescD[rjus,index]*Fesc_rjus1.photonr[index])/np.sum(Fesc_rjus1.photonr[index])
    return fesc

def ClumpTrace(Part,Clump,YoungClumpFind,dir,inisnap,endsnap,trace, wid, dep,rjus,axlist,field,cm,vmin,vmax,rulercolor):

    numsnap = endsnap - inisnap + 1
    for m in range(numsnap):
        nout = inisnap + m

        Init = YoungClumpFind(Part,Clump,dir,nout)
        InitPart=Part(dir,nout)
        InitClump=Clump(dir,nout,InitPart)
        print('xclump = ',InitClump.xclump[Init.youngclumpindex[0]]-InitPart.boxpc/2,'yclump = ',InitClump.yclump[Init.youngclumpindex[0]]-InitPart.boxpc/2)
        #self.initclumpid = np.arange(len(Init.nmember))
        print('the number of star-forming clumps = ', len(Init.youngclumpindex))
        print('the number of stars younger than 1 Myr = ', len(np.where(InitPart.starage < 1)[0]))
        print('the number of young stars in clumps = ', int(np.sum(Init.nmember)))
        for i in range(len(Init.youngclumpindex)):
            if i !=0:
                break
            avgfescarr=np.zeros((len(rjusarr)+1,6))
            timearr=np.array([])
            noroop = 0
            b=-1
            for n in range(trace):

                if n > 9:
                    break
                #Clump1 = YoungClumpFind(Part, Clump, dir, nout)
                traceout = inisnap + n + m
                clusterindexarr = np.array([])

                Part1 = Part(dir, traceout)
                Cell1 = Cell(dir, traceout, Part1)
                Clump1=Clump(dir,traceout,Part1)

                timearr=np.append(timearr,Part1.snaptime-InitPart.snaptime)
                starindexarr=np.array([]) # set of initially detected stars
                clumpindexarr=np.array([])
                if n ==0:
                    staridarr = np.array([])
                    nmember = int(Init.nmember[i])
                else:
                    nmember = len(staridarr)

                for j in range(nmember):

                    if n ==0:
                        starindex = np.unique(np.where(Part1.starid == Init.starid[int(np.sum(Init.nmember[:i]))+j])[0])
                        print(starindex,Init.starid[int(np.sum(Init.nmember[:i]))+j] )
                    else:
                        starindex = np.unique(np.where(Part1.starid == staridarr[j])[0])
                        print(starindex)
                    starindexarr = np.append(starindexarr, starindex)

                    if noroop==0:
                        xstar = Part1.xp[0][int(starindex)]
                        ystar = Part1.xp[1][int(starindex)]
                        zstar = Part1.xp[2][int(starindex)]
                        rr = np.sqrt((Clump1.xclump - xstar) ** 2 + (Clump1.yclump - ystar) ** 2 + (
                                Clump1.zclump - zstar) ** 2)
                        clumpindex = np.where((rr - Clump1.rclump < 0))[0]
                        #starindexarr = np.append(starindexarr, starindex)
                        clumpindexarr = np.append(clumpindexarr, clumpindex)

                if len(np.unique(clumpindexarr)) >0 and noroop ==0:
                    print('loop')
                    clusterindexarr=np.append(clusterindexarr,starindexarr)
                    for c in range(len(np.unique(clumpindexarr))):
                        rrr = np.sqrt((Part1.xp[0] - Clump1.xclump[int(clumpindexarr[c])]) ** 2 + (
                                    Part1.xp[1] - Clump1.yclump[int(clumpindexarr[c])]) ** 2 + (
                                                  Part1.xp[2] - Clump1.zclump[int(clumpindexarr[c])]) ** 2)
                        clusterindex = np.where((rrr < Clump1.rclump[int(clumpindexarr[c])]) & (Part1.starage <1))[0]
                        staridset = Part1.starid[np.unique(clusterindex).astype(int)]
                        staridarr = np.append(staridarr,staridset)
                        clusterindexarr = np.append(clusterindexarr, clusterindex) # set of stars included in target clumps
                else:
                    noroop = 1
                    clusterindexarr = starindexarr

                clusterindexarr = np.unique(clusterindexarr)
                staridarr = np.unique(staridarr)

                nmember = len(np.unique(clusterindexarr))
                print(staridarr, clusterindexarr, nmember)

                nextxstar = Part1.xp[0][starindexarr.astype(int)]
                nextystar = Part1.xp[1][starindexarr.astype(int)]
                nextzstar = Part1.xp[2][starindexarr.astype(int)]

                xcenter = np.average(nextxstar)
                ycenter = np.average(nextystar)
                zcenter = np.average(nextzstar)
                print('*********************')
                print('#%d' % i, 'inisnap = ', nout, ' ,', n, 'th snapshot, n=', len(clusterindexarr))
                print(Part1.mp0[starindexarr.astype(int)])
                print(Part1.mp[starindexarr.astype(int)])
                print('*********************')
                if n>=3 and n<=8:
                    b=b+1

                    for ii in range(len(rjus)):
                        avgfescarr[ii, b]=rjus_selected(dir, traceout, rjus[ii], starindexarr.astype(int))
                    aa = Fesc_new8(dir,traceout)
                    avgfescarr[-1, b] = np.sum(aa.fescD[starindexarr.astype(int)]*aa.photonr[starindexarr.astype(int)])/np.sum(aa.photonr[starindexarr.astype(int)])


                    print('*********************')
                    print('#%d' % i, 'inisnap = ',nout,' ,', n,'th snapshot')
                    print(Part1.mp0[starindexarr.astype(int)])
                    print(Part1.mp[starindexarr.astype(int)])
                    print('*********************')

                    a = GenerateArray(Part1, Cell1, dir, nout, wid, dep, xcenter, ycenter, zcenter)
                    if n==3:
                        ss = a.projectionPlot(Cell1, axlist[b], cm, 'xy', field, 100/(a.mindx*2*wid), 100, True,vmin,vmax,rulercolor)
                    else:
                        ss = a.projectionPlot(Cell1, axlist[b], cm, 'xy', field, 100/(a.mindx*2*wid), 100, False,vmin,vmax,rulercolor)
                    dd = a.star_plot(Part1, axlist[b], clusterindexarr,'orange')

                    dd = a.star_plot(Part1, axlist[b], starindexarr,'cyan')

                    aa, bb = a.clump_plot(Clump1, axlist[b], clumpindexarr)
                    print(clumpindexarr)
                    aa.set_facecolor('none')
                    axlist[b].set_xticks([])
                    axlist[b].set_yticks([])

            return ss, avgfescarr, Init.youngclumpindex[0]



plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 10
#plt.style.use('dark_background')

fig = plt.figure(figsize=(9.5*1.2, 7*1.2), dpi=144)

ax1 = fig.add_axes([0.1, 0.4, 0.14, 0.19])
ax2 = fig.add_axes([0.24, 0.4, 0.14, 0.19])
ax3 = fig.add_axes([0.38, 0.4, 0.14, 0.19])
ax4 = fig.add_axes([0.52, 0.4, 0.14, 0.19])
ax5 = fig.add_axes([0.66, 0.4, 0.14, 0.19])
ax6 = fig.add_axes([0.8, 0.4, 0.14, 0.19])

ax11 = fig.add_axes([0.1, 0.59, 0.14, 0.19])
ax22 = fig.add_axes([0.24, 0.59, 0.14, 0.19])
ax33 = fig.add_axes([0.38, 0.59, 0.14, 0.19])
ax44 = fig.add_axes([0.52, 0.59, 0.14, 0.19])
ax55 = fig.add_axes([0.66, 0.59, 0.14, 0.19])
ax66 = fig.add_axes([0.8, 0.59, 0.14, 0.19])

ax111 = fig.add_axes([0.1, 0.78, 0.14, 0.19])
ax222 = fig.add_axes([0.24, 0.78, 0.14, 0.19])
ax333 = fig.add_axes([0.38, 0.78, 0.14, 0.19])
ax444 = fig.add_axes([0.52, 0.78, 0.14, 0.19])
ax555 = fig.add_axes([0.66, 0.78, 0.14, 0.19])
ax666 = fig.add_axes([0.8, 0.78, 0.14, 0.19])

ax1.set_ylabel('gas\ndensity',rotation=90,fontsize=15)
ax11.set_ylabel('neutral\nfraction',rotation=90,fontsize=15)
ax111.set_ylabel('temperature',rotation=90,fontsize=15)
ax7 = fig.add_axes([0.1, 0.1, 0.84, 0.2])
rjusarr=[0,1,2]
axlist=[ax1,ax2,ax3,ax4,ax5,ax6]
axlist2=[ax11,ax22,ax33,ax44,ax55,ax66]
axlist3=[ax111,ax222,ax333,ax444,ax555,ax666]

cm1 = plt.get_cmap('inferno')
cm2 = plt.get_cmap('summer_r')
cm3 = plt.get_cmap('YlGnBu_r')

ss, fescarr, clumpindex = ClumpTrace(Part,Clump,YoungClumpFind,read_new_01Zsun,124,124,30,30,15,rjusarr,axlist,'nH',cm1,-3,2,'w')
ss2, fescarr2, clumpindex2 = ClumpTrace(Part,Clump,YoungClumpFind,read_new_01Zsun,124,124,30,30,15,rjusarr,axlist2,'xHI',cm2,-4,0,'k')
ss3, fescarr3, clumpindex3 = ClumpTrace(Part,Clump,YoungClumpFind,read_new_01Zsun,124,124,30,30,15,rjusarr,axlist3,'T',cm3,1,7,'k')

timearr = np.arange(6)+2
strarr = ['$f_{esc,80pc}$','$f_{esc,200pc}$','$f_{esc,vir}$']
for i in range(3):
    ax7.plot(timearr, fescarr[i,:], label=strarr[i],marker='s')
#cax1 = fig.add_axes([0.89, 0.3, 0.01, 0.6])
cm = plt.get_cmap('inferno')

#cbar1 = plt.colorbar(ss, cax=cax1, cmap=cm)
#cbar1.set_label('nH')

ax7.set_ylim(0,1)
ax7.set_xlim(1.5,7.5)
#ax1.set_ylabel('Y(kpc)')
ax7.set_ylabel('$f_{esc}$',fontsize=25)
ax7.set_xlabel('time elapsed since star formation (Myr)',fontsize=20)
ax7.legend()
"""
fig2=plt.figure(figsize=(9,9),dpi=144)

ax1=fig2.add_axes([0.15,0.15,0.7,0.7])
#cax=fig2.add_axes([0.85,0.15,0.01,0.7])

cm = plt.get_cmap('inferno')

Part1 = Part(read_new_01Zsun,127)
Cell1 = Cell(read_new_01Zsun,127,Part1)
Clump1 = Clump(read_new_01Zsun,127,Part1)
zcenter1 = np.sum(Cell1.z*Cell1.nH*Cell1.dx**3)/np.sum(Cell1.nH*Cell1.dx**3)

a = GenerateArray(Part1, Cell1, read_new_01Zsun, 127, 180, 200,Part1.boxpc/2,Part1.boxpc/2, zcenter1)
ss = a.projectionPlot(Cell1, ax1, cm, 'xy', 'nH',500/(180*a.mindx*2), 500, True,-3,2,'w')
#cbar= plt.colorbar(ss,cmap=cm,cax=cax)
#cbar.set_label('nH')
a.star_plot(Part1, ax1, np.array([]))
#print(clumpindex)
a.clump_plot(Clump1, ax1, np.array([]))
#ax1.set_xlabel('X(kpc)')
#ax1.set_ ylabel('Y(kpc)')
ax1.set_xticks([])
ax1.set_yticks([])
#plt.show()
"""
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig4.pdf')
plt.show()