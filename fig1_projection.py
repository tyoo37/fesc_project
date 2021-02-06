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
plt.switch_backend('agg')
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






class GenerateArray2():

    def __init__(self, Cell, dir, nout, xwid ,ywid,zwid, xcenter,ycenter,zcenter, levelmin, levelmax, field, direction,density_weight):
        """
        This class basically divide all the cells into minimum level
        use this for large projected image

        :param Cell: the name of Cell class ex) if you declare Cell1 = Cell(read, nout) then insert Cell1.
        :param dir: directory
        :param nout: snapshot number
        :param xwid: xwidth (in pc scale)
        :param ywid: ywidth
        :param zwid: zwidth
        :param xcenter: xcenter (in pc scale)
        :param ycenter: ycenter
        :param zcenter: zcenter
        :param levelmin: minimum refinement level
        :param levelmax: maximum
        :param field: choose vairable 'nH' or 'T' you can add
        :param direction: projected direction "xy" or "yz" or "zx"
        """
        start1 = time.time()
        start=time.time()

        self.dir = dir
        self.nout = nout
        self.xwid=xwid
        self.ywid=ywid
        self.zwid=zwid

        pc = 3.08e18

        mindx = np.min(Cell.dx)
        self.mindx = mindx
        maxgrid = int(np.log2(np.max(Cell.dx) / mindx))

        xind = Cell.x / mindx - 0.5
        yind = Cell.y / mindx - 0.5
        zind = Cell.z / mindx - 0.5

        #center = int(Part.boxpc / 2 / mindx)
        self.xcenter = int(xcenter/mindx)
        self.ycenter = int(ycenter/mindx)
        self.zcenter = int(zcenter/mindx)

        dlevel = levelmax - levelmin

        xcen_ind_minlev = int(self.xcenter/2**(dlevel))
        ycen_ind_minlev = int(self.ycenter/2**(dlevel))
        zcen_ind_minlev = int(self.zcenter/2**(dlevel))

        xl_ind_minlev = int((xcenter-xwid)/mindx/2**(dlevel))
        xr_ind_minlev = int((xcenter+xwid)/mindx/2**(dlevel))
        yl_ind_minlev = int((ycenter - ywid) / mindx / 2 ** (dlevel))
        yr_ind_minlev = int((ycenter + ywid) / mindx / 2 ** (dlevel))
        zl_ind_minlev = int((zcenter - zwid) / mindx / 2 ** (dlevel))
        zr_ind_minlev = int((zcenter + zwid) / mindx / 2 ** (dlevel))


        xnumiter=xr_ind_minlev-xl_ind_minlev+1
        ynumiter=yr_ind_minlev-yl_ind_minlev+1
        znumiter=zr_ind_minlev-zl_ind_minlev+1

        for i in range(xnumiter):
            for j in range(ynumiter):

                print ('segmentation of box (%d/%d),(%d/%d)'%(i+1,xnumiter,j+1,xnumiter))
                leafcell = np.zeros((2**dlevel, 2**dlevel, znumiter*2**dlevel)) # no truncation in vertical
                xini_minlev = xl_ind_minlev+i
                yini_minlev = yl_ind_minlev+j
                zini_minlev = zl_ind_minlev

                if xini_minlev <0 or yini_minlev <0 or zini_minlev<0:
                    raise ValueError('out of boundary!')


                xini = xini_minlev * 2**dlevel ; yini = yini_minlev * 2 **dlevel ; zini = zini_minlev * 2 **dlevel

                xfin = xini + 2**dlevel
                yfin = yini + 2**dlevel
                zfin = zini + znumiter*2**dlevel
                #print(xini,xfin)

                ind_1 = np.where((Cell.dx == mindx) & (xind >= xini) & (xind < xfin) & (yind >= yini) & (yind < yfin) & (zind >= zini) & (zind < zfin))[0]
                leafcell[xind[ind_1].astype(int) - int(xini), yind[ind_1].astype(int) - int(yini), zind[ind_1].astype(
                    int) - int(zini)] = ind_1.astype(int)
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
                    """
                    ind = np.where(
                        (Cell.dx == mindx * 2 ** (n + 1)) & (xind + Cell.dx / 2 / mindx >= xini) & (
                                    xind - Cell.dx / 2 / mindx < xfin) & (
                                yind + Cell.dx / 2 / mindx >= yini) & (yind - Cell.dx / 2 / mindx < yfin) & (
                                zind + Cell.dx / 2 / mindx >= zini) & (zind - Cell.dx / 2 / mindx < zfin))[0]
                                """
                    ind = np.where((Cell.dx==mindx*2**(n+1))&(xind>=xini)&(xind<xfin)&(yind>=yini)&(yind<yfin)&(zind>=zini)&(zind<zfin))[0]
                    if len(ind)==0:
                        print('no cells in level %d (n = %d), t = %.2f [sec]'%(n+2,0,time.time()-start))
                        start= time.time()
                        continue
                    print(len(ind), len(ind) * (2 ** (n + 1)) ** 3)
                    if 2**(3*n+3) < len(ind):
                        for a in range(2 ** (n + 1)):
                            for b in range(2 ** (n + 1)):
                                for c in range(2 ** (n + 1)):
                                    xx = xind[ind] - xini + mul[a]
                                    yy = yind[ind] - yini + mul[b]
                                    zz = zind[ind] - zini + mul[c]
                                    if len(np.where(leafcell[xx.astype(int), yy.astype(int), zz.astype(int)] != 0)[
                                               0]) > 0:
                                        raise ValueError('overlap in allocation')
                                    leafcell[xx.astype(int), yy.astype(int), zz.astype(int)] = ind
                                    if len(np.where(ind==0)[0])>0:
                                        raise ValueError('zero in ind')
                                    ##   print('zerocell')
                                    nnn = nnn + len(ind)
                        print('level %d grids are allocated (n = %d), t = %.2f [sec]' % (n + 2, len(ind)*2**(3*n+3), time.time() - start))

                    else:
                        mul_lv1 = np.arange(2**(n))+0.5
                        mul_lv2 = np.arange(2**(n))*(-1)-0.5
                        mul_lv = np.zeros(2**(n+1))

                        for k in range(2**n):
                            mul_lv[2*k]=mul_lv1[k]
                            mul_lv[2*k+1]=mul_lv2[k]
                        xxx,yyy,zzz = np.meshgrid(mul_lv,mul_lv,mul_lv)
                        #print(n+1,len(xxx.flatten()))
                        for a in range(len(ind)):
                            xx = xind[ind[a]] - xini +xxx.flatten()
                            yy = yind[ind[a]] - yini +yyy.flatten()
                            zz = zind[ind[a]] - zini +zzz.flatten()
                            #rint(leafcell[xx.astype(int), yy.astype(int), zz.astype(int)])
                            if len(np.where(leafcell[xx.astype(int), yy.astype(int), zz.astype(int)]!=0)[0]) > 0:
                                raise ValueError('overlap in allocation')
                            leafcell[xx.astype(int), yy.astype(int), zz.astype(int)] = ind[a]
                            #print(    leafcell[xx.astype(int), yy.astype(int), zz.astype(int)])
                            if ind[a]==0:
                                print(ind[a])
                            ##   print('zerocell')
                            nnn = nnn + len(xx)
                            #print(2**(3*(n+1)),len(xx))

                        print('level %d grids are allocated (n = %d), t = %.2f [sec]' % (n + 2, len(xx)*len(ind), time.time() - start))
                    start = time.time()
                    nn = nn + nnn
                   # nonzero_lv = len(np.where(leafcell != 0)[0])
                   # print('nonzero_lv,nn',nonzero_lv,nn)
                    """
                
                    if nonzero_lv != nn:
                        print(nonzero_lv, nn)
                        raise ValueError('error')
                    print('level %d grids are allocated(n = %d)' % (n + 2, nnn))
                
                    
                    if nnn == 0:
                        break
                    """
                num_allot = len(ind_1) + nn
                num_boxcell = 2**dlevel*2**dlevel*znumiter*2**dlevel
                #nonzero = len(np.where(leafcell != 0)[0])
                print('the # of allocated cells = ', num_allot )
                print('the # of box cells  = ', num_boxcell)
                #print('the # of non zero cells in the box = ', nonzero)
                print(leafcell.shape)
                #if num_allot!=num_boxcell or num_allot!=nonzero :
                #    raise ValueError('error in allocation')

                #print('minmax',np.min(Cell.nH[leafcell[:,:,:].astype(int)]), np.max(Cell.nH[leafcell[:,:,:].astype(int)]))


                if field == 'nH':
                    var = Cell.nH
                if field == 'T':
                    var = Cell.T
                if field == 'nHI':
                    var = Cell.nHI
                if field == 'xHI':
                    var = Cell.xHI
                if field == 'xHII':
                    var = Cell.xHII
                if field == 'xH2':
                    var = Cell.xH2
                if direction == 'xy':
                    axis=2
                if direction == 'yz':
                    axis=0
                if direction == 'zx':
                    axis=1
                if density_weight == True:
                    plane = np.log10(np.sum(var[leafcell[:, :, :].astype(int)]*Cell.nH[leafcell[:,:,:].astype(int)],axis=axis) / np.sum(Cell.nH[leafcell[:,:,:].astype(int)],axis=axis))


                else:
                    plane = np.log10(np.sum(var[leafcell[:, :, :].astype(int)], axis=axis) / (znumiter*2**dlevel))



                    #print(znumiter*2**dlevel)
                    #print(np.sum(var[leafcell[:, :, :].astype(int)], axis=2))
                    #print(np.log10(np.sum(var[leafcell[:, :, :].astype(int)], axis=2) / (znumiter*2**dlevel)))
                    #print('minmax(plane) = ',np.min(plane),np.max(plane))


                if j==0 :
                    iniplane = plane
                    #print('iniplane',iniplane, iniplane.shape)

                else:
                    iniplane = np.column_stack((iniplane,plane))
                    #print('iniplane',iniplane.shape)
                print('finishing projection, t = %.2f [sec]' % (time.time() - start))
                start = time.time()

            if i==0:
                iniiniplane = iniplane
                #print('iniiniplane', iniiniplane.shape)

            else:
                iniiniplane = np.row_stack((iniiniplane,iniplane))
                #print('iniiniplane', iniiniplane.shape)

        print('finishing patches assembling, t = %.2f [sec]' % (time.time() - start))
        start = time.time()
        # plane truncation for centering
        self.xl_trunc = self.xcenter-int(xwid/mindx)- xl_ind_minlev*(2**dlevel)
        print(self.xcenter,int(xwid/mindx), xl_ind_minlev*(2**dlevel) )
        self.xr_trunc = self.xcenter+int(xwid/mindx)- xl_ind_minlev*(2**dlevel)
        self.yl_trunc = self.ycenter - int(ywid / mindx) -  yl_ind_minlev*(2**dlevel)
        self.yr_trunc = self.ycenter + int(ywid / mindx) -  yl_ind_minlev*(2**dlevel)
        print(self.xl_trunc, self.xr_trunc, self.yl_trunc, self.yr_trunc)
        #print(iniiniplane)
        self.plane = iniiniplane[self.xl_trunc:self.xr_trunc,self.yl_trunc:self.yr_trunc]
        print(np.min(self.plane),np.max(self.plane))
        #print(self.plane)

        #print(self.plane.shape)
        print('finisihing truncation, t=%.2f [sec]'%(time.time() - start))
        print('Calculation for discomposing , t = %.2f [sec]' %(time.time()-start1))

        self.field = field

    def projectionPlot(self, ax, cm, ticks):
        start=time.time()
        im= ax.imshow(np.rot90(self.plane), cmap=cm,
                            extent=[-self.xwid / 1000, self.xwid / 1000, -self.ywid / 1000,
                                    self.ywid / 1000], vmin=np.min(ticks), vmax=np.max(ticks), aspect='equal')
        ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
        ax.set_ylim( -self.ywid / 1000, self.ywid / 1000)

        cbaxes = inset_axes(ax, width="25%", height="3%", loc=3)
        cbar = plt.colorbar(im, cax=cbaxes, ticks=ticks, orientation='horizontal', cmap=cm)
        cbar.set_label('log('+self.field+')', color='w', labelpad=-11, fontsize=10)
        cbar.ax.xaxis.set_tick_params(color='w')
        cbar.ax.xaxis.set_ticks_position('top')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')

        """
        you have to insert appropriate number for below 'rectangles' 
        this is ruler which indicates the size of projected image
        ex) if you want to draw 5 kpc width projected image and want to insert ruler with 3 kpc size, then 
        replace 5 kpc into 3 kpc and you have to multiply 3/5 instead of 5/14 in width. 
        """

        rectangles = {
            '1 kpc': patches.Rectangle(xy=(0.1 * self.xwid / 1000, -0.95 * self.xwid  / 1000),
                                       width=(2 * self.xwid  / 1000) *1/4,
                                       height=0.01 * (2 * self.ywid  / 1000), facecolor='white')}
        for r in rectangles:
            ax.add_artist(rectangles[r])
            rx, ry = rectangles[r].get_xy()
            cx = rx + rectangles[r].get_width() / 2.0
            cy = ry + rectangles[r].get_height() / 2.0

            ax.annotate(r, (cx, cy + 0.02 * (2 * self.ywid / 1000)), color='w', weight='bold',
                        fontsize=15, ha='center', va='center')


        return im

    def star_plot(self, Part, ax):

        start=time.time()

        print('star plotting...')

        ex_xcenter = self.xcenter*self.mindx+0.5*self.mindx
        ex_ycenter = self.ycenter*self.mindx+0.5*self.mindx
        ex_zcenter = self.zcenter*self.mindx+0.5*self.mindx

        sxplot = (Part.xp[0] - ex_xcenter)/1000
        syplot = (Part.xp[1] - ex_ycenter)/1000
        szplot = (Part.xp[2] - ex_zcenter)/1000
        cax1 = ax.scatter(sxplot, syplot,  c='grey', s=0.1,alpha=0.3)


        print('plotting stars finished , t = %.2f [sec]' %(time.time()-start))
        return cax1

    def star_plot2(self, Part, ax, Fesc_indSED, filter):

        start = time.time()

        print('star plotting...')

        ex_xcenter = self.xcenter * self.mindx + 0.5 * self.mindx
        ex_ycenter = self.ycenter * self.mindx + 0.5 * self.mindx
        ex_zcenter = self.zcenter * self.mindx + 0.5 * self.mindx

        sxplot = (Part.xp[0] - ex_xcenter) / 1000
        syplot = (Part.xp[1] - ex_ycenter) / 1000
        szplot = (Part.xp[2] - ex_zcenter) / 1000


        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

        fwhm = cosmo.kpc_comoving_per_arcmin(0.1)
        print('fwhm',fwhm)

        rflux = Fesc_indSED.rflux
        gflux = Fesc_indSED.gflux
        bflux = Fesc_indSED.bflux

        vmin1, vmax1 = zscale(rflux)
        vmin2, vmax2 = zscale(gflux)
        vmin3, vmax3 = zscale(bflux)

        rflux = rflux / vmax1
        gflux = gflux / vmax2
        bflux = bflux / vmax3

        rflux[rflux>1]=1.0
        gflux[gflux>1]=1.0
        bflux[bflux>1]=1.0

        rflux[rflux<vmin1/vmax1]=0
        gflux[gflux<vmin2/vmax2]=0
        bflux[bflux<vmin3/vmax3]=0





        print(np.min(rflux),np.max(rflux))
        print(np.min(gflux),np.max(gflux))
        print(np.min(bflux),np.max(bflux))




        flux = np.transpose(np.array([rflux*0.7+0.3, gflux*0.7+0.3, bflux*0.7+0.3]))

        flux_t = tuple((map(tuple, flux)))
        print(flux_t)



        cax1 = ax.scatter(sxplot, syplot, c=flux_t, s=0.1, alpha=0.2)

        print('plotting stars finished , t = %.2f [sec]' % (time.time() - start))
        ax.set_xlim(-self.xwid / 1000, self.xwid / 1000)
        ax.set_ylim(-self.ywid / 1000, self.ywid / 1000)

        rectangles = {
            '1 kpc': patches.Rectangle(xy=(0.1 * self.xwid / 1000, -0.95 * self.xwid / 1000),
                                       width=(2 * self.xwid / 1000) * 1/4,
                                       height=0.01 * (2 * self.ywid / 1000), facecolor='white')}
        for r in rectangles:
            ax.add_artist(rectangles[r])
            rx, ry = rectangles[r].get_xy()
            cx = rx + rectangles[r].get_width() / 2.0
            cy = ry + rectangles[r].get_height() / 2.0

            ax.annotate(r, (cx, cy + 0.02 * (2 * self.ywid / 1000)), color='w', weight='bold',
                        fontsize=15, ha='center', va='center')
        return cax1

    def clump_plot(self,Clump,ax):
        #for appropriate description of size of clump, dpi = 144, figsize * size of axis = size
        size=6.3
        # clump finding
        start=time.time()
        print('finding gas clumps...')


        xclumpind = Clump.xclump / self.mindx - self.xcenter + self.xwid
        yclumpind = Clump.yclump / self.mindx - self.ycenter + self.ywid
        zclumpind = Clump.zclump / self.mindx - self.zcenter + self.zwid

        clumpind = np.where(
            (xclumpind >= 0) & (xclumpind < self.xfwd ) & (yclumpind >= 4) & (yclumpind < self.yfwd - 4) & (zclumpind >= 1) & (
                        zclumpind < self.zfwd - 1))[0]

        xclumpind = xclumpind[clumpind]
        yclumpind = yclumpind[clumpind]
        zclumpind = zclumpind[clumpind]

        xclumpplot = (xclumpind - self.xwid) * self.mindx
        yclumpplot = (yclumpind - self.ywid) * self.mindx
        zclumpplot = (zclumpind - self.zwid) * self.mindx

        cax1 = ax.scatter(xclumpplot/1000, yclumpplot/1000 ,edgecolor='k', marker='o', s=(Clump.rclump[clumpind]*144*size/self.mindx/self.xfwd)**2,linewidths=1,facecolors='none')

        return cax1

def zscale(arr):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(arr)

    return vmin, vmax

def CoM_check_plot(Part1, Cell1, wid, depth, xcen,ycen,zcen):
    fig = plt.figure(figsize=(8, 8),dpi=144)

    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    cm1 = plt.get_cmap('rainbow')

    a = GenerateArray2(Part1, Cell1, xcen,ycen,zcen, wid, depth)
    ss1 = a.projectionPlot(Cell1, ax1, cm1, 'xy', 'nH', -3, 2)
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

def plot(ax1, ax2, read, nout, Part, Cell, xwid, ywid, zwid, label,minlv,maxlv,diskmass):

    start = time.time()
    Part1 = Part(read, nout)
    Cell1 = Cell(read, nout, Part1)
    print('reading finished , t = %.2f [sec]' % (time.time() - start))

    xcen, ycen, zcen = CoM_main(Part1, Cell1,diskmass)
    print('finish to find CoM, t= %.2f [sec]'%(time.time()-start))
    a = GenerateArray2( Cell1, read, nout, xwid, ywid, zwid, xcen, ycen, zcen, minlv, maxlv, 'nH','xy')
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

    ss = plot(ax11, ax12,read_new_01Zsun_05pc,380,Part,Cellfromdat,7000,7000,3000,'G9_01Zsun_5pc',6,16,1.75e9)


    ss = plot(ax5, ax6,read_new_01Zsun,480,Part,Cell,7000,7000,3000,'G9_01Zsun',6,15,1.75e9)
    ss = plot(ax7, ax8,read_new_1Zsun_highSN_new,380,Part,Cell,7000,7000,3000,'G9_1Zsun_SNboost',6,15,1.75e9)
    ss = plot(ax1, ax2,read_new_gasrich,250,Part,Cellfromdat,7000,7000,7000,'G9_01Zsun_gasrich',6,15,1.15e10)
    ss = plot(ax9, ax10,read_new_03Zsun_highSN,380,Part,Cell,7000,7000,3000,'G9_03Zsun_SNboost',6,15,1.75e9)
    ss = plot(ax3, ax4,read_new_1Zsun,480,Part,Cell,7000,7000,3000,'G9_1Zsun',6,15,1.75e9)


    #plt.savefig('/Volumes/THYoo/2019_thesis/projection(fig1).png' )
    plt.show()
def single(read,Cell,inisnap, endsnap,label):
    numsnap = endsnap - inisnap + 1

    for i in range(numsnap):
        nout = i + inisnap
        fig = plt.figure(figsize=(9, 9))
        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        if not os.path.isfile(read + '/SAVE/part_%05d.sav' % (nout)):
            print(read + '/SAVE/part_%05d.sav' % (nout))
            continue
        if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
            print(read + '/SAVE/cell_%05d.sav' % (nout))
            continue

        Part1 = Part(read, nout)
        Cell1 = Cell(read, nout, Part1)

        xcen, ycen, zcen = CoM_main(Part1, Cell1, 1.75e9)
        minlv = np.log2(int(Part1.boxpc / np.min(Cell1.dx)))
        maxlv = np.log2(int(Part1.boxpc / np.max(Cell1.dx)))

        a = GenerateArray2(Cell1, read, nout, 7000,7000,3000, xcen, ycen, zcen, 6,15, 'nH', 'xy')
        cm = plt.get_cmap('inferno')
        a.projectionPlot(ax1,cm)

        ax1.text(-a.xwid/1000*0.8, a.ywid/1000*(0.8),'time = %3.1f (Myr)'%Part1.snaptime, color='w')
        ax1.set_xticks([])
        ax1.set_yticks([])

        fig.savefig('/Volumes/THYoo/kisti/plot/proj_movie/proj_%05d.png'%nout)
        plt.close()

def thr_fiducial(read,nout,Cell,savedirec):

    fig = plt.figure(figsize=(24, 9))

    ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
    ax2 = fig.add_axes([0.35, 0.1, 0.3, 0.8])
    ax3 = fig.add_axes([0.65, 0.1, 0.3, 0.8])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    ax3.set_facecolor('black')




    Part1 = Part(read, nout)
    Cell1 = Cell(read, nout, Part1)
    Fesc_indSED1 = Fesc_indSED(read, nout)
    xcen, ycen, zcen = CoM_main(Part1, Cell1, 1.75e9)

    a = GenerateArray2(Cell1, read, nout, 7000, 7000, 4000, xcen, ycen, zcen, 6, 15, 'nH', 'xy',False)
    cm1 = plt.get_cmap('inferno')
    cm2 = plt.get_cmap('viridis_r')
    a.projectionPlot(ax1,cm1,[-3,-2,-1,0,1,2])
    b = GenerateArray2(Cell1, read, nout, 7000, 7000, 4000, xcen, ycen, zcen, 6, 15, 'xHI', 'xy',True)
    b.projectionPlot(ax2,cm2,[-2, -1,0])
    filter = [[7950/4, 10050/4],[17550/4, 22260/4],[24160/4,31270/4]]
    b.star_plot2(Part1,ax3,Fesc_indSED1,filter)
    plt.savefig(savedirec+'thr_fiducial%05d.pdf')

def four_fiducial(read,nout,Cell):

    fig = plt.figure(figsize=(11, 11))

    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
    ax2 = fig.add_axes([0.5, 0.1, 0.4, 0.4])
    ax3 = fig.add_axes([0.1, 0.5, 0.4, 0.4])
    ax4 = fig.add_axes([0.5, 0.5, 0.4, 0.4])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    ax3.set_facecolor('black')

    if not os.path.isfile(read + '/SAVE/part_%05d.sav' % (nout)):
        print(read + '/SAVE/part_%05d.sav' % (nout))
        raise ValueError('no savefile')
    if not os.path.isfile(read + '/SAVE/cell_%05d.sav' % (nout)):
        print(read + '/SAVE/cell_%05d.sav' % (nout))
        raise ValueError('no cellfile')


    Part1 = Part(read, nout)
    Cell1 = Cell(read, nout, Part1)
    #Fesc_indSED1 = Fesc_indSED(read, nout)
    xcen, ycen, zcen = CoM_main(Part1, Cell1, 1.75e9)

    a = GenerateArray2(Cell1, read, nout, 7000, 7000, 3000, xcen, ycen, zcen, 6, 15, 'nH', 'xy',False)
    cm1 = plt.get_cmap('inferno')
    cm2 = plt.get_cmap('viridis_r')
    a.projectionPlot(ax1,cm1,[-3,-2,-1,0,1,2])
    b = GenerateArray2(Cell1, read, nout, 7000, 7000, 3000, xcen, ycen, zcen, 6, 15, 'xHI', 'xy',True)
    b.projectionPlot(ax2,cm2,[-2, -1,0])
    filter = [[7950/4, 10050/4],[17550/4, 22260/4],[24160/4,31270/4]]
    #b.star_plot2(Part1,ax3,Fesc_indSED1,filter)

    c = GenerateArray2(Cell1, read, nout, 7000, 7000, 3000, xcen, ycen, zcen, 6, 15, 'T', 'xy',True)
    cm3 = plt.get_cmap('Greys')
    c.projectionPlot(ax4,cm3,[2,7])

    plt.show()
#main()
#single(read_new_01Zsun,Cell,3,480,'G9_01Zsun')
#four_fiducial('/blackwhale/dbahck37/kisti/',124,Cell)
thr_fiducial(read_new_01Zsun,480,Cell,'/Volumes/THYoo/kisti/plot/2019thesis/')
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
        thr_fiducial(read,nout,Cell,savedirec)




