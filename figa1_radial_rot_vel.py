import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time

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
        self.boxlen = partdata.info.boxlen
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

        dmxp = partdata.part.xp[0]* partdata.info.unit_l / 3.08e18
        dmm = partdata.part.mp[0]

        dmm = dmm * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l
        dmindex = np.where(dmm>2000)
        self.dmxpx = dmxp[0][dmindex]
        self.dmxpy = dmxp[1][dmindex]
        self.dmxpz = dmxp[2][dmindex]

        self.dmm = dmm[dmindex]

        dat = FortranFile(dir+'ray_nside4_80pc/ray_%05d.dat' % (nout), 'r')
        npart, nwave2 = dat.read_ints()
        wave = dat.read_reals(dtype=np.double)
        sed_intr = dat.read_reals(dtype=np.double)
        sed_attH = dat.read_reals(dtype=np.double)
        sed_attD = dat.read_reals(dtype=np.double)
        npixel = dat.read_ints()
        tp = dat.read_reals(dtype='float32')
        fescH = dat.read_reals(dtype='float32')
        self.fescD_80pc = dat.read_reals(dtype='float32')
        photonr = dat.read_reals(dtype=np.double)
        self.fesc_80pc = np.sum(self.fescD_80pc*photonr)/np.sum(photonr)

        dat2 = FortranFile(dir + 'ray_nside4/ray_%05d.dat' % (nout), 'r')
        npart, nwave2 = dat2.read_ints()
        wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attH = dat2.read_reals(dtype=np.double)
        sed_attD = dat2.read_reals(dtype=np.double)
        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * photonr) / np.sum(photonr)

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
        self.m = celldata.cell[0][4][0] *Part.unit_d * Part.unit_l / 1.989e33 * Part.unit_l *Part.unit_l *(celldata.cell.dx[0]*Part.boxlen)**3
        #self.T = celldata.cell[0][4][5]/celldata.cell[0][4][0] * 517534.72 * mu


def Cir_Vel(Part,x,y,z,m,inir):
    G = 6.67408e-11

    zindex = np.where(abs(z - Part.boxpc / 2) < 2000)[0]
    x = x[zindex]
    y = y[zindex]
    xy = np.sqrt((x - Part.boxpc / 2) ** 2 + (y - Part.boxpc / 2) ** 2)
    for i in range(20):
        index = np.where(xy < inir * (0.95 ** i))
        xcenter = np.sum(x[index] * m[index]) / np.sum(m[index])
        ycenter = np.sum(y[index] * m[index]) / np.sum(m[index])
        zcenter = np.sum(z[index] * m[index]) / np.sum(m[index])
        print('xcenter', xcenter-Part.boxpc/2, 'ycenter', ycenter-Part.boxpc/2, 'zcenter', zcenter-Part.boxpc/2)
        xy = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)


    nbin = 40
    binwidth = 50  # pc scale
    binarr = np.linspace(binwidth, nbin * binwidth, nbin)
    rotvel = np.zeros(nbin)

    for i in range(nbin):
        xy = np.sqrt((x-xcenter)**2+(y-ycenter)**2)
        index = np.where(xy<binarr[i])[0]
        rotvel[i] = np.sqrt(G*np.sum(m[index])*1.989e30/binarr[i]/3.08e16)/1e3
        print(np.sum(m[index]))
    return binarr, rotvel

def draw2(read1,nout1,read2,nout2):

    Part1 = Part(read1,nout1)
    print('part')
    Cell1 = Cell(read1,nout1,Part1)
    print('cell')
    print('gas')
    binarr, gas1 = Cir_Vel(Part1, Cell1.x, Cell1.y, Cell1.z, Cell1.m, 5000)
    print('star')
    binarr, star1 = Cir_Vel(Part1, Part1.xp[0], Part1.xp[1], Part1.xp[2], Part1.mp0, 5000)
    print('dm')
    binarr, dm1 = Cir_Vel(Part1, Part1.dmxpx, Part1.dmxpy, Part1.dmxpz, Part1.dmm, 5000)
    Part2 = Part(read2, nout2)
    Cell2 = Cell(read2, nout2,Part2)
    print('gas')
    binarr2, gas2 = Cir_Vel(Part2, Cell2.x, Cell2.y, Cell2.z, Cell2.m, 5000)
    print('star')
    binarr2, star2 = Cir_Vel(Part2, Part2.xp[0], Part2.xp[1], Part2.xp[2], Part2.mp0, 5000)
    print('dm')
    binarr2, dm2 = Cir_Vel(Part2, Part2.dmxpx, Part2.dmxpy, Part2.dmxpz, Part2.dmm, 5000)


    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 25

    fig = plt.figure(figsize=(20,8))
    ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
    ax2 = fig.add_axes([0.5,0.1,0.4,0.8])


    ax1.plot(binarr, gas1, label='gas', marker='o')
    ax1.plot(binarr, star1, label='star',marker='*')
    ax1.plot(binarr,dm1, label='DM',marker='x')

    ax2.plot(binarr, gas2, label='gas',marker='o')
    ax2.plot(binarr, star2, label='star',marker='*')
    ax2.plot(binarr, dm2, label='DM',marker='x')


    ax1.set_title('0.1Zsun, t=%3.2f(Myr)'%Part1.snaptime)
    ax2.set_title('1Zsun, t=%3.2f(Myr)'%Part2.snaptime)

    ax1.set_xlabel('R(pc)')
    ax1.set_ylabel('$V_R$(km/s)')
    ax2.set_xlabel('R(pc)')
    ax2.set_yticks([])
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    plt.show()
    plt.savefig('r_vel/radialvelocity_%05d.png'%nout1)

inisnap = 100
endsnap = 100
numsnap = endsnap - inisnap + 1
for i in range(numsnap):
    nout = inisnap + i
    draw2(read_new_01Zsun_re,nout,read_new_1Zsun_re,nout)


