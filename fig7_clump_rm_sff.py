import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from scipy.optimize import curve_fit
from scipy import stats
import os.path
import matplotlib.ticker
read_old_01Zsun = '/Volumes/THYoo/RHD_10pc/'

read_old_002Zsun = '/Volumes/THYoo/RHD_10pc_lowZ/'

read_new_01Zsun = '/Volumes/THYoo/kisti/RHD_10pc_0.1Zsun/'

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
plt.rcParams['legend.fontsize']=18
plt.rcParams['patch.linewidth']=2


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
        self.boxlen = partdata.info.boxlen

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
        self.vx = celldata.cell[0][4][1] * Part.unit_l / 4.70430e14
        self.vy = celldata.cell[0][4][2] * Part.unit_l / 4.70430e14
        self.vz = celldata.cell[0][4][3] * Part.unit_l / 4.70430e14

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
        self.vx = celldata.cell[0][4][1] * Part.unit_l / 4.70430e14
        self.vy = celldata.cell[0][4][2]* Part.unit_l / 4.70430e14
        self.vz = celldata.cell[0][4][3]* Part.unit_l / 4.70430e14

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



fig = plt.figure(figsize=(10, 9))

ax1 = fig.add_axes([0.15, 0.1, 0.83, 0.23])
ax2 = fig.add_axes([0.15, 0.43, 0.4, 0.53])
ax3 = fig.add_axes([0.58, 0.43, 0.4, 0.53])
#ax4 = fig.add_axes([0.5, 0.4, 0.3, 0.2])
#ax5 = fig.add_axes([0.1, 0.5, 0.3, 0.3])
#ax6 = fig.add_axes([0.5, 0.5, 0.3, 0.3])

class YoungClumpFind():
    def __init__(self,dir,snap, preexist,corr):
        #generating star informaton belongs to clump
        self.dir=dir

        nmemberarr = np.array([])
        staridarr = np.array([])
        staragearr = np.array([])
        self.preexist=[]
        self.youngclumpindex=[]
        prevPart = Part(dir,snap-1)

        Part1 = Part(dir,snap)
        Clump1 = Clump(dir,snap,Part1)
        nclump = len(Clump1.xclump)

        for i in range(nclump):
            rr = np.sqrt((Clump1.xclump[i] - Part1.xp[0]) ** 2 + (Clump1.yclump[i] - Part1.xp[1]) ** 2 + (Clump1.zclump[i] - Part1.xp[2]) ** 2)
            youngindex = np.where((rr < Clump1.rclump[i]) & (Part1.starage < Part1.snaptime - prevPart.snaptime))

            if len(youngindex[0])>0:

                # init correction for early escape of young stars due to preexisting star feedback
                # count only clump which does not have pre-existing stars for initial correction period
                # after that, we don't care whether clump has pre-existing stars or not.
                if preexist==False:
                    if snap <= corr:
                        preexistindex = np.where(
                            (rr < Clump1.rclump[i]) & (Part1.starage >= Part1.snaptime - prevPart.snaptime))
                        if len(preexistindex[0] == 0):
                            starid = Part1.starid[youngindex]
                            staridarr=np.append(staridarr,starid)
                            staragearr=np.append(staragearr,Part1.starage[youngindex])
                            self.youngclumpindex.append(i)


                    else:
                        starid = Part1.starid[youngindex]
                        staridarr = np.append(staridarr, starid)
                        staragearr = np.append(staragearr, Part1.starage[youngindex])

                        self.youngclumpindex.append(i)
                else:
                    preexistindex = np.where(
                        (rr < Clump1.rclump[i]) & (Part1.starage >= Part1.snaptime - prevPart.snaptime))

                    if len(preexistindex[0] == 0):
                        starid = Part1.starid[youngindex]
                        staridarr = np.append(staridarr, starid)
                        staragearr = np.append(staragearr, Part1.starage[youngindex])
                        self.youngclumpindex.append(i)

        self.starid,indices = np.unique(staridarr, return_index=True)
        self.starage = staragearr[indices]
        self.initime = Part1.snaptime
def weighted_median(a,weights):
    index=np.argsort(a)
    sorted_weights=weights[index]/np.sum(weights)

    for i in range(len(sorted_weights)):
        if np.sum(sorted_weights[:i])>0.5:
            index2=i
            break
    return a[index[index2]]
def err(a,weights,lowper,uppper):
    index = np.argsort(a)
    sorted_weights = weights[index] / np.sum(weights)

    for i in range(len(sorted_weights)):
        if np.sum(sorted_weights[:i]) > lowper:
            index2 = i

            break
    for i in range(len(sorted_weights)):
        if np.sum(sorted_weights[:i]) > uppper:
            index3= i
    return a[index[index2]],a[index[index3]]

def plot(ax1,ax2,ax3, read, color, label, x):

    #rclumparr = np.loadtxt(read+'rclumparr.dat')
    #mclumparr = np.loadtxt(read+'mclumparr.dat')
    rclumparr2 = np.loadtxt(read + 'rclumparr2.dat')
    mclumparr2 = np.loadtxt(read + 'mclumparr2.dat')

    sfeavarr = np.loadtxt(read + 'sfeavarr.dat')
    timearr = np.loadtxt(read + 'timearrsfe.dat')
    timestep = np.loadtxt(read+'timestep.dat')
    #print(np.mean(10**rclumparr))
    #print(np.mean(10**mclumparr))
    ax1.set_xlim(0.7,4.5)
    sfeavarr = sfeavarr[~np.isnan(sfeavarr)]

    med=weighted_median(sfeavarr,timestep)

    #med2 = np.mean(sfeavarr)
    low,upp=err(sfeavarr,timestep,0.25,0.75)

    ax1.errorbar(x,med,yerr=np.array([[low,upp]]).T,fmt='o',ecolor=color,elinewidth=2,c=color,capsize=8)
    ax1.scatter(x, med,c=color,s=50,marker='s',facecolor=color)

    #print('minmax',np.min(timestep),np.max(timestep))
    #print(sfeavarr)

    #print(low,upp)
    #print(float(abs(low-med)),float(abs(upp-med)))med =

    print(10**np.median(rclumparr2))
    print(10**np.median(mclumparr2))
    print(med)
    #print(med2)
    print(np.log10(np.mean(10**sfeavarr)))
    ax1.set_yscale('log',nonposy='clip')

    ax2.hist(rclumparr2, range=(0.7,2),bins=30, histtype="stepfilled",edgecolor=color, label=label,facecolor="None",weights=np.ones_like(rclumparr2)/float(len(rclumparr2)))
    ax3.hist(mclumparr2, range=(4,8),bins=30, histtype="stepfilled",edgecolor=color, label=label,facecolor="None",weights=np.ones_like(mclumparr2)/float(len(mclumparr2)))
    #ax1.hist(rclumparr2, range=(0.5, 2), bins=30, normed=True, histtype="stepfilled", edgecolor=color,
    #         facecolor=color,linestyle='dashed',alpha=0.3)
    #ax2.hist(mclumparr2, range=(3, 8), bins=30, normed=True, histtype="stepfilled", edgecolor=color,
    #         facecolor=color,linestyle='dashed',alpha=0.3)
    #ax1.plot([np.log10(np.mean(10**rclumparr)),np.log10(np.mean(10**rclum,parr))], [0,0.3], ls='dashed',c=color)
    #ax2.plot([np.log10(np.mean(10**mclumparr)),np.log10(np.mean(10**mclumparr))], [0,0.3], ls='dashed',c=color)
    #ax2.arrow(np.log10(np.mean(10**rclumparr2)),0,0,0.02, facecolor=color,edgecolor=color,head_width=0.025,head_length=0.01)
    #ax3.arrow(np.log10(np.mean(10**mclumparr2)),0,0,0.02, facecolor=color,edgecolor=color,head_width=0.1,head_length=0.01)
    ax2.arrow(np.median(rclumparr2), 0, 0, 0.02, facecolor=color, edgecolor=color, head_width=0.025,
              head_length=0.01)
    ax3.arrow(np.median(mclumparr2), 0, 0, 0.02, facecolor=color, edgecolor=color, head_width=0.1,
              head_length=0.01)
    ax2.set_ylim(0,0.25)
    ax3.set_ylim(0,0.25)
    ax1.set_ylim(0.0005,0.3)
plot(ax1,ax3,ax2,read_new_01Zsun,'k','G9_Zlow',1)
plot(ax1,ax3,ax2,read_new_1Zsun,'firebrick','G9_Zhigh',2)
plot(ax1,ax3,ax2,read_new_gasrich,'dodgerblue','G9_Zlow_gas5',4)
plot(ax1,ax3,ax2,read_new_1Zsun_highSN_new,'magenta','G9_Zhigh_SN5',3)
#plot(ax1,ax2,read_new_03Zsun_highSN,104,300,'green','G9_03Zsun')

#plot(ax1,ax2,ax3,read_new_01Zsun_05pc,'olive','G9_01Zsun_5pc',4)
ax1.set_xticks([1,2,3,4])
ax1.set_xticklabels(['G9_Zlow', 'G9_Zhigh','G9_Zhigh_SN5','G9_Zlow_gas5'],rotation=0,fontsize=18)
ax2.legend(loc='upper left',frameon=False)
ax3.set_xlabel('$log R_{clump} (pc)$')
ax2.set_xlabel('$log M_{clump} (M_{\odot})$')


ax2.set_ylabel('PDF',fontsize=25)
ax1.set_ylabel(r'$\epsilon_{clump}$',fontsize=30)
ax2.set_xticks([4,5,6,7,8])
ax3.set_xticks([0.5,1.0,1.5,2.0])
ax1.set_ylim(5e-4,0.2)
ax1.set_yticks([1e-3,1e-2,1e-1])
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax1.yaxis.set_minor_locator(locmin)
ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
#ax3.set_ylabel('pdf')
ax3.set_yticks([])
plt.show()
#plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fig7.pdf')