import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import os.path

"""
estimate sff based on the timescale b/t previous snapshot and current snapshot
"""

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

class readSED():
    def __init__(self, dir, nout):
        dat = FortranFile(dir+'/ray_%05d.dat' % (nout), 'r')
        npart, nwave2 = dat.read_ints()
        wave = dat.read_reals(dtype=np.double)
        sed_intr = dat.read_reals(dtype=np.double)
        sed_attH = dat.read_reals(dtype=np.double)
        sed_attD = dat.read_reals(dtype=np.double)

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
        self.starid = np.abs(partdata.star.id[0])
        self.mp0 = partdata.star.mp0[0] * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l
        self.starage = (partdata.info.time-partdata.star.tp[0]) * 4.70430e14 / 365. /24./3600/1e6
        self.boxlen = partdata.info.boxlen
        youngindex = np.where(self.starage<10)
        self.SFR = np.sum(self.mp0[youngindex])/1e7

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

def GetSFE(Part1, Clump1, prevsnap):

    youngstar =[]
    clumpindex = []
    sfeindarr = []
    tffarr=[]
    for i in range(len(Clump1.xclump)):

        dd = np.sqrt((Part1.xp[0]-Clump1.xclump[i])**2+(Part1.xp[1]-Clump1.yclump[i])**2+(Part1.xp[2]-Clump1.zclump[i])**2)

        index = np.where((dd-Clump1.rclump[i]<0)&(Part1.starage<Part1.snaptime-prevsnap))[0]
        if len(index)>0:
            clumpindex.append(i)
            youngstar.append(np.sum(Part1.mp0[index]))
            sfeindarr.append(np.sum(Part1.mp0[index])/(Clump1.massclump[i]+np.sum(Part1.mp0[index])))
            rho = Clump1.massclump[i] / (4 / 3 * np.pi * Clump1.rclump ** 3) * 1.989e33 / (3.08e18 ** 3)
            t_ff = np.sqrt(3 * np.pi / 32 / 6.67e-8 / rho)
            tffarr.append(t_ff/365/3600/24/1e6)
        #indexarr = np.append(indexarr, index.astype(int))


        if len(np.where(np.isnan(np.array(youngstar))==True)[0])>0:
            print('youngstar',youngstar)
        if len(np.where(np.isnan(Clump1.massclump[np.array(clumpindex).astype(int)])==True)[0])>0:
            print('clumpindex',clumpindex)
            print('massclump',Clump1.massclump)
    #noindexarr = np.delete(np.arange(len(Part1.xp[0])),indexarr)
    if np.isnan(np.sum(np.array(youngstar))/(np.sum(Clump1.massclump[np.array(clumpindex).astype(int)])+np.sum(np.array(youngstar))))==True:
        print(clumpindex, np.array(youngstar),Clump1.massclump[np.array(clumpindex).astype(int)])
    return np.sum(np.array(youngstar))/(np.sum(Clump1.massclump[np.array(clumpindex).astype(int)])+np.sum(np.array(youngstar))), np.array(sfeindarr),np.array(tffarr)




class YoungClumpFind():
    def __init__(self,Part,Clump,dir,snap, preexist, verbose):
        #generating star informaton belongs to clump
        self.dir=dir

        staridarr = np.array([])
        staragearr = np.array([])
        numyoung = []
        self.preexist=[]
        self.youngclumpindex=[]
        youngclumpind2 = []
        youngind3=np.array([])
        justyoungarr = []
        prevPart = Part(dir,snap-1)

        Part1 = Part(dir,snap)
        Clump1 = Clump(dir,snap,Part1)
        nclump = len(Clump1.xclump)


        if preexist == False:

            for i in range(nclump):
                rr = np.sqrt((Clump1.xclump[i] - Part1.xp[0]) ** 2 + (Clump1.yclump[i] - Part1.xp[1]) ** 2 + (Clump1.zclump[i] - Part1.xp[2]) ** 2)
                youngindex = np.where((rr < Clump1.rclump[i]) & (Part1.starage < Part1.snaptime - prevPart.snaptime))
                justyoung = np.where((Part1.starage < Part1.snaptime - prevPart.snaptime))[0]
                if len(youngindex[0])>0:

                    # init correction for early escape of young stars due to preexisting star feedback
                    # count only clump which does not have pre-existing stars for initial correction period
                    # after that, we don't care whether clump has pre-existing stars or not.

                    starid = Part1.starid[youngindex]
                    staridarr=np.append(staridarr,starid)
                    staragearr=np.append(staragearr,Part1.starage[youngindex])
                    justyoungarr.append(justyoung)
                    numyoung.append(len(youngindex[0]))
                    self.youngclumpindex.append(i)
            """       
            if len(staridarr)>0 and len(staragearr)>0:
                staridarr = np.concatenate(np.array(staridarr),axis=None)
                staragearr = np.concatenate(np.array(staragearr),axis=None)
            """
            if verbose ==True:
                print('the # of total clump = %d'%len(Clump1.xclump))
                print('the # of star-forming clump = %d'%len(self.youngclumpindex))
                print('the # of total young star born at prev snap = %d'%len(justyoungarr))
                print('the # of young star born at prev snap in clumps = %d'%len(staridarr))


        else:

            for i in range(nclump):
                rr = np.sqrt((Clump1.xclump[i] - Part1.xp[0]) ** 2 + (Clump1.yclump[i] - Part1.xp[1]) ** 2 + (
                            Clump1.zclump[i] - Part1.xp[2]) ** 2)
                youngindex = np.where((rr < Clump1.rclump[i]) & (Part1.starage < Part1.snaptime - prevPart.snaptime))
                if len(youngindex[0])>0:
                    youngclumpind2.append(i)

                preexistindex = np.where(
                    (rr < Clump1.rclump[i]) & (Part1.starage >= Part1.snaptime - prevPart.snaptime))
                justyoung = np.where((Part1.starage < Part1.snaptime - prevPart.snaptime))[0]
                justyoungarr=np.append(justyoungarr,justyoung)

                youngind3=np.append(youngind3,youngindex[0])

                if len(youngindex[0]) > 0 and len(preexistindex[0] == 0):
                    starid = Part1.starid[youngindex]
                    staridarr=np.append(staridarr,starid)
                    staragearr=np.append(staragearr,Part1.starage[youngindex])
                    self.youngclumpindex.append(i)
            if len(staridarr)>0 and len(staragearr)>0:
                staridarr = np.concatenate(np.array(staridarr),axis=None)
                staragearr = np.concatenate(np.array(staragearr),axis=None)
            if verbose == True:
                print('the # of total clump = %d' % len(Clump1.xclump))
                print('the # of star-forming clump = %d' % len(youngclumpind2))
                print('the # of star-forming clump w/o preexist = %d' % len(self.youngclumpindex))
                print('the # of total young star born at prev snap = %d' % len(np.unique(justyoungarr)))
                print('the # of young star born at prev snap in clumps = %d' % len(np.unique(youngind3)))
                print('the # of young star born at prev snap in clumps w/o preexist= %d' % len(np.unique(staridarr)))



        self.starid,indices = np.unique(staridarr, return_index=True)
        self.numcomp = np.zeros(len(self.starid))
        self.numclump = np.zeros(len(self.starid))
        """
        for j in range(len(self.starid)):
            tempnum=0
            index = np.where(self.starid[j]==Part1.starid)
            rr2 = np.sqrt((Clump1.xclump - Part1.xp[0][index]) ** 2 + (Clump1.yclump - Part1.xp[1][index]) ** 2 + (
                    Clump1.zclump - Part1.xp[2][index]) ** 2)
            index2 = np.where(rr2 - Clump1.rclump < 0)
            for k in range(len(index2[0])):
                rr3 = np.sqrt((Clump1.xclump[index2[0][k]] - Part1.xp[0]) ** 2 + (Clump1.yclump[index2[0][k]] - Part1.xp[1]) ** 2 + (
                    Clump1.zclump[index2[0][k]] - Part1.xp[2]) ** 2)
                index3 = np.where((rr3 - Clump1.rclump[index2[0][k]]< 0)&(Part1.starage<Part1.snaptime-prevPart.snaptime))
                tempnum = tempnum + len(index3[0])-1
            self.numclump[j] = len(index2[0])
            self.numcomp[j] = tempnum - 1
        self.numcomp = self.numcomp
        """
        if len(indices)>0:
            self.starage = staragearr[indices]
        else:
            self.starage=[]
        self.initime = Part1.snaptime


def GetSFE2(Part, Clump, Fesc, dir, inisnap, endsnap, trace, preexist, verbose ):
    numsnap = endsnap - inisnap + 1
    nosnap = [];
    misssnap = []

    for n in range(numsnap):
        nout = inisnap + n
        if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (nout)):
            print('there is no ' + dir + '/SAVE/part_%05d.sav' % (nout))
            nosnap.append(nout)
            continue

        if not os.path.isfile(dir + '/clump3/clump_%05d.txt' % (nout)):
            print('there is no ' + dir + '/clump3/clump3_%05d.txt' % (nout))
            nosnap.append(nout)
            continue
        if Fesc == Fesc_new:
            if not os.path.isfile(dir + '/ray_nside4_laursen/ray_%05d.dat' % (nout)):
                print('there is no ' + dir + '/ray_nside4_laursen/ray_%05d.dat' % (nout))
                nosnap.append(nout)
                continue
        if Fesc == Fesc_new8:
            if not os.path.isfile(dir + '/ray_nside8_laursen/ray_%05d.dat' % (nout)):
                print('there is no ' + dir + '/ray_nside8_laursen/ray_%05d.dat' % (nout))
                nosnap.append(nout)
                continue

        InitPart = Part(dir, nout)
        Fesc1 = Fesc(dir, nout)

        if len(InitPart.starage) != len(Fesc1.fescD):
            print('mismatching')
            misssnap.append(nout)
            continue
        if len(np.where(np.isnan(Fesc1.photonr) == True)[0]) >= 1:
            print('nan in photonr')
            nosnap.append(nout)
            continue

        if np.min(InitPart.starage) < 0:
            print('negative star age')
            nosnap.append(nout)
            continue

    print('nosnap', nosnap)
    print('misssnap', misssnap)

    lifetimearr = np.array([])
    photonrarr = np.zeros(trace)
    fracarr = np.zeros(trace)
    count = 1
    totnumcomp = np.array([])
    for m in range(numsnap):
        tuning = 0
        nout = inisnap + m
        print(nout, dir)
        exit = False

        for j in range(len(misssnap)):
            if nout - misssnap[j] >= -20 and nout - misssnap[j] < 0:
                exit = True
                print('mismatching')
                break

        if nout in nosnap:
            print('nosnap')
            continue
        if nout - 1 in nosnap:
            print('nosnap in prev')
            continue

        if exit == True:
            continue

        Init = YoungClumpFind(Part, Clump, dir, nout, preexist, verbose)
        InitPart = Part(dir, nout)
        initime = Init.initime
        Fesc1 = Fesc(dir, nout)

        PrevPart = Part(dir, nout - 1)

        if len(np.where(InitPart.starage < InitPart.snaptime - PrevPart.snaptime)[0]) < int(len(Init.starid)):
            raise ValueError('the number of young stars < the number of young stars in the clump, Error!')
        inistaridarr = Init.starid
        staridarr = inistaridarr
        lifetime = np.array([])
        maskarr = np.zeros(len(staridarr))

        exit = False




        for n in range(trace):  # loop starts from n=1 to 20 (trace)

            if n == trace - 1:
                num_neg_mask = len(np.where(maskarr == 0)[0])
                if num_neg_mask == 0:
                    exit = True
            if exit == True:
                break


def sfe(ax,ax2,read, inisnap, endsnap, color, label, loadsave,text):
    if loadsave == False:

        numsnap = endsnap - inisnap + 1
        timearr=np.array([])
        sfeavarr=np.array([])
        sfearr=np.array([])
        timestep=np.array([])
        tffarr = np.array([])
        for i in range(numsnap):
            nout = i + inisnap
            #print(nout)
            if not os.path.isfile(read + '/SAVE/part_%05d.sav' % (nout)):
                print(read + '/SAVE/part_%05d.sav' % (nout))
                continue

            if not os.path.isfile(read + '/clump3/clump_%05d.txt' % (nout)):
                print(read + '/clump3/clump_%05d.txt' % (nout))
                continue


            Part1 = Part(read, nout)

            if not os.path.isfile(read + '/SAVE/part_%05d.sav' % (nout - 1)):
                # print(read + '/SAVE/cell_%05d.sav' % (nout-1))
                prevsnap = Part1.snaptime-1
            else:
                prevPart = Part(read, nout - 1)
                prevsnap = prevPart.snaptime
                if Part1.snaptime - prevsnap >2:
                    prevsnap=Part1.snaptime-1


            if i>0:
                if prevsnap > Part1.snaptime:
                    print('skip', nout,Part1.snaptime)
                    continue
            #print(Part1.snaptime)
            Clump1 = Clump(read, nout,Part1)
            sfeav, sfearr1,tffarr1 = GetSFE(Part1,Clump1,prevsnap)
            #print(sfeav)
            sfearr=np.append(sfearr,sfearr1)
            sfeavarr = np.append(sfeavarr, sfeav)
            timearr = np.append(timearr, Part1.snaptime)
            timestep = np.append(timestep, Part1.snaptime-prevsnap)
            tffarr = np.append(tffarr, tffarr1)

        sfearr=np.concatenate(sfearr,axis=None)

        np.savetxt(read+'sfearr.dat',sfearr, newline='\n',delimiter=' ')
        np.savetxt(read+'timearrsfe.dat',timearr,newline='\n', delimiter=' ')
        np.savetxt(read+'sfeavarr.dat',sfeavarr,newline='\n', delimiter=' ')
        np.savetxt(read+'timestep.dat',timestep,newline='\n',delimiter=' ')
        np.savetxt(read+'tffarr.dat',tffarr,newline='\n',delimiter=' ')
    else:
        timearr = np.loadtxt(read+'timearrsfe.dat')
        sfeavarr = np.loadtxt(read + 'sfeavarr.dat')
        sfearr = np.loadtxt(read+'sfearr.dat')
        tffarr = np.loadtxt(read+'tffarr.dat')
    print('tffarr',np.mean(tffarr))
    ax.plot(timearr, sfeavarr, color=color)
    ax2.hist(np.log10(sfearr),range=(-5,0),bins=50,histtype="stepfilled",edgecolor=color, label=label,facecolor="None",weights=np.ones_like(sfearr)/float(len(sfearr)) )

    ax.set_xlim(100,480)

    ax.set_ylim(0.0001,0.1)
    ax.set_yscale('log')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] =16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

fig = plt.figure(figsize=(9, 9), dpi=144)

ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.25])
ax2 = fig.add_axes([0.1, 0.45, 0.8, 0.5])

sfe(ax1,ax2,read_new_01Zsun,150,300,'gray','G9_01Zsun',False,False)

#sfe(ax2,read_new_01Zsun,100,480,'grey','G9_01Zsun',True,False)
#sfe(ax3,read_new_01Zsun,100,480,'grey','G9_01Zsun',True,False)
#sfe(ax4,read_new_01Zsun,100,480,'grey','G9_01Zsun',True,False)
#sfe(ax5,read_new_01Zsun,100,480,'grey','G9_01Zsun',True,False)
#sfe(ax6,read_new_01Zsun,100,480,'grey','G9_01Zsun',True,False)

sfe(ax1,ax2,read_new_1Zsun,150,300,'lightseagreen','G9_1Zsun',False,False)

sfe(ax1,ax2,read_new_1Zsun_highSN_new,70,220,'r','G9_1Zsun_SNboost',False,False)
sfe(ax1,ax2,read_new_gasrich,150,300,'b','G9_01Zsun_gasrich',False,False)
#sfe(ax1,ax2,read_new_03Zsun_highSN,70,380,'orange','G9_03Zsun_SNboost',True,False)
#sfe(ax1,ax2,read_new_01Zsun_05pc,70,380,'cyan','G9_01Zsun_5pc',True,False)

#fig.text(0.43,0.04,'time (Myr)',fontsize=20)

ax1.set_xlabel('time (Myr)')
ax2.set_xlabel('$log (\epsilon)$')
ax1.set_ylabel(r'$\langle \epsilon \rangle$')
ax2.set_ylabel('PDF')
ax2.set_yscale('log')
ax2.legend()
plt.show()

#ax1.set_yscale('log')









