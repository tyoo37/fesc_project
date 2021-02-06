import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
import time
import os.path
from matplotlib.lines import Line2D

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
        sfrindex = np.where((tp >= 0) & (tp < 10))[0]
        self.SFR = np.sum(self.mp0[sfrindex]) / 1e7

        """
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
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
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
        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)

class YoungClumpFind():
    def __init__(self,Part,Clump,dir,snap, preexist, verbose):
        #generating star informaton belongs to clump
        self.dir=dir

        staridarr = []
        staragearr = []
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
                    staridarr.append(starid)
                    staragearr.append(Part1.starage[youngindex])
                    justyoungarr.append(justyoung)

                    self.youngclumpindex.append(i)
            staridarr = np.concatenate(np.array(staridarr))
            staragearr = np.concatenate(np.array(staragearr))
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
                    staridarr.append(starid)
                    staragearr.append(Part1.starage[youngindex])
                    self.youngclumpindex.append(i)
            if len(staridarr)>0 and len(staragearr)>0:
                staridarr = np.concatenate(np.array(staridarr))
                staragearr = np.concatenate(np.array(staragearr))
            if verbose == True:
                print('the # of total clump = %d' % len(Clump1.xclump))
                print('the # of star-forming clump = %d' % len(youngclumpind2))
                print('the # of star-forming clump w/o preexist = %d' % len(self.youngclumpindex))
                print('the # of total young star born at prev snap = %d' % len(np.unique(justyoungarr)))
                print('the # of young star born at prev snap in clumps = %d' % len(np.unique(youngind3)))
                print('the # of young star born at prev snap in clumps w/o preexist= %d' % len(np.unique(staridarr)))



        self.starid,indices = np.unique(staridarr, return_index=True)
        if len(indices)>0:
            self.starage = staragearr[indices]
        else:
            self.starage=[]
        self.initime = Part1.snaptime


def lifetime_in_clump(Part,Clump,Fesc,YoungClumpFind,dir,inisnap,endsnap,trace, preexist,verbose):
    # calculate how long the star paticles are embedded into the clump until the clump is disrupted
    nocount=0
    numsnap = endsnap - inisnap + 1
    lifetimearr = np.array([])
    photonrarr = np.zeros(20)
    fracarr = np.zeros(20)
    count=1
    for m in range(numsnap):
        tuning=0
        nout = inisnap + m
        print(nout)
        if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (nout)):
            print('there is no ' +dir + '/SAVE/part_%05d.sav' % (nout))
            nocount = nocount+1
            continue
        if not os.path.isfile(dir + '/clump3/clump_%05d.txt' % (nout)):
            print('there is no ' +dir + '/SAVE/clump_%05d.txt' % (nout))
            nocount = nocount+1
            continue
        if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (nout - 1 - nocount)):
            print('there is no ' + dir + '/SAVE/part_%05d.sav' % (nout - 1 - nocount))
            nocount = nocount + 1
            continue
        if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (nout - 1 )):
            print('there is no ' + dir + '/SAVE/part_%05d.sav' % (nout - 1 ))
            nocount = nocount + 1
            continue
        if Fesc==Fesc_new:
            if not os.path.isfile(dir + '/ray_nside4_laursen/ray_%05d.dat' % (nout )):
                print('there is no ' + dir + '/ray_nside4_laursen/ray_%05d.dat' % (nout ))
                nocount = nocount + 1
                continue
        if Fesc==Fesc_new8:
            if not os.path.isfile(dir + '/ray_nside8_laursen/ray_%05d.dat' % (nout )):
                print('there is no ' + dir + '/ray_nside8_laursen/ray_%05d.dat' % (nout ))
                nocount = nocount + 1
                continue
        Init = YoungClumpFind(Part,Clump,dir,nout, preexist,verbose)
        InitPart=Part(dir,nout)
        initime = Init.initime
        Fesc1 = Fesc(dir, nout)

        if len(InitPart.starage)!=len(Fesc1.fescD):
            print('mismatching')
            continue

        if len(np.where(np.isnan(Fesc1.photonr)==True)[0])>=1:
            print('nan in photonr')
            continue

        PrevPart=Part(dir,nout-1-nocount)

        #print('inisnap = ',nout)
        #print('the number of star-forming clumps = ', len(Init.youngclumpindex))
        #print('the number of stars younger than 1 Myr = ', len(np.where(InitPart.starage < InitPart.snaptime -PrevPart.snaptime)[0]))
        #print('the number of young stars in clumps = ', int(len(Init.starid)))
        if len(np.where(InitPart.starage < InitPart.snaptime -PrevPart.snaptime)[0]) < int(len(Init.starid)):
            raise ValueError('the number of young stars < the number of young stars in the clump, Error!')
        inistaridarr = Init.starid
        staridarr = inistaridarr
        lifetime = []
        maskarr = np.zeros(len(staridarr))
        nocount2=0
        for n in range(trace):

            #Clump1 = YoungClumpFind(Part, Clump, dir, nout)
            traceout = inisnap + n + m

            if not os.path.isfile(dir + '/clump3/clump_%05d.txt' % (traceout)) or not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (traceout)):

                print('there is no file in trace, %05d'%traceout)
                #nocount2 = nocount2 + 1

                if n==trace-1:
                    num_neg_mask = len(np.where(maskarr==0)[0])
                    for k in range(num_neg_mask):
                        lifetime=np.append(lifetime,[21])
                    break

                else:
                    break


            Part1 = Part(dir, traceout)

            Clump1 = Clump(dir,traceout,Part1)


            for i in range(len(staridarr)):

                if maskarr[i] == -1:
                    continue
                """
                if n == trace-1:
                    lifetime = np.append(lifetime,[21])
                    continue
                """
                starindex = np.where(Part1.starid == int(staridarr[i]))[0]

                if len(starindex)!=1:
                    print(starindex)
                    print(staridarr[i], traceout)
                    print('error in star-id matching')
                    tuning = 1
                    break


                xstar = Part1.xp[0][int(starindex)]
                ystar = Part1.xp[1][int(starindex)]
                zstar = Part1.xp[2][int(starindex)]
                rr = np.sqrt((Clump1.xclump - xstar) ** 2 + (Clump1.yclump - ystar) ** 2 + (
                        Clump1.zclump - zstar) ** 2)
                clumpindex = np.where((rr - Clump1.rclump < 0))[0]
                if len(clumpindex)==0:
                    maskarr[i] =-1

                    if n ==0:
                        lifetime=np.append(lifetime,Init.starage[i]/2)
                    else:
                        if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (traceout-1)):
                            minus = 0.5
                        else:
                            Partprev = Part(dir,traceout-1)
                            minus = (Part1.snaptime - Partprev.snaptime)/2

                        lifetime=np.append(lifetime,(Part1.snaptime-initime) + Init.starage[i] - minus)
                elif n == trace-1:
                    lifetime=np.append(lifetime, n + Init.starage[i] - 0.5)

            if tuning ==1:
                break
        if tuning==1:

            continue


        if len(staridarr)!=len(lifetime):
            print(len(staridarr),len(lifetime))
            print('error in cal')
            break
        #print('lifetime',lifetime)
        lifetimearr=np.append(lifetimearr,lifetime)


        # photonr

        photonr = Fesc1.photonr

        for j in range(20):
            ind = np.where((InitPart.starage>=j)&(InitPart.starage<j+1))
            photonrarr[j] = photonrarr[j] + np.sum(photonr[ind])/len(ind[0])
        count = count+1


    #photonrarr = photonrarr/count

   # for l in range(20):
   #     fracarr[l] = photonrarr[l]/np.sum(photonrarr)
   # if len(lifetimearr)>0:
   #     lifetimearr = np.concatenate(lifetimearr, axis=None)
        return lifetimearr, photonrarr, fracarr


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['patch.linewidth']=2
#plt.style.use('dark_background')

def ax_plot(load, ax, ax2,ax3, Fesc_new,read, inisnap, endsnap, max_trace, preexist, pdf, color, label, ls,verbose):
    if load==True:
        lifetimearr = np.loadtxt(read+'lifetimearr.dat')
        photonrarr = np.loadtxt(read+'photonrarr.dat')
        fracarr = np.loadtxt(read+'fracarr.dat')

        #print(lifetimearr)

        print(np.min(lifetimearr), np.max(lifetimearr))
        print(np.mean(lifetimearr))
        print(fracarr.shape)
        print(fracarr)
        print(photonrarr)
    else:

        lifetimearr, photonrarr, fracarr = lifetime_in_clump(Part,Clump,Fesc_new,YoungClumpFind,read,inisnap,endsnap,max_trace,preexist,verbose)
        print(lifetimearr)
        print(lifetimearr.shape)
        print(fracarr.shape)
        print(np.min(lifetimearr), np.max(lifetimearr))
        print(np.mean(lifetimearr))




        #np.savetxt(read+'lifetimearr.dat',lifetimearr, newline='\n', delimiter=' ')
        #np.savetxt(read+'photonrarr.dat',photonrarr, newline='\n', delimiter=' ')
        #np.savetxt(read+'fracarr.dat',fracarr, newline='\n', delimiter=' ')
    ax.hist(lifetimearr, bins=22, range=(0,21),density=pdf, histtype='stepfilled',edgecolor=color, label=label, facecolor="None",ls=ls)
    ax2.plot(np.linspace(0,19,num=20)+0.5,np.log10(photonrarr),color=color,ls='dashed')
    ax3.plot(np.linspace(0,19,num=20)+0.5,fracarr,color=color,ls='solid',marker='s')
    ax3.set_ylim(0,0.3)
    print('plot')

"""
lifetimearr1 = lifetime_in_clump(Part,Clump,YoungClumpFind,read_new_01Zsun,118,230,20,5)
np.savetxt('lifetimearr_01Zsun.dat', lifetimearr1, newline='\n', delimiter=' ')
ax1.hist(lifetimearr1, bins=22, range=(0,21))

lifetimearr2 = lifetime_in_clump(Part,Clump,YoungClumpFind,read_new_1Zsun,118,230,20,5)
np.savetxt('lifetimearr_1Zsun.dat', lifetimearr2, newline='\n', delimiter=' ')
ax2.hist(lifetimearr2, bins=22, range=(0,21))

#lifetimearr3 = lifetime_in_clump(Part,Clump,YoungClumpFind,read_new_gasrich,110,200,20)
#np.savetxt('lifetimearr_1Zsun.dat', lifetimearr3, newline='\n', delimiter=' ')
#ax3.hist(lifetimearr3, bins=21, range=(0,20))

lifetimearr4 = lifetime_in_clump(Part,Clump,YoungClumpFind,read_new_1Zsun_highSN,118,230,20,5)
np.savetxt('lifetimearr_1Zsun_highSN.dat', lifetimearr4, newline='\n', delimiter=' ')
ax4.hist(lifetimearr4, bins=21, range=(0,21))
"""
#plot5, lifetimearr5 = ax_plot(True, ax1,read_new_01Zsun_05pc,74,226,20,5,'lifetimearr_01Zsun_5pc.dat',True,'purple','G9_01Zsun_5pc','dashed')

#ax_plot(True, ax1,read_new_gasrich,100,250,20,5,'lifetimearr_gasrich.dat',True,'b','gas-rich','solid')
#plot6, lifetimearr6 = ax_plot(True, ax1,read_new_03Zsun_highSN,25,280,20,5,'lifetimearr_03Zsun_SNen.dat',True,'g','G9_03Zsun_SNboost','solid')
#ax_plot(True, ax1,read_new_01Zsun,118,300,20,5,'lifetimearr_01Zsun.dat',True, 'olive','ref','solid')
#plot2, lifetimearr2 = ax_plot(True, ax1,read_new_1Zsun,118,210,20,5,'lifetimearr_1Zsun.dat',True,'r','G9_1Zsun','solid')
#ax_plot(True, ax1,read_new_1Zsun_highSN_new,149,250,20,5,'lifetimearr_1Zsun_SNen.dat',True,'orange','metal-rich','solid')

for n in range(100):
    fig = plt.figure(figsize=(11, 12))

    ax1 = fig.add_axes([0.15, 0.1, 0.7, 0.35])
    ax2 = fig.add_axes([0.15, 0.55, 0.7, 0.35])
    #ax3 = fig.add_axes([0.7, 0.1, 0.2, 0.3])
    #ax4 = fig.add_axes([0.7, 0.1, 0.2, 0.8])
    ax1.set_xlim(0, 20)
    ax1.set_xlabel('trapped timescale (Myr)')
    # ax2.set_xlabel('time(Myr)')
    # ax3.set_xlabel('time(Myr)')
    # ax4.set_xlabel('time(Myr)')
    ax1.set_ylabel('PDF')
    ax1.set_xticks(np.linspace(0, 20, num=11))
    ax2.set_xlabel('age (Myr)')
    ax2.set_ylabel('$\dot N(t)/\sum \dot N(t)$')
    ax3 = ax2.twinx()
    ax3.set_ylabel(r'$log(\dot N(t))$')
    ax3.set_xticks([])
    ##########################
    nout = 160+n

    if not os.path.isfile(read_new_01Zsun + '/SAVE/part_%05d.sav' % (nout)):
        print('there is no ' + read_new_01Zsun + '/SAVE/part_%05d.sav' % (nout))
        continue
    if not os.path.isfile(read_new_01Zsun + '/clump3/clump_%05d.txt' % (nout)):
        print('there is no ' + read_new_01Zsun + '/SAVE/clump_%05d.txt' % (nout))
        continue
    if not os.path.isfile(read_new_01Zsun + '/SAVE/part_%05d.sav' % (nout - 1 - nocount)):
        print('there is no ' + read_new_01Zsun + '/SAVE/part_%05d.sav' % (nout - 1 - nocount))
        continue
    if not os.path.isfile(read_new_01Zsun + '/SAVE/part_%05d.sav' % (nout - 1)):
        print('there is no ' + read_new_01Zsun + '/SAVE/part_%05d.sav' % (nout - 1))
        continue
    if not os.path.isfile(read_new_01Zsun + '/ray_nside8_laursen/ray_%05d.dat' % (nout)):
        print('there is no ' + read_new_01Zsun + '/ray_nside8_laursen/ray_%05d.dat' % (nout))
        continue

    ax_plot(False, ax1,ax3,ax2,Fesc_new8, read_new_01Zsun,164+n,164+n,20,False,True, 'grey','G9_01Zsun','solid',False)
    ax_plot(False, ax1,ax3,ax2,Fesc_new, read_new_gasrich,164+n,164+n,20,False,True, 'b','G9_01Zsun_gasrich','solid',False)
    plt.savefig('/Volumes/THYoo/kisti/plot/clump_lifetime/test1/fig_%05d.png'%(n+4))
    plt.close()

ax_plot(False, ax1,ax3,ax2,Fesc_new8,read_new_03Zsun_highSN,195,371,20,False,True,'orange','G9_03Zsun_SNboost','solid',False)
ax_plot(False, ax1,ax3,ax2,Fesc_new8, read_new_1Zsun_highSN_new,190,374,20,False,True,'red','G9_1Zsun_SNboost','solid',False)

#plot6, lifetimearr6 = ax_plot(True, ax1,read_new_03Zsun_highSN,25,280,20,5,'lifetimearr_03Zsun_SNen.dat',True,'g','G9_03Zsun_SNboost','solid')

#plot2, lifetimearr2 = ax_plot(True, ax1,read_new_1Zsun,118,210,20,5,'lifetimearr_1Zsun.dat',True,'r','G9_1Zsun','solid')



legend_elements1 = [Line2D([0], [0], color='k',  label=r'$log(\dot N(t))$', ls='dashed'),
                   Line2D([0], [0], color='k', label='$\dot N(t)/\sum \dot N(t)$',
                        )]
#ax2.set_ylabel('pdf')
#ax3.set_ylabel('N')
#ax4.set_ylabel('pdf')
#ax2.set_title('metal_rich')
#ax4.set_title('gas_rich')
ax1.set_xlim(0,20)
#ax2.set_xlim(0,20)
#ax4.set_xlim(0,20)
ax1.legend()
ax2.legend(handles=legend_elements1)

plt.show()

