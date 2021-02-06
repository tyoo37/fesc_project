import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import time
import os.path

read_old_01Zsun = '/Volumes/THYoo/RHD_10pc/'

read_old_002Zsun = '/Volumes/THYoo/RHD_10pc_lowZ/'

read_new_01Zsun = '/Volumes/THYoo/kisti/RHD_10pc_0.1Zsun/'

read_new_1Zsun = '/Volumes/THYoo/kisti/RHD_10pc_1Zsun/'

read_new_01Zsun_se = '/blackwhale/dbahck37/kisti/0.1Zsun/'
read_new_01Zsun_5pc_se = '/blackwhale/dbahck37/kisti/0.1Zsun_5pc/'

read_new_1Zsun_se = '/blackwhale/dbahck37/kisti/1Zsun/'

read_new_1Zsun_highSN_new_se = '/blackwhale/dbahck37/kisti/1Zsun_SNen_new/'

read_new_03Zsun_highSN_se = '/blackwhale/dbahck37/kisti/1Zsun_SNen_new/'

read_new_01Zsun_lyaoff_se = '/blackwhale/dbahck37/kisti/0.1Zsun_lyaoff/'

read_new_gasrich_se = '/blackwhale/dbahck37/kisti/gasrich/'

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
        self.boxlen = partdata.info.boxlen
        self.boxpc = partdata.info.boxlen * partdata.info.unit_l / pc
        xp = partdata.star.xp[0]
        self.xp = xp * partdata.info.unit_l / 3.08e18
        self.unit_l = partdata.info.unit_l
        self.unit_d = partdata.info.unit_d
        self.starage = self.snaptime - partdata.star.tp[0]*4.70430e14/365/3600/24/1e6
        self.starid = np.abs(partdata.star.id[0])
        self.mp0 = partdata.star.mp0[0] * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l

        tp = partdata.star.tp[0] * 4.70430e14 / 365. /24./3600/1e6
        self.tp = tp
        starage = self.snaptime-tp
        sfrindex = np.where((starage >= 0) & (starage < 10))[0]
        self.SFR = np.sum(self.mp0[sfrindex]) / 1e7

        dmxp = partdata.part.xp[0]* partdata.info.unit_l / 3.08e18
        dmm = partdata.part.mp[0]

        dmm = dmm * partdata.info.unit_d * partdata.info.unit_l / 1.989e33 * partdata.info.unit_l*partdata.info.unit_l
        dmindex = np.where(dmm>2000)
        self.dmxpx = dmxp[0][dmindex]
        self.dmxpy = dmxp[1][dmindex]
        self.dmxpz = dmxp[2][dmindex]

        self.dmm = dmm[dmindex]
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
        self.fescD_80pc = dat.read_reals(dtype='float32')
        photonr = dat.read_reals(dtype=np.double)
        self.fesc_80pc = np.sum(self.fescD_80pc*photonr)/np.sum(photonr)
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
class Fesc_old():
    def __init__(self, dir, nout):
        dat2 = FortranFile(dir + 'ray_nside4/ray_%05d.dat' % (nout), 'r')
        npart, nwave2 = dat2.read_ints()
        wave = dat2.read_reals(dtype=np.double)
        sed_intr = dat2.read_reals(dtype=np.double)
        sed_attH = dat2.read_reals(dtype=np.double)
        sed_attD = dat2.read_reals(dtype=np.double)
        npixel = dat2.read_ints()
        tp = dat2.read_reals(dtype='float32')
        self.fescH = dat2.read_reals(dtype='float32')
        self.fescD = dat2.read_reals(dtype='float32')
        self.photonr = dat2.read_reals(dtype=np.double)
        self.fesc = np.sum(self.fescD * self.photonr) / np.sum(self.photonr)
        self.fescwodust = np.sum(self.fescH * self.photonr) / np.sum(self.photonr)


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
def lifetime_in_clump(Part, Clump, Fesc, YoungClumpFind, dir, inisnap, endsnap, trace, preexist, verbose):
    # calculate how long the star paticles are embedded into the clump until the clump is disrupted

    numsnap = endsnap - inisnap + 1
    nosnap = [];
    misssnap = []

    for n in range(numsnap):
        nout = inisnap + n
        if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (nout)):
            print('there is no ' + dir + '/SAVE/part_%05d.sav' % (nout))
            nosnap.append(nout)
            continue
        if not os.path.isfile(dir + '/SAVE/cell_%05d.sav' % (nout)):
            print('there is no ' + dir + '/SAVE/cell_%05d.sav' % (nout))
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

    print('nosnap', nosnap)
    print('misssnap', misssnap)

    lifetimearr = np.array([])
    photonrarr = np.zeros(20)
    fracarr = np.zeros(20)
    count = 1
    totnumcomp = np.array([])
    for m in range(numsnap):
        tuning = 0
        nout = inisnap + m
        print(nout)
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

        # print('inisnap = ',nout)
        # print('the number of star-forming clumps = ', len(Init.youngclumpindex))
        # print('the number of stars younger than 1 Myr = ', len(np.where(InitPart.starage < InitPart.snaptime -PrevPart.snaptime)[0]))
        # print('the number of young stars in clumps = ', int(len(Init.starid)))
        if len(np.where(InitPart.starage < InitPart.snaptime - PrevPart.snaptime)[0]) < int(len(Init.starid)):
            raise ValueError('the number of young stars < the number of young stars in the clump, Error!')
        inistaridarr = Init.starid
        staridarr = inistaridarr
        lifetime = []
        maskarr = np.zeros(len(staridarr))

        # numcomp = np.zeros((len(staridarr),trace))
        # snapintv = np.zeros((len(staridarr),trace))
        # numclump = np.zeros((len(staridarr),trace))
        # numcomp[:,0] = Init.numcomp
        # snapintv[:,0] = Init.starage
        # numclump[:,0]=Init.numclump

        for n in range(trace - 1):

            # Clump1 = YoungClumpFind(Part, Clump, dir, nout)
            traceout = inisnap + n + m + 1

            if not os.path.isfile(dir + '/clump3/clump_%05d.txt' % (traceout)):
                print('there is no file in trace, %05d' % traceout)
                # nocount2 = nocount2 + 1

                if n == trace - 2:
                    num_neg_mask = len(np.where(maskarr == 0)[0])
                    for k in range(num_neg_mask):
                        lifetime = np.append(lifetime, [21])
                    break

                else:
                    break

            if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (traceout)):
                print('there is no file in trace, %05d' % traceout)
                continue
            Part1 = Part(dir, traceout)

            Clump1 = Clump(dir, traceout, Part1)

            for i in range(len(staridarr)):

                if maskarr[i] == -1:
                    continue
                """
                if n == trace-1:
                    lifetime = np.append(lifetime,[21])
                    continue
                """
                starindex = np.where(Part1.starid == int(staridarr[i]))[0]

                if len(starindex) != 1:
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
                if len(clumpindex) == 0:
                    maskarr[i] = -1

                    if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (traceout - 1)):
                        minus = 0.5
                    else:
                        Partprev = Part(dir, traceout - 1)
                        minus = (Part1.snaptime - Partprev.snaptime) / 2

                    lifetime = np.append(lifetime, (Part1.snaptime - initime) + Init.starage[i] - minus)
                else:
                    """
                    compnum_temp = np.array([])
                    for t in range(len(clumpindex)):
                        rr2 = np.sqrt((Clump1.xclump[clumpindex[t]] - Part1.xp[0]) ** 2 + (
                                    Clump1.yclump[clumpindex[t]] - Part1.xp[1]) ** 2 + (
                                              Clump1.zclump[clumpindex[t]] - Part1.xp[2]) ** 2)
                        compindex = np.where((rr2 - Clump1.rclump[clumpindex[t]] < 0)&(Part1.starage<10))[0]
                        compnum_temp = np.append(compnum_temp, compindex)

                    numcomp[i,n+1] = len(np.unique(compnum_temp.astype(int)))-len(clumpindex)
                    if np.where(numcomp[i,n+1]==-1)[0]>0:
                        raise ValueError('damm',nout,traceout)
                    snapintv[i,n+1] = snapintv[i,n+1] + plus
                    if np.where(snapintv[i,n+1]==-1)[0]>0:
                        raise ValueError('wtf',nout,traceout)
                    numclump[i,n+1] = len(clumpindex)
                    """
                    if n == trace - 2:
                        lifetime = np.append(lifetime, (Part1.snaptime - initime) + Init.starage[i] - 0.5)

            if tuning == 1:
                break
        if tuning == 1:
            continue

        if len(staridarr) != len(lifetime):
            print(len(staridarr), len(lifetime))
            print(lifetime)
            continue
            # raise ValueError('error in cal')

        # print('lifetime',lifetime)
        # print('lifetime',lifetime)
        lifetimearr = np.append(lifetimearr, lifetime)

        # photonr

        photonr = Fesc1.photonr

        for j in range(20):
            ind = np.where((InitPart.starage >= j) & (InitPart.starage < j + 1))
            if len(ind[0]) > 0:
                photonrarr[j] = photonrarr[j] + np.sum(photonr[ind]) / np.sum(InitPart.mp0[ind])

        count = count + 1
        # print(photonrarr)
        """
        print('numcomp',numcomp)
        print('snapintv',snapintv)
        print('numclump',numclump)
        avnumcomp = np.zeros(len(staridarr))
        for l in range(len(staridarr)):
            avnumcomp[l] = np.sum(numcomp[l, :] * snapintv[l, :]) / np.sum(snapintv[l, :])
        print('avnumcomp',avnumcomp)

    totnumcomp = np.append(totnumcomp, avnumcomp)
    totnumcomp = np.concatenate(totnumcomp, axis=None)
    print('totnumcomp',totnumcomp)
    """
    photonrarr = photonrarr / count

    for l in range(20):
        fracarr[l] = photonrarr[l] / np.sum(photonrarr)
    if len(lifetimearr) > 0:
        lifetimearr = np.concatenate(lifetimearr, axis=None)

    return lifetimearr, photonrarr, fracarr
def ax_plot(load, ax, Fesc_new,read, inisnap, endsnap, max_trace, preexist, pdf, color, label, ls,verbose):
    if load==True:
        lifetimearr = np.loadtxt(read+'lifetimearr.dat')
        photonrarr = np.loadtxt(read+'photonrarr.dat')
        fracarr = np.loadtxt(read+'fracarr.dat')
        #avnumcomp = np.loadtxt(read+'avnumcomp.dat')
        #print(lifetimearr)

        #print(np.min(lifetimearr), np.max(lifetimearr))
        print(np.mean(lifetimearr))
        #print(fracarr.shape)
        #print(fracarr)
        #print(photonrarr)
    else:

        lifetimearr, photonrarr, fracarr = lifetime_in_clump(Part,Clump,Fesc_new,YoungClumpFind,read,inisnap,endsnap,max_trace,preexist,verbose)
        #print(lifetimearr)
        #print(lifetimearr.shape)
        #print(fracarr.shape)
        #print(fracarr)
        #print(np.min(lifetimearr), np.max(lifetimearr))
        print(np.mean(lifetimearr))




        np.savetxt(read+'lifetimearr.dat',lifetimearr, newline='\n', delimiter=' ')
        np.savetxt(read+'photonrarr.dat',photonrarr, newline='\n', delimiter=' ')
        np.savetxt(read+'fracarr.dat',fracarr, newline='\n', delimiter=' ')
        #np.savetxt(read+'avnumcomp.dat',avnumcomp, newline='\n',delimiter=' ')
    ax.hist(lifetimearr, bins=22, range=(0,21),density=pdf, histtype='stepfilled',edgecolor=color, label=label, facecolor="None",ls=ls)
    #ax2.plot(np.linspace(0,19,num=20)+0.5,np.log10(photonrarr),color=color,ls='dashed')
    #ax4.hist(lifetimearr, bins=22, range=(0,21),density=pdf, histtype='stepfilled',edgecolor=color, label=label, facecolor="None",ls=ls)
    print('plot')

def plot(ax1,ax2,read, color, label,ls,numsnap):
    rclumparr2 = np.loadtxt(read + 'rclumparr2.dat')
    mclumparr2 = np.loadtxt(read + 'mclumparr2.dat')
    print(np.mean(10**rclumparr2))
    print(np.mean(10**mclumparr2))
    ax1.hist(rclumparr2, range=(0.5,2),bins=30,ls=ls, histtype="stepfilled",edgecolor=color, label=label,facecolor="None",weights=np.ones_like(rclumparr2)/float(numsnap))
    ax2.hist(mclumparr2, range=(3,8),bins=30,ls=ls, histtype="stepfilled",edgecolor=color, label=label,facecolor="None",weights=np.ones_like(mclumparr2)/float(numsnap))
    ax1.arrow(np.log10(np.mean(10**rclumparr2)),0,0,0.02, facecolor=color,edgecolor=color,head_width=0.025,head_length=0.01)
    ax2.arrow(np.log10(np.mean(10**mclumparr2)),0,0,0.02, facecolor=color,edgecolor=color,head_width=0.1,head_length=0.01)
    ax1.set_ylim(0,0.25)
    ax2.set_ylim(0,0.25)
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
fig = plt.figure(figsize=(9,9))

ax1 = fig.add_axes([0.15, 0.1, 0.7, 0.35])
ax2 = fig.add_axes([0.15, 0.55, 0.35, 0.35])
ax3 = fig.add_axes([0.5, 0.55, 0.35, 0.35])

ax1.set_xlim(0, 20)
ax1.set_xlabel('trapped timescale (Myr)')
ax1.set_ylabel('PDF')
ax1.set_xticks(np.linspace(0, 20, num=11))
ax_plot(True, ax1,Fesc_new8, read_new_01Zsun,150,480,20,False,True, 'grey','G9_01Zsun','solid',False)
ax_plot(True, ax1,Fesc_new8, read_new_01Zsun_05pc,150,380,20,False,True, 'magenta','G9_01Zsun_5pc','dashed',False)
plot(ax2,ax3,read_new_01Zsun,'k','G9_01Zsun','solid',331)
plot(ax2,ax3,read_new_01Zsun_05pc,'magenta','G9_01Zsun_5pc','dashed',231)
ax2.set_ylabel('PDF',fontsize=25)
ax2.set_xticks([0.5,1,1.5,2])
ax3.set_xticks([3,4,5,6,7,8])
ax2.set_xlabel('log R (pc)')
ax3.set_xlabel('log M ($M_{\odot}$)')
ax3.set_yticks([])
ax1.set_xlim(0,10)
ax1.legend()
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/add1.pdf')
plt.show()



