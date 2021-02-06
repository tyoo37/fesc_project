import numpy as np
from scipy.io import FortranFile
from scipy.io import readsav
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import time
import os.path
import matplotlib.ticker
import matplotlib as mpl
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

def fescplot(ax, ax2, Part, Fesc, dir, inisnap, endsnap,  label,  load, color, color2, text, legend):

    if load==False:
        numsnap = endsnap - inisnap + 1
        fescarray = np.array([])
        fescarray2 = np.array([])
        cumfescarr = np.array([])
        SFRarray = np.array([])
        timearray = np.array([])
        #SFRarray = np.array([])
        photonrarr=np.array([])
        prev=0
        for i in range(numsnap):
            nout = inisnap + i
            if not os.path.isfile(dir + '/SAVE/part_%05d.sav' % (nout)):
                print(dir + '/SAVE/part_%05d.sav' % (nout))
                continue
            if Fesc==Fesc_old:
                if not os.path.isfile(dir + '/ray_nside4/ray_%05d.dat' % (nout)):
                    print(dir + '/ray_nside4/ray_%05d.dat'  % (nout))
                    continue
            if Fesc==Fesc_new:
                if not os.path.isfile(dir + '/ray_nside4_laursen/ray_%05d.dat' % (nout)):
                    print(dir + '/ray_nside4_laursen/ray_%05d.dat'  % (nout))
                    continue
            if Fesc==Fesc_new8:
                if not os.path.isfile(dir + '/ray_nside8_laursen/ray_%05d.dat' % (nout)):
                    print(dir + '/ray_nside8_laursen/ray_%05d.dat'  % (nout))
                    continue

            if dir==read_new_1Zsun:
                if nout==26 or nout==27 or nout==28 or nout==29:
                    continue



            Part1 = Part(dir, nout)

            if i >1:
                if prev > Part1.snaptime:
                    print('skip error info')
                    continue
            Fesc1 = Fesc(dir, nout)
            if np.isnan(Fesc1.fesc)==True:
                continue
            if Fesc1.fesc==0:
                continue


            fescarray=np.append(fescarray, Fesc1.fesc)
            fescarray2=np.append(fescarray2, Fesc1.fesc2)
            photonrarr = np.append(photonrarr,np.sum(Fesc1.photonr))

            SFRarray=np.append(SFRarray,Part1.SFR)
            cumfescarr=np.append(cumfescarr, np.mean(fescarray))

            timearray=np.append(timearray, Part1.snaptime)

            prev = Part1.snaptime
            print(nout, Part1.snaptime,Fesc1.fesc)

            #print(nout, Part1.snaptime,Fesc1.fesc)

        """
        for time in timearray:
            age = time - Part1.tp
            young = np.where((age>=0) & (age <10))
            SFRarray= np.append(SFRarray,np.sum(Part1.mp0[young])/1e7)
        
        """
        np.savetxt(dir+'timearr.dat',timearray,delimiter=' ', newline='\n')
        np.savetxt(dir+'SFRarr.dat',SFRarray,delimiter=' ', newline='\n')
        np.savetxt(dir+'fescarr.dat',fescarray,delimiter=' ', newline='\n')
        np.savetxt(dir+'fescarr2.dat',fescarray2,delimiter=' ', newline='\n')
        np.savetxt(dir+'photonrarr.dat',photonrarr,delimiter=' ',newline='\n')

    else:
        timearray = np.loadtxt(dir+'timearr.dat')
        SFRarray = np.loadtxt(dir+'SFRarr.dat')
        fescarray = np.loadtxt(dir+'fescarr.dat')
        fescarray2 = np.loadtxt(dir+'fescarr2.dat')
        photonrarr = np.loadtxt(dir+'photonrarr.dat')

    index1 = np.where(timearray<=150)[0]
    index2 = np.where(timearray>=150)[0]
    index1 = np.append(index1, index2[0])

    fesc_lw = np.sum(fescarray[index2]*photonrarr[index2])/np.sum(photonrarr[index2])
    fesc_lw_d = np.sum(fescarray2[index2]*photonrarr[index2])/np.sum(photonrarr[index2])

        #print(np.mean(10**fescarray[index]))


    #pp2=ax.plot(timearray, cumfescarr, color='grey', ls='dotted',lw=1, label=r'$\langle f_{esc}(t) \rangle$')
    pp2=ax2.plot(timearray[index2], SFRarray[index2], color=color2, ls='-', lw=3, label='SFR')
    ax2.plot(timearray[index1], SFRarray[index1], color=color2, ls='-', lw=1)
    ax2.plot([0,500],np.ones(2)*np.mean(SFRarray[index2]),color=color2,ls='dashed',lw=1.5)

    pp1=ax.plot(timearray[index2], fescarray[index2], color=color, ls='-',lw=3, label=r'$\langle f_{esc} \rangle$')
    ax.plot(timearray[index1], fescarray[index1], color=color, ls='-',lw=1)
    #ax.plot([0,500],np.ones(2)*fesc_lw,color=color2,ls='dotted',lw=1.4)

    ax.plot([0,500],np.ones(2)*fesc_lw,color=color,ls='dashed',lw=1.5)
    pp3=ax.plot(timearray[index2], fescarray2[index2], color='brown', ls='dotted',lw=3,label=r'$\langle f_{esc}\rangle$ w/o dust')
    ax.plot(timearray[index1], fescarray2[index1], color='brown', ls='dotted', lw=1)
    #ax2.plot(timearray, np.log10(SFRarray), color=color, label=label, ls=ls)
    #ax.text(np.min(timearray),0.1, label)
    print(dir)
    print('fesc /w dust = ', np.mean(fescarray))
    print('fesc /wo dust = ',np.mean(fescarray2))
    print('fesc /w dust (t>150 Myr) = ',np.mean(fescarray[index2]))
    print('fesc /wo dust (t>150 Myr)= ',np.mean(fescarray2[index2]))
    print('lum weighted fesc /w dust (t>150 Myr) = ',fesc_lw)
    print('lum weighted fesc /wo dust (t>150 Myr) = ',fesc_lw_d)

    print('SFR = ',np.mean(SFRarray))
    print('SFR (t>150 Myr)= ',np.mean(SFRarray[index2]))
    print('')
    if text ==True:
        ax.text(60, 0.88, label, fontsize=15)

    pp=pp1+pp2+pp3

    labs = [l.get_label() for l in pp]

    if legend==True:
       ax.legend(pp,labs,loc=1,frameon=False,ncol=2)

    #ax.set_ylabel('log($f_{es c}$)')
    #ax2.set_ylabel('$dM_*/dt (M_\odot/yr)$')
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_ylim(0.0001,5)
    ax2.set_ylim(0.008,30)
    ax.set_xlim(0,500)
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax2.yaxis.set_major_locator(locmaj)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    locmin = mpl.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    ax2.yaxis.set_minor_locator(locmin)
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return fesc_lw



plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] =16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 20
#plt.style.use('dark_background')

fig = plt.figure(figsize=(7, 7), dpi=144)

ax1 = fig.add_axes([0.17, 0.1, 0.66, 0.26])
ax3 = fig.add_axes([0.17, 0.4, 0.66, 0.26])
ax5 = fig.add_axes([0.17, 0.7, 0.66, 0.26])

plt.tick_params(axis='y',which='minor')

"""

timearr1, SFR1 = SFR(Part,read_new_01Zsun, 30, 480)
timearr2, SFR2 = SFR(Part,read_new_1Zsun, 30, 473)
timearr3, SFR3 = SFR(Part,read_new_gasrich, 30, 190)
timearr4, SFR4 = SFR(Part,read_new_1Zsun_highSN, 30, 240)
"""
ax11=ax1.twinx()
ax33=ax3.twinx()
ax55=ax5.twinx()
fesc1 = fescplot(ax5, ax55, Part, Fesc_new8, read_new_01Zsun, 2, 480, 'G9_Zlow',True, 'dodgerblue','r',True,True)

#fescplot(ax2, ax22, Part, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True,'grey', 'grey', False, False)
#fescplot(ax3, ax33, Part, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True,'grey', 'grey', False, False)
#fescplot(ax4, ax44, Part, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True,'grey', 'grey', False, False)
#fescplot(ax5, ax55, Part, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True,'grey', 'grey', False, False)
#fescplot(ax6, ax66, Part, Fesc_new8, read_new_01Zsun, 30, 480, 'G9_01Zsun',True,'grey', 'grey', False, False)

fesc5 =fescplot(ax3, ax33, Part, Fesc_new8, read_new_gasrich, 2, 300, 'G9_Zlow_gas5',True,'dodgerblue', 'r',  True, False)
fesc6 =fescplot(ax1, ax11, Part, Fesc_new8, read_new_1Zsun, 2, 480, 'G9_Zhigh',True,'dodgerblue', 'r',  True, False)

#fescplot(ax1, ax11, Part, Fesc_new8, read_new_01Zsun_lyaoff_se, 1, 30, 'G9_01Zsun_lyaoff',False, 'dodgerblue','r',True,True)
#fescplot(ax2, ax22, Part, Fesc_new, read_new_01Zsun_5pc_se, 3,380, 'G9_01Zsun_5pc',False,'dodgerblue', 'r',  True, False)
ax3.xaxis.set_major_formatter(plt.NullFormatter())
ax5.xaxis.set_major_formatter(plt.NullFormatter())


"""
ax3.set_xticks([])
ax4.set_xticks([])
ax5.set_xticks([])
ax6.set_xticks([])
ax2.set_yticks([])
ax4.set_yticks([])
ax6.set_yticks([])

ax11.set_yticks([])
ax33.set_yticks([])
ax55.set_yticks([])
"""
#ax1.legend(frameon=False)
ax1.set_xlabel('time (Myr)',fontsize=25)
#ax1.set_ylabel(r'$\langle f_{esc} \rangle$')
ax3.set_ylabel('$f_{esc}^{3D}$',fontsize=28)
ax33.set_ylabel('$dM_*/dt$ $(M_\odot/yr)$',fontsize=23)
#barx = ['G9_Zlow',  'G9_Zlow_gas5', 'G9_Zmid_SN5','G9_Zhigh','G9_Zhigh_SN5','G9_Zlow_HR']
#bary = [fesc1, fesc5,  fesc6, fesc2, fesc4]
color = ['black','dodgerblue','orange','firebrick','magenta','lightseagreen']
#ax7.bar(barx, bary,color=color,width=0.4)
#ax7.set_ylabel(r'$\langle f_{esc}^{3D} \rangle_{\scrL}$',fontsize=23)
#ax7.set_yscale('log')
#fig.text(0.44,0.03,'time (Myr)',fontsize=23)
#ax7.set_title('time (Myr)',fontsize=23,pad=28)
#ax7.set_ylim(5e-3, 0.2)
#plt.show()
plt.savefig('/Volumes/THYoo/kisti/plot/2019thesis/fesc_SFR_fig3_thesis1.pdf')
