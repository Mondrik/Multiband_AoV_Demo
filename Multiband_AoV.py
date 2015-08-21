# by Nicholas Mondrik & James Long
# An example code demonstrating how to construct a multiband AoV periodogram
# from its single band implementation

# Before running this code, you need to create an aov.so (or aov.pyd on Windows) shared object
# 1. Download http://users.camk.edu.pl/alex/soft/aovgui.tgz
# 2. Uncompress archive
# 3. In terminal run: f2py -m aov -c aovconst.f90 aovsub.f90 aov.f90
# 4. Place aov.so (or aov.pyd) in this directory

import numpy as np
import aov
import matplotlib.pyplot as plt

# returns list of times, magnitudes, and errors for a simulated light curve
# signal can be altered
# n is th number of observations in the light curve
def simulate_lc(n,white=False):
    t = np.cumsum(np.random.exponential(1,n))
    er = np.random.exponential(scale=1.3, size=n).astype('f')
    w = (2*np.pi)/0.5
    f = 1.5*np.sin(w*t + np.pi) + 0.1*np.sin(2*w*t + np.pi/2) + np.random.normal(0,scale=er,size=n)
    return [t,f,er]
 
# Given a list of periodograms, light curves (which consist of a time, magnitude,
# and error), the number of observations (taken to be constant across all bands), and 2x the
# number of harmonics (nh2), construct multiband AoV lightcurves.
# The frequencies the multiband periodogram is evaluated at are the same as those in its
# single band inputs.
# Variable names in general are similar to those from SC1996 
def makeMBpgram(pgrams,lcs,nobs,nh2):
	d1 = float(nh2)
	d2 = float(nobs-nh2-1)
	bands = ['u','g','r','i','z']
	c2 = np.zeros_like(pgrams[0])
	vrf_tot = 0.
	for i,p in enumerate(pgrams):
		data = lcs[i]
		rw = 1./data[:,2]
		m = data[:,1]
		#construct variance following AoV code (see aovmhw routine)
		avf = np.sum(m*rw*rw)/np.sum(rw*rw)
		vrf = np.sum(((m-avf)*rw)**2.)
		vrf_tot += vrf
		#invert eqn 11 of Schwarzenberg-Czerny (1996)
		c2 += d1*p*vrf / (d2+d1*p)
	#make multiband d_1,d_2.  This implementation works only if 
	#the same number of observations and harmonics are used in each single
	#band model. otherwise, the specific band values for nh2 and nobs should be used.
	d2 = float(d2*len(pgrams))
	d1 = float(d1*len(pgrams))
	MBpg = d2*c2/(d1*(vrf_tot-c2))
	return MBpg

# Fix seed for demonstrative purposes
np.random.seed(5)
bands = ['u','g','r','i','z']
nh2 = 2
n = 10
lcs = []
for b in bands:
	lc = simulate_lc(n)
	#plt.scatter(lc[0], lc[1])
	#plt.title("Band %s" % (b))
	#plt.show()
	lcs.append(lc)
lcs = np.array(lcs)

# choose frequency grid
# it is important that the frequency grid be exactly the same for each band
# in the single band implementation.  For simplicity, we use the same frequency 
# grid defined in the paper
fr0 = 1.
fru = 5.
frs = 0.0001


nfr = np.ceil((fru-fr0)/frs+1)
fs = fr0 + np.linspace(0,nfr-1,nfr)*frs


## compute periodogram for each band using multiharmonic AoV of CS
pgrams = []
for i in xrange(len(bands)):
	lc = lcs[i]
	cs_periodogram = aov.aov.aovmhw(lc[0],lc[1],lc[2],frs,nfr,fr0,nh2)
	pgrams.append(cs_periodogram[0])



for p in pgrams:
	plt.axhline(0) ## periodogram should be > 0
	plt.axvline(1./0.5,lw=2,ls='--',color='k') ## true period
	plt.plot(fs,p,'-b')
	plt.show()
plt.show()

plt.axhline(0) ## periodogram should be > 0
plt.axvline(1./0.5) ## true period
MB = makeMBpgram(pgrams,lcs,n,nh2)
plt.plot(fs,MB,'-k',lw=2)
plt.show()

