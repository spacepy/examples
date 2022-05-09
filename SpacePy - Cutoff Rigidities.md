---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# SpacePy Tutorial -- Epoch-Dependent Rigidity Cutoff

### Background
Earth's magnetic field provides protection from high energy charged particles originating outside the magnetosphere, such as solar energetic particles (SEPs) and galactic cosmic rays (GCRs). Properties of both the particle (mass, energy, charge) and the magnetic field (strength, topology)  will determine how deep into the magnetosphere a particle can penetrate.

This example is largely inspired by ([Smart and Shea, 1993](https://adsabs.harvard.edu/full/1993ICRC....3..781S)). It will also present the energetic charged particle data from the Global Positioning System constellation ([Morley et al., 2017](https://doi.org/10.1002/2017SW001604)).

It illustrates several areas of functionality in SpacePy and the broader scientific Python ecosystem, as well as some approaches less widely used in academic programming, including:

  - Simple file retrieval from the web
  - Working with JSON-headed ASCII [spacepy.datamodel](https://spacepy.github.io/datamodel.html)
  - Time-system conversion [spacepy.time](https://spacepy.github.io/time.html) and [astropy.time](https://docs.astropy.org/en/stable/time/index.html)
  - Binning 1D data into 2D [spacepy.plot.Spectrogram](https://spacepy.github.io/autosummary/spacepy.plot.spectrogram.html)
  - Classes and inheritance

### Setup

```python
import glob
import datetime as dt
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

import spacepy.datamodel as dm
from spacepy import igrf
import spacepy.plot as splot
import spacepy.time as spt

# for convenient notebook display and pretty out-of-the-box plots...
%matplotlib inline
splot.style('default')
```

### Illustrating geomagnetic shielding using GPS particle data

To illustrate what rigidity is and does, we'll start with some energetic charged particle data from the GPS constellation. If you've already downloaded this data, just skip the cell.

```python
satnums = [64, 65, 66, 68, 69, 71]
nsdir = 'https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/gps/data/ns{}/'
gpsfile = 'ns{}_150621_v1.09.ascii'
import urllib.request
for ns in satnums:
    req = urllib.request.urlretrieve(''.join([nsdir.format(ns), gpsfile.format(ns)]), gpsfile.format(ns))
```

This will download one file of charged particle data from each of the satellites listed (ns64, etc.), which have a Combined X-ray Dosimeter (CXD) on board that measures energetic electrons and protons. The data is provided using an ASCII format that is self-describing. Think of it as implementing something like HDF5 or NASA CDF in a text file. The metadata and non-record-varying data are stored in the header to the ASCII file, which is encoded using JSON (JavaScript Object Notation). There's a convience routine to read these in `spacepy.datamodel`.

```python
gps = dm.readJSONheadedASCII(gpsfile.format(71))
```

This reads the file all-at-once into a `spacepy.datamodel.SpaceData`, which is a dictionary that also carries metadata. The data are then stored as arrays (`spacepy.datamodel.dmarray`) that also carry metadata. We can quickly inspect the contents with the `.tree()` method, and can also see data types and metadata entries at-a-glance by adding the `verbose` and `attrs` keywords.

```python
gps.tree(verbose=True, attrs=True)
```

Most commonly we deal with times expressed in UTC, that is, Coordinated Universal Time. UTC is the reference time for everyday life: this notebook is begin written in a timezone that's UTC-6 hours. However, UTC isn't a continuous time scale with a constant number of seconds per day. Sometimes leap seconds are applied, and these aren't known years in advance. So a lot of applications and operational systems will use time scales that don't use leap seconds. Common scales include TAI (International Atomic Time), GPS (Global Positioning System) time, and nanoseconds since the J2000 epoch.

Our GPS particle data file has the time written out in GPS time, so we want to convert that to UTC.


But first, if we just interpret the year and decimal day as being UTC what would we get? Let's try it. We can use `spacepy.toolbox.doy2date` here to convert from a fractional Day of Year to a datetime.

```python
datearray = spt.doy2date(gps['year'].astype(int), gps['decimal_day'], dtobj=True, flAns=True)
print(datearray[0])
```

But this doesn't take leap seconds into account. Since the zero epoch of the GPS time scale is defined as 1980-01-96T00:00:00 UTC we can convert the GPS date/time into "GPS seconds", which is a fairly standard way of expressing the system. That is, seconds since the GPS zero epoch.

Once we know the GPS seconds, we can get UTC (or other time systems) from either `spacepy.time.Ticktock` or `astropy.time.Time`.

```python
def ticks_from_gps(year, decday, use_astropy=False):
    '''Get a Ticktock from the year and decimal day in GPS time

    Notes
    -----
    1 - The decimal day is given as "GPS time" which is offset
    from UTC by the number of leapseconds since 1980.
    2 - The timestamps correspond to the midpoints of the integration
    intervals
    '''
    intyear = year.astype(int)
    datearray = spt.doy2date(intyear, decday, dtobj=True, flAns=True)
    # this is GPS time, so needs to be adjusted by leap seconds
    GPS0 = dt.datetime(1980, 1, 6)  # Zero epoch for GPS seconds system
    gpsoffset = datearray - GPS0
    gpsseconds = [tt.total_seconds() for tt in gpsoffset]
    if not use_astropy:
        return spt.Ticktock(gpsseconds, dtype='GPS')
    else:
        import astropy.time
        return astropy.time.Time(gpsseconds, format='gps')
```

```python
gpticktock = ticks_from_gps(gps['year'], gps['decimal_day'])
print(gpticktock.UTC[0], type(gpticktock))
```

`spacepy.time.Ticktock` manages things internally as TAI, but the string representation is assumed to be UTC for input and output. So if we print the string (ISO8601 format) we get:

```python
print(gpticktock.ISO[0])
```

This is missing the milliseconds that we saw above. What's happening is that the precision shown by `.ISO` defaults to seconds. It's just a display setting, so let's show the full precision.

```python
gpticktock.isoformat('microseconds')
print(gpticktock.ISO[0])
```

If you are using astropy or sunpy then you may want the time as an `astropy.time.Time`, so let's check that our times agree.

```python
gpastro = ticks_from_gps(gps['year'], gps['decimal_day'], use_astropy=True)
print(gpastro.iso[0], type(gpastro))
```

Note that you can make a `spacepy.time.Ticktock` directly from an `astropy.time.Time` by specifying the `APT` datatype on instantiation. For convenience moving forward, let's add this to our GPS data collection. We can also verify that we still have the same times...

```python
gps['Time'] = spt.Ticktock(gpastro, dtype='APT')
print(gps['Time'].UTC[0], type(gps['Time']))
gps['Time'].isoformat('microseconds')  # set this for display purposes later
```

<!-- #region -->
##### Question: If we have times as a `spacepy.time.Ticktock`, how can we convert that to an `astropy.time.Time`?
<details>
    <summary>(Click for answer)</summary>

<p>

```python
astrotime = gps['Time'].APT
```

</p>
Yes, it's that simple. `astropy.time.Time` is supported both as an input data type and an output data type.
</details>

<!-- #endregion -->

### Showing geomagnetic shielding of solar energetic particles
Let's plot some GPS proton data!

```python
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.semilogy(gps['Time'].UTC, gps['proton_integrated_flux_fit'][:, 0])
ax.set_ylabel('Integral p$^{+}$ flux [pfu]')
_ = splot.applySmartTimeTicks(ax, gps['Time'].UTC)
```

```python
specdata = dm.SpaceData()
mask = gps['proton_integrated_flux_fit'][:, 0] > 0
mask = np.logical_and(mask, gps['L_LGM_T89IGRF'] < 20)
specdata['L'] = gps['L_LGM_T89IGRF'][mask]
specdata['Time'] = gps['Time'].UTC[mask]
specdata['Data'] = gps['proton_integrated_flux_fit'][mask, 0]
spec = splot.Spectrogram(specdata, variables=['Time', 'L', 'Data'],
                         xlim=spt.Ticktock(['2015-06-22','2015-06-24']).UTC.tolist(),
                         ylim=[4, 12],
                         extended_out=True)
ax2 = spec.plot(figsize=(10, 5), cmap='gnuplot2')
```

```python
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
fns = glob.glob('ns*ascii')

gpsmulti = dm.readJSONheadedASCII(fns)
gpsmulti['Time'] = ticks_from_gps(gpsmulti['year'], gpsmulti['decimal_day'])
mask = gpsmulti['proton_integrated_flux_fit'][:, 2]> 0
mask = np.logical_and(mask, gpsmulti['L_LGM_T89IGRF'] < 20)

specdata = dm.SpaceData()
specdata['L'] = gpsmulti['L_LGM_T89IGRF'][mask]
specdata['Time'] = gpsmulti['Time'].UTC[mask]
specdata['Data'] = gpsmulti['proton_integrated_flux_fit'][mask, 2]

# Explicitly set bin sizes
# Time bins are 90 minutes
tstart = '2015-06-22T12:00:00'
tend = '2015-06-23T12:00:00'
tbins = spt.tickrange(tstart, tend, deltadays=1.5/24).UTC
# L bins are variable 1/3, 1/2, and 1
ybins = np.hstack([np.arange(4, 5, 0.125),
                   np.arange(5, 7, 1/3),
                   np.arange(7, 9, 1)])  # uneven bins are allowed!

spec = splot.Spectrogram(specdata, variables=['Time', 'L', 'Data'],
                         bins=[tbins, ybins],
                         xlim=spt.Ticktock([tstart, tend]).UTC.tolist(),
                         ylim=[4, 8],
                         extended_out=True)
_ = spec.plot(target=ax, cmap='gnuplot2',
              ylabel='L (T89)',
              colorbar_label='Int. Flux [cm$^{-2}$s$^{-1}$sr$^{-1}$MeV$^{-1}$]')
```

As you can see, throughout the SEP interval the fluxes at higher L remain elevated. At lower L the fluxes start to drop. This is due to more of the lower energy particles being excluded due to the higher magnetic field strength, thus the overall flux drops.

### Modeling Particle Access
While the Störmer model takes the direction into account, we'll just use the vertical approximation for our calculations to save on code. For a full implementation see the SHIELDS-PTM code.


First, import the packages we'll be using. We'll be using the `IGRF` module in SpacePy to evaulate Earth's dipole moment. Currently this isn't a fully-featured IGRF implemenation - it supports the coordinate system transformations and so we can get the dipole axis and the magnetic moment. First we need to make an IGRF object, then initialize it for a specific time.
##### Before you run the next cell, any guesses on roughly what the magnetic moment is?

```python
magmod = igrf.IGRF()
magmod.initialize(gps['Time'][0])
print('The centered dipole magnetic moment at {} is {:0.2f} nT'.format(gps['Time'].ISO[0], magmod.moment['cd']))
```

```python
def get_stormer_coefficient(cd_moment):
    """Set rigidity coefficient as dipole moment in mixed units

    See section 2 of Smart and Shea (1993).

    References
    ----------
    - Smart, D. F. and Shea, M. A., “The Change in Geomagnetic Cutoffs Due to
      Changes in the Dipole Equivalent of the Earth's Magnetic Field”, in
      23rd International Cosmic Ray Conference (ICRC23), Volume 3, 1993, p. 781.
    """
    re_cm = 6371.0008*1e5  # Volumetric Earth radius in cm
    mom_gauss_cm3 = (cd_moment/1e5)*re_cm**3
    gauss_cm = mom_gauss_cm3/re_cm**2
    # Now apply unit conversion, eV to Gev and volts to abvolts
    coeff = 300*gauss_cm/1e9
    coeff_v = coeff/4  # vertical reduction
    return coeff_v
```

### Impact of varying dipole moment 
The Störmer model of geomagnetic rigidity cutoff depends on the strength of the Earth's dipole moment. This varies with time, although commonly the values are taken for reference epochs. E.g., the commonly-used value of 60 corresponds to a reference epoch around 1932. For the case of vertically-incident particles the equation can be simplified and a factor of 4 appears in the denominator. Thus this is often presented as a coefficient with a value of 15. Later work by Smart and Shea uses a vertically-incident coefficient of 14.5. This corresponds to a reference epoch of around 1988.

To bring previous explorations of how the time-varying dipole moment impacts the coefficient used in the Störmer model, we'll graph the variation with time here.


For this work we need to convert between particle energy, and particle _rigidity_. I'm borrowing the `Particle` class from our particle tracing code `SHIELDS-PTM` as an illustration of how to make simple classes and re-use them.

```python
from abc import ABC, abstractmethod

# This is from the ptm_python.ptm_tools in the SHIELDS-PTM repository
class Particle(ABC):
    """Generic particle container

    Subclass to make proton, etc.
    """
    @abstractmethod
    def __init__(self):
        self._checkvalues()

    def _checkvalues(self):
        assert self.energy
        assert self.charge
        assert self.restmass
        assert self.mass

    def getRigidity(self, units='GV'):
        """Calculate rigidity in GV

        Energy & rest mass energy are in MeV
        Mass is in atomic mass number
        Charge is in units of elementary charge (proton is 1, electron is -1)
        """
        mcratio = self.mass/self.charge
        en_part = self.energy**2 + 2*self.energy*self.restmass
        rigidity_MV = mcratio * np.sqrt(en_part)
        if units.upper() == 'GV':
            rigidity = rigidity_MV/1e3
        else:
            raise NotImplementedError('Units other than GV for rigidity are not supported')
        return rigidity

    @classmethod
    def fromRigidity(cls, rigidity_GV):
        """Given rigidity in GV, make particle
        """
        rmv = rigidity_GV*1e3
        asq = cls.mass**2
        rmsq = cls.restmass**2
        csq = cls.charge**2
        part = asq*(asq*rmsq + csq*rmv**2)
        e_k = (np.sqrt(part) - asq*cls.restmass)/asq
        return cls(e_k)
```

Now that we have an abstract class for _any_ particle, let's make one specifically for protons. When using physical constants it's a good idea to be both consistent and precise. A lot of useful constants are given in `scipy.constants`.

```python
class Proton(Particle):
    charge = 1
    mass, _, _ = constants.physical_constants['proton mass in u']  # AMU
    restmass, _, _ = constants.physical_constants['proton mass energy equivalent in MeV']

    def __init__(self, energy):
        self.energy = energy
        super().__init__()
```

So how does this work? The `classmethod` means that we expect to use the method without directly making a `Particle` first.

```python
myproton = Proton.fromRigidity(0.5)  # Half a GV
print(myproton.energy)
```

##### Exercises: What would classes for electrons and O<sup>2+</sup> look like?


Now we'll instatiate an IGRF object and set up an array of times that we want to evaluate the magnetic moment at.

IGRF13 currently covers the period 1900 to 2025, so we can update the results presented by Smart and Shea to use one consistent model from 1900 through 2025. In their paper, pre-1945 was taken from work by Akasofu and Chapman (1972) and the last year shown was 1990. As the IGRF can be slow to evaluate and the magnetic moment veries slowly, we'll do one data point every 30 days. This give us approximately 1500 points sampled over the data range, compared to the 16 used by Smart and Shea.

```python
magmod = igrf.IGRF()
epochs = spt.tickrange('1900-1-1', '2025-1-1', 30)
print('Number of epochs is {}'.format(len(epochs)))
```

The next step is to loop over the times and, for each, calculate the moment of the centered dipole. This might take a couple of seconds as evaluating IGRF is relatively computationally expensive.

```python
moments = np.empty(len(epochs))
for idx, tt in enumerate(epochs):
    magmod.initialize(tt)
    moments[idx] = (magmod.moment['cd'])
```

The coefficient used in the vertical simplification of the Störmer model is equivalent to the cutoff rigidity for vertical incidence at L=1. This would be multiplied by 4 to obtain the moment in the mixed units used by Störmer.

```python
lvals = [1]
cutoffs = np.empty([len(epochs), len(lvals)])
for idx, mm in enumerate(moments):
    sv = ptt.StormerVertical(mm)
    cutoffs[idx, :] = sv.cutoff_at_L(lvals, as_energy=False)
```

Having obtained the array of cutoff rigidities (or, equivalently, the coefficient for vertical incidence) as a function of epoch, we can plot these. We'll also convert the rigidities used in the tick labels to cutoff energy (assuming protons).

```python
dates = epochs.UTC
ind15 = np.argmin(np.abs(cutoffs - 15))
ind14p5 = np.argmin(np.abs(cutoffs - 14.5))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epochs.UTC, cutoffs)
ax = plt.gca()
ax2 = ax.twinx()
ax2.grid(False)
ax.set_ylabel('Vert. Cutoff Rigidity @ L=1 [GV]')
ax.axvline(dates[ind15], c='k', ls='--')
ax.axvline(dates[ind14p5], c='k', ls='--')
ax.hlines(14.5, dates[ind14p5-50], dates[ind14p5+50], colors='darkgrey', ls='--')
_ = ax.text(dates[ind15+10], 15.2, dates[ind15].year)
_ = ax.text(dates[ind14p5+10], 14.7, dates[ind14p5].year)
plt.draw()
rig_vals = ax.get_yticklabels()
ax2.plot(dates, cutoffs)
ax2.set_yticklabels(['{:.2f}'.format(ptt.Proton.fromRigidity(float(rr.get_text())).energy/1e3) for rr in rig_vals])
ax2.set_ylabel('Proton Energy [GeV]')
_ = ax.hlines(15, dates[ind15-50], dates[ind15+50], colors='darkgrey', ls='--', zorder=99)

```

(If you're reading this as markdown rather than running as a Jupyter notebook, the output figure will look like the image embedded below)

![rigidity_variation](rigidity_variation.png)

```python
print(epochs[np.argmin(np.abs(cutoffs - 15))])
print(epochs[np.argmin(np.abs(cutoffs - 14.5))])
```

```python
dif = [(ep - dt.datetime(2017,9,9)).days for ep in epochs.UTC]
print(np.argmin(np.abs(dif)))
```

```python
epochs.ISO[1433]
```

```python
cutoffs[1432:1435]
```

```python
magmod.initialize(dt.datetime(2017,9,9))
```

```python
magmod.moment
```

```python

```
