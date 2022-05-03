---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python [conda env:panhelio] *
    language: python
    name: conda-env-panhelio-py
---

SpacePy Tutorial -- Cusp Energetic Particles
====================================

This tutorial reproduces key figures from "Association of cusp energetic ions with geomagnetic storms and substorms" (Niehof et al, 2012; [doi:10.5194/angeo-30-1633-2012](https://doi.org/10.5194/angeo-30-1633-2012)).

It illustrates several functions in SpacePy and the scientific Python ecosystem:

  - Import of IDL data [scipy.io](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.readsav.html#scipy.io.readsav)
  - Omni and related solar wind data [spacepy.omni](https://spacepy.github.io/omni.html)
  - Pressure-corrected Dst* [spacepy.empiricals.getDststar](https://spacepy.github.io/autosummary/spacepy.empiricals.getDststar.html#spacepy.empiricals.getDststar)
  - Workflow support for by-hand event identification [spacepy.plot.utils.EventClicker](https://spacepy.github.io/autosummary/spacepy.plot.utils.EventClicker.html#spacepy.plot.utils.EventClicker)
  - Superposed epoch analysis [spacepy.seapy](https://spacepy.github.io/seapy.html)
  - Point-processes [spacepy.poppy](https://spacepy.github.io/poppy.html)


Setup
--------
This tutorial uses solar wind and leapsecond data that SpacePy normally maintains on a per-user basis. (To download this data on your own installation of SpacePy, use [toolbox.update()](https://spacepy.github.io/autosummary/spacepy.toolbox.update.html#spacepy.toolbox.update)).

However, for the purposes of this summer school, we have provided a shared directory with the normal SpacePy configuation and managed data. This saves us from waiting for 300 people to download at once. There are also other data files specific to this project.

So we use a single directory containing all the data for this tutorial and also the `.spacepy` directory (normally in a user's home directory). We use an environment variable to [point SpacePy at this directory](https://spacepy.github.io/configuration.html) before importing SpacePy; although we set the variable in Python, it can also be set outside your Python environment. Most users need never worry about this.

```python
tutorial_data = 'spacepy_tutorial'  # All data for this summer school, will be used throughout
import os
os.environ['SPACEPY'] = tutorial_data  # Use .spacepy directory inside this directory
```

Background
------------------
This study relates to energetic ions observed in the Earth's magnetospheric cusp and the connection to the tail region. In order to illustrate this, we also will illustrate the use of built-in magnetic field models and their visualization--using a sledghammer to kill a flea.

```python
import datetime

import matplotlib.colors
import matplotlib.pyplot
import numpy
import spacepy.coordinates
import spacepy.empiricals
import spacepy.irbempy
import spacepy.pybats
import spacepy.pybats.trace2d
import spacepy.time

# Coordinates in GSE expressed as an XZ grid (will assume Y=0, noon-midnight plane)
x = numpy.arange(-15, 10.1, 0.1)
z = numpy.arange(-8, 8.1, 0.1)
# Repeat so that every X component is repeated across every Z, and throw in Y=0
_ = numpy.meshgrid(x, 0, z)
# Combine into a single array
_ = numpy.stack(_, axis=-1)
# And flatten out to an array of (x, y, z)
location = _.reshape(-1, 3)
location = spacepy.coordinates.Coords(location, 'GSE', 'car', use_irbem=False)
# Assume a time 
ticks = spacepy.time.Ticktock([datetime.datetime(2000, 4, 6)] * len(location))
b = spacepy.irbempy.get_Bfield(ticks, location, extMag='T96')
b_mag, b_vec = b['Blocal'], b['Bvec']
b_mag = b_mag.reshape(len(x), len(z))
c = spacepy.coordinates.Coords(b_vec, 'GEO', 'car', ticks=ticks, use_irbem=False)
b_vec = c.convert('GSE', 'car').data.reshape(len(x), len(z), 3)
b_hat = b_vec / b_mag[..., None]
fig = matplotlib.pyplot.figure(dpi=150)
ax = fig.add_subplot(111)
ax.pcolormesh(x, z, b_mag.transpose(), norm=matplotlib.colors.LogNorm(vmin=10,vmax=1e4))
ax.set_aspect('equal')
spacepy.pybats.add_planet(ax)
ax.quiver(x[::4], z[::4], b_hat[::4,::4, 0].transpose(), b_hat[::4,::4, 2].transpose(), units='x')
for lat in numpy.arange(-180, 185, 5):
    direction = 2 * (abs(lat) > 90) - 1
    l = numpy.radians(lat)
    startx = numpy.sin(l)
    startz = numpy.cos(l)
    tracex, tracez = spacepy.pybats.trace2d.trace2d_rk4(
        b_hat[..., 0].transpose() * direction,
        b_hat[..., 2].transpose() * direction,
        startx, startz, x, z, ds=0.1)
    r = numpy.sqrt(tracex ** 2 + tracez ** 2)
    hit_earth = numpy.nonzero(r < .95)[0]
    if len(hit_earth):
        tracex = tracex[:hit_earth[0]]
        tracez = tracez[:hit_earth[0]]
    ranged_out = numpy.nonzero((tracex < min(x)) | (tracex > max(x)) | (tracez > max(z)) | (tracez < min(z)))[0]
    if len(ranged_out):
        tracex = tracex[:ranged_out[0]]
        tracez = tracez[:ranged_out[0]]
    ax.plot(tracex, tracez, color='k', marker='', lw=0.75)
mp_loc = spacepy.empiricals.getMagnetopause(spacepy.time.Ticktock(datetime.datetime(2000, 4, 6)))
ax.autoscale(enable=False)
ax.plot(mp_loc[0, :, 0], mp_loc[0, :, 1], color='r', marker='', lw=1)


```

```python

```
