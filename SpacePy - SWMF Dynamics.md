---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# SpacePy Tutorial -- SWMF Dynamics

## Background
The Space Weather Modeling Framework (SWMF) is a powerful coupled-model approach for investigating the system dynamics of the magnteophere, ionosphere, ring current, and other regions of geospace. It couples several models together to create self-consistent simulations of geospace. Most commonly, this includes BATS-R-US, the Ridley Ionosphere Model (RIM), and one of several ring current models. The output from these simulations can be complicated to work with as they have different formats, different values, and different approaches to analysis. This tutorial demonstrates how to use the Spacepy Pybats module to explore and analyze SWMF output.

Specifically, we're going to explore the result set from [Welling et al., 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020SW002489), Figures 2 and 3. The SWMF was used to explore the geospace response to a hypothetical worst-case-scenario storm sudden commencement. A subset of outputs are provided in the tutorial data directory representing the two simulations explored in the paper: an extreme CME arrival with *northward* IMF, and an extreme CME arrival with *southward* IMF. 

For our purposes, let's assume that the dataset has just been downloaded from the supercomputing environment. We do not know if the results are valid and reasonable yet. We need to first confirm that the results are reasonable. Then we want to illustrate the magnetosphere's response to the CMEs, including the compression of the dayside magnetopause, the formation of Region 1 Field-Aligned Currents (R1 FACs), and the time dynamics of the magnetopause. This will require opening files from BATS-R-US and RIM, creating 2D figures, and performing field-line tracing to see the open-closed boundary.

Concepts explored in this tutorial include,

  - Opening SWMF output files using [spacepy.pybats](https://spacepy.github.io/pybats.html).
  - Using the features of [spacepy.datamodel](https://spacepy.github.io/datamodel.html) to explore unfamiliar data sets.
  - Using [spacepy.pybats](https://spacepy.github.io/pybats.html) classes to analyze MHD output, extract values at arbitrary points, and trace magnetic field lines.
  - Various features of [spacepy.plot](https://spacepy.github.io/plot.html), including the `target` keyword argument syntax.
  - Classes and inheritance in Python.



## Setup

As is the case for the other Spacepy tutorials, we use a single directory containing all the data for this tutorial and also the `.spacepy` directory (normally in a user's home directory). We use an environment variable to [point SpacePy at this directory](https://spacepy.github.io/configuration.html) before importing SpacePy; although we set the variable in Python, it can also be set outside your Python environment. Most users need never worry about this.

```python
tutorial_data = '/shared/jtniehof/spacepy_tutorial/'  #All data for this summer school, will be used throughout
import os
os.environ['SPACEPY'] = tutorial_data  # Use .spacepy directory inside this directory
```

Next, import the pertinent modules that will support this tutorial.

```python
import matplotlib.pyplot as plt
import spacepy.plot as splot
from spacepy import pybats

# for convenient notebook display and pretty out-of-the-box plots...
%matplotlib inline
splot.style('default')
```

## Diving into the data

Our first step is to look at the output files we have and determine what values they contain. In the example data folder, there are two folders: `swmf_ssi_north` and `swmf_ssi_south`. The "north" folder is an extreme SSI with northward IMF, the "south" folder are results with southward IMF. Each is organized in a similar manner to real world SWMF output directories. There is a sub-folder for the "global magnetosphere" component (`GM`) and the "ionospheric electrodynamics" component (`IE`). For this simulation, the codes used for these components are *BATS-R-US* and the *Ridley Ionosphere Model (RIM)*, respectively. Users may see additional folders if other components were included (e.g., `IM`, `PW`, etc.). Output from each code is placed into the respective folders.

Because the SWMF combines independently developed codes, we are often presented with a variety of data files of a variety of formats. Further, because these codes are highly configurable, the contents are often not clear to people not privvy to the simuation configuration. 
It may be instructive to [peruse the manual](http://herot.engin.umich.edu/~gtoth/SWMF/doc/SWMF.pdf)  to see just how many different output types are available (see the `#SAVEPLOT` command entries).
Our first job is to determine how to open these files and then see what values they contain. Peeking into the directories, this is what we see:

From BATS-R-US (`GM` directory):

- `y0*.outs` files: 2d cuts of the MHD result in the Y=0 plane.
- `log*.log` files: Timeseries of diagnostic values, Dst.

From RIM (`IE` directory):

- `it*.idl.gz` files: 2d ionospheric electrodynamic values.

Each of these has its own format type, either binary (as in the case with the `.outs` files) or ascii. Some formats are common across different models within the SWMF. In these cases, we can use the base classes within `pybats` to open and explore the contents. Specifically, `.out` and `.outs` files, either binary or ascii, can be opened with the`spacepy.pybats.IdlFile` class. Simple log files can often be loaded with the `spacepy.pybats.LogFile` class. Loading such files is a simple affair:

```python
# Some convenience variables to help us keep organized:
path_north = tutorial_data + 'swmf_ssi_north/'
path_south = tutorial_data + 'swmf_ssi_south/'

# Open the log file:
log = pybats.LogFile(path_north + 'GM/log_e20150321-054500.log')

# Open the 2D slice files:
mhd = pybats.IdlFile(path_north + 'GM/y=0_mhd_1_e20150321-060040-000_20150321-060510-000.outs')
```

*Note: When writing scripts, I often just use the glob module to avoid looking up every precise file name.*

The `log` and `mhd` objects are built on Spacepy's data model, giving us powerful tools to explore their contents.

- The `.attrs` object attributes list global-level attributes about the files.
- Values are accessed via key-value syntax, like dictionaries. The `.keys()` method will yield available values.
- Each value has its own `.attrs` attribute that gives value-level information when available.
- The `.tree()` object method can make short work of unravelling what's in the file.

Let's see what this looks like:

```python
mhd.attrs # Print out object-level attributes
```

```python
log.attrs # Same, but for our log file.
```

```python
mhd.keys() # Look at the values stored in the object
```

```python
mhd.tree(verbose=True, attrs=True) # Print out the "tree" representation of the file
```

Each value is a `spacepy.pybats.dmarray` object- this is a subclassed `numpy` array object but with `.attrs` attributes and other functionality. We can continue our exploration of the files:

```python
mhd['P'].attrs
```

```python
log['dst']
```

With the data available to us, we can plot and manipulate as would normally would with Numpy arrays and Matplotlib.


## Series of snapshots: `.out` vs. `.outs`

Before we continue, we need to discuss our MHD data file. It's actually a *time series* of data files concatenated together from previously separate files with the suffix `.out` (no *s* on *out*). Each snapshot of the results is called a *frame*. We need to be able understand how many frames are stored in the file, know what times they represent, and have a way to switch between them. As far as understanding the number and time stamps of the frames, that information is stored in the object-level attributes:

```python
print(f"There are {mhd.attrs['nframe']} frames in this file.")
mhd.attrs['times']
```

Conveniently, this file contains the precise epochs we wish to plot.

There's also information about the currently loaded frame:

```python
print(f"Currently loaded frame {mhd.attrs['iframe']} representing T={mhd.attrs['time']}")
```

We can switch frames using the `switch_frame()` object method. This will load only the data from the file corresponding to that frame, helping to manage memory usage on large (hundreds of gigabyte) files. Following Python conventions, we use zero-based indexing.

```python
mhd.switch_frame(3)
print(f"Currently loaded frame  is now #{mhd.attrs['iframe']} representing T={mhd.attrs['time']}")
```

## Find the right class for the job
The above syntax gets us to the data, but doesn't do us any favors in terms of doing actual work. Further, our RIM data is not in a common data format, so we haven't been able to open that data yet. Let's take a moment to see how *model- and output-specific classes* enable advanced features and powerful plotting methods. 

Our MHD data is from BATS-R-US, meaning we should turn to the *model-specific submodule* of pybats: `spacepy.pybats.bats`. Recognizing that the MHD files are 2D cuts, we can jump to the `spacepy.pybats.bats.Bats2d` class, which inherits from the `IdlFile` class and adds a lot of functionality.
