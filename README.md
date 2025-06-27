# heparchy

[![DOI](https://zenodo.org/badge/411699933.svg)](https://doi.org/10.5281/zenodo.15753768)

Hierarchical database storage and access for high energy physics event data.

Docstrings are available, markdown documentation coming soon.

## Installation
```
pip install heparchy
```

# Features
- Fast and efficient storage
- Writes and reads from HDF5 files
- Data stored hierarchically
  - Files contain processes
  - Processes contain events
  - Events contain datasets for the final state particles
- Process level metadata can be attached
- Context managers provide access to these containers

## Data writing interface

### Process
- Metadata writing methods:
  - `string()`: MadGraph formatted string, _ie._ `p p > t t~`
  - `decay()`: pdgids of incoming and outgoing particles for the hard event
  - `com_energy()`
  - `signal_id()`: pdgid of the particle of interest in the hard event
  - `custom()`: extend with your own key / value metadata pair for the process

### Events
- Data writing methods for final state particles:
  - `pmu()`: 2d `numpy` array of 4-momenta, each row `[px, py, pz, e]`
  - `pdg()`: pdgids
  - `is_signal()`: boolean tags identifying if particle constituent of signal
  - `custom()`: extend with your own key / value dataset pair for the event

# Coming soon
- [X] Direct interface from HepMC files to HDF5 format
- [X] Jupyter notebook examples
- [X] Pip installation script

Warning: before the first release, the read interface may change
to improve consistency with the write interface.

Breaking changes will be avoided following the iminent release of 1.0.0.
