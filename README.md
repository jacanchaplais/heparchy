# heparchy
Hierarchical database storage and access for high energy physics event data.

Docstrings are available, markdown documentation coming soon.

## Features
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
  - `pmu()`: 4-momentum, 2d `numpy` array, each row `[px, py, pz, e]` for a
    single particle
  - `pdg()`: pdgids
  - `is_signal()`: boolean tags identifying if particle constituent of signal
  - `custom()`: extend with your own key / value dataset pair for the event

## Coming soon
- [ ] Direct interface from LHE and HepMC files to HDF5 format
- [ ] Jupyter notebook examples
