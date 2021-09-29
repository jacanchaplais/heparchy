import h5py

class DataWriter:
    """Class provides nested context managers for handling writing
    particle data for hierarchical access.
    
    The outer context manager (instantiated by this class) constructs
    and prepares the file for writing.

    The next layer down, a process context manager can be instantiated
    with the `new_process()` method. This acts as a container for
    hadronised showers generated with the same underlying process.
    It may be called repeatedly to store data from different processes
    within the same file.
    Optional methods are provided to write metadata to these containers,
    describing the hard event.
    The number of events written within this container is automatically
    stored to the metadata.

    The final layer is the event context manager, and can be
    instantiated with the `new_event()` method.
    This creates an event within the process container, and provides
    methods for writing particle data to the event.
    Repeatedly opening this context writes successive events under
    the same process.

    Keyword arguments:
        path (str): filepath for output.
            Note: currently only support .h5 and .hdf5 extensions.

    See `notebooks/conversion.ipynb` for example usage.
    """
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        self._buffer = h5py.File(self.path, 'w', libver='latest')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._buffer.close()

    def new_process(self, key: str):
        """Returns a context handler object for storage of data in
        a given hard process.

        Events can be iteratively added to this process by repeatedly
        calling the `new_event()` method, which itself returns a context
        handler object.

        Keyword arguments:
            key (str): arbitrary name, used to look up data with reader

        """
        return self.ProcessHandler(self, key)

    class ProcessHandler:
        def __init__(self, file_obj, key: str):
            self.__file_obj = file_obj
            self.key = key
            self._evt_idx = 0

        def __enter__(self):
            self._grp = self.__file_obj._buffer.create_group(self.key)
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            # count all of the events and write to attribute
            self._grp.attrs['num_evts'] = self._evt_idx

        def string(self, proc_str: str) -> None:
            """Writes the string formatted underlying event to the
            process metadata.
            
            Keyword arguments:
                proc_str (str): MadGraph formatted string representing
                the hard event, eg. p p > t t~
            """
            self._grp.attrs['process'] = proc_str

        def decay(self, in_pcls: tuple, out_pcls: tuple) -> None:
            """Writes the pdgids of incoming and outgoing particles to
            process metadata.

            Keyword arguments:
                in_pcls (tuple of ints): pdgids of incoming particles,
                    eg. for p p => (2212, 2212)
                out_pcls (tuple of ints): pdgids of outgoing particles,
                    eg. for t t~ => (6, -6)
            """
            self._grp.attrs['in_pcls'] = in_pcls
            self._grp.attrs['out_pcls'] = out_pcls

        def com_energy(self, energy: float, unit: str) -> None:
            """Writes the CoM energy and unit for the collision to
            process metadata.

            Keyword arguments:
                energy (float): eg. 13.0
                out_pcls (str): eg. 'GeV'
            """
            self._grp.attrs['com_e'] = energy
            self._grp.attrs['e_unit'] = unit

        def signal_id(self, signal_pcl: int) -> None:
            """Writes the pdgid of the signal particle, from the hard
            event, to process metadata.

            Keyword arguments:
                signal_pcl (int): pdgid for particle of interest from
                    hard event products

            eg. if you're tagging / clustering tops, use `signal_id(6)`.
            """
            self._grp.attrs['signal_pcl'] = signal_pcl

        def custom(self, name: str, metadata) -> None:
            """Store custom metadata to the process.

            Keyword arguments:
                name (str): handle to access the metadata at read time.
                metadata: str, int, float, or iterables thereof.
            """
            self._grp.attrs[name] = metadata

        def new_event(self):
            return self.EventHandler(self)

        class EventHandler:
            def __init__(self, grp_obj):
                self.__grp_obj = grp_obj # pointer to parent group obj
                self._idx = grp_obj._evt_idx # index for current event
                self.num_pcls = None

            def __enter__(self):
                self.__evt = self.__grp_obj._grp.create_group(
                        f'event_{self._idx:09}'
                        )
                return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                # increasing the index for the group to create next event
                # also acts as running counter for number events
                if self.num_pcls == None:
                    self.num_pcls = 0
                self.__evt.attrs['num_pcls'] = self.num_pcls
                self.__grp_obj._evt_idx += 1

            def __set_num_pcls(self, data):
                shape = data.shape
                num_pcls = shape[0]

                if self.num_pcls == None:
                    self.num_pcls = num_pcls
                elif num_pcls != self.num_pcls:
                    raise ValueError(
                            f'Datasets within same event must have the same '
                            + f'number of particles (rows, if 2D array). '
                            + f'Previous datasets had {self.num_pcls}, '
                            + f'attempted dataset would have {num_pcls}.'
                            )
                else:
                    return

            def __mk_dset(self, name: str, data, shape, dtype):
                """Generic dataset creation and population function.
                Wrap in methods exposed to the user interface.
                """
                # check data can be broadcasted to dataset:
                if data.squeeze().shape != shape:
                    raise ValueError(
                            f'Input data shape {data.shape} '
                            + f'incompatible with dataset shape {shape}.'
                            )
                dset = self.__evt.create_dataset(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        shuffle=True,
                        compression='lzf',
                        )
                dset[...] = data

            def pmu(self, data) -> None:
                """Write 4-momentum for all particles to event.
                
                Keyword arguments:
                    data ((num_pcls, 4) numpy array): 2d array of floats.
                        Each row contains momentum of one
                        particle, in order [px, py, pz, e].
                """
                self.__set_num_pcls(data)
                self.__mk_dset(
                        name='pmu',
                        data=data,
                        shape=(self.num_pcls, 4),
                        dtype='<f'
                        )

            def pdg(self, data) -> None:
                """Write pdg codes for all particles to event.
                
                Keyword arguments:
                    data (python iterable or 1d numpy array): 1d
                        iterable of ints.
                        Contains pdg codes for every particle in event.
                """
                self.__set_num_pcls(data)
                self.__mk_dset(
                        name='pdg',
                        data=data,
                        shape=(self.num_pcls,),
                        dtype='<i4',
                        )
            
            def is_signal(self, data) -> None:
                """Write bool signal marker for all particles to event.
                
                Keyword arguments:
                    data (python iterable or 1d numpy array): 1d
                        iterable of bools.
                        Contains True / False values for every particle
                        in event.
                        Note: would also accept int iterable of 0s / 1s.
                """
                self.__set_num_pcls(data)
                self.__mk_dset(
                        name='is_signal',
                        data=data,
                        shape=(self.num_pcls,),
                        dtype='<?',
                        )

            def custom_dataset(self, name: str, data, dtype) -> None:
                """Write a custom dataset to the event.

                Keyword arguments:
                    name (str): handle used when reading the data.
                    data (nd numpy array): data to store.
                    dtype: data type;
                        Note: using little Endian for builtin datasets.
                """
                self.__set_num_pcls(data)
                self.__mk_dset(
                        name=name,
                        data=data,
                        shape=data.shape,
                        dtype=dtype,
                        )
