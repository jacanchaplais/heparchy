import h5py
from os.path import basename

class EventLoader:
    def __init__(self, path, key):
        self.path = path
        self.key = key
        self.__evt_iter = None
        self.__grp = None

    def __enter__(self):
        self.__buffer = h5py.File(self.path, 'r')
        self._meta = dict(self.__buffer[self.key].attrs)
        return self

    def __iter__(self):
        self.__evt_iter = iter(self.__buffer[self.key])
        return self

    def __next__(self):
        grp_key = next(self.__evt_iter)
        self.__grp = self.__buffer[self.key][grp_key]
        return self

    def __len__(self):
        return self.__buffer[self.key].attrs['num_evts']

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__buffer.close()

    # # process level interface
    # def __fmt_pdg_meta(self, key):
    #     meta = self._meta[key]
    #     str_to_set = lambda s: set(int(pdg) for pdg in s.split('|'))
    #     if isinstance(meta, str):
    #         return str_to_set(meta)
    #     else:
    #         pdg_ids = list()
    #         for pdg_str in meta:
    #             pdg_ids.append(str_to_set(pdg_str))
    #     return pdg_ids

    def get_ue_pcls(self, key, strict=True):
        """Returns the pdg(s) of the particles in the underlying event.
        Keyword arguments:
            key (str) -- possible values:
                - in_pcls:
                      the incoming particles, eg. p p, or e+ e-
                - out_pcls:
                      the (final) outgoing particles
                - signal_pcl:
                      the outgoing particle generating the signal jet
            strict (bool) -- if set to False, will return NoneType
                and issue a warning when data not found, instead of
                throwing an error
        Output:
            pcls (list of ints)
        """
        try:
            ids = self._meta[key]
        except KeyError:
            msg = f"Metadata with key {key} not found in {self.path}."
            if strict:
                print(msg)
                print("Aborting!")
                raise
            else:
                print(msg)
                print("Returning NoneType.")
                return None

        if hasattr(ids, '__iter__'):
            ids = [int(i) for i in ids]
        else:
            ids = int(ids)
        return ids

    def get_unit(self):
        return self._meta['unit']

    def get_com(self, key='com_energy'):
        return float(self._meta[key])

    # event level interface
    def get_pmu(self, key='pmu'):
        return self.__grp[key][...]

    def get_signal(self, key='is_signal'):
        return self.__grp[key][...]

    def get_pdg(self):
        return self.__grp['pdg'][...]

    def get_custom(self, key):
        return self.__grp[key][...]

    def get_num_pcls(self, key='num_pcls'):
        return self.__grp.attrs[key]

    def get_evt_name(self):
        return basename(self.__grp.name)

    def set_evt(self, evt_num):
        self.__evt_iter = None
        grp_key = f'event_{evt_num:09}'
        self.__grp = self.__buffer[self.key][grp_key]
