"""

 2020 (c) piteren

    Parameters Dictionary (dna)
        - advanced control of dict update (of keys and values)
        - optionally loads from / saves to folder (dna)

"""

import copy
import itertools
import os
import shutil
from typing import Dict, Any

from ptools.textools.text_metrics import lev_dist
from ptools.lipytools.little_methods import prep_folder, w_pickle, r_pickle
from ptools.pms.paspa import PaSpa

"""
    DNA - parameters dictionary, ~kwargs
    {parameter_name: parameter_value}
"""
DNA_PARAM_NAME =    str
DNA_PARAM_VALUE =   Any
DNA =               Dict[DNA_PARAM_NAME, DNA_PARAM_VALUE]

FN_PREFIX = 'dna'   # default dna filename prefix


class ParaDict(dict):

    OBJ_SUFFIX = '.dct' # default dna obj(pickle) filename suffix
    TXT_SUFFIX = '.txt' # default dna text filename suffix

    def __init__(
            self,
            dct: DNA=           None,       # initial params dictionary
            dna_TFD: str=       None,       # dna top-folder
            dna_SFD: str=       None,       # dna sub-folder
            fn_pfx: str=        FN_PREFIX,  # dna filename prefix
            verb=               0):

        dict().__init__()
        self.verb = verb

        if not dct: dct = {}
        self.update(dct)

        if verb > 0: print('\n *** ParaDict *** initialized')

        self.dna_FD = None
        if dna_TFD:

            if not dna_SFD and 'name' in self: dna_SFD = self['name']
            assert dna_SFD, 'ERR: dna subfolder must be given!'

            prep_folder(dna_TFD)
            self.dna_FD = f'{dna_TFD}/{dna_SFD}'
            self._fpx = fn_pfx

            self.add_new(self.__load()) # add new from save

    # updates values of self with dct - BUT ALL keys from dct MUST BE PRESENT in self
    def refresh(
            self,
            dct :dict):
        for key in dct:
            assert key in self, f'ERR: key "{key}" from given dict (refresher) not present self'
        self.update(dct)

    # updates values of self keys with dct - BUT ONLY ALREADY PRESENT (in self) keys
    def update_present(
            self,
            dct :dict):
        for key in dct:
            if key in self:
                self[key] = dct[key]

    # extends self ONLY with NEW dct keys, default policy while loading dna from folder
    def add_new(
            self,
            dct :dict,
            check_params_sim=   True):

        if check_params_sim: self.check_params_sim(dct)
        for key in dct:
            if key not in self:
                self[key] = dct[key]

    # returns self as a (clear) dict
    def get_dict(self) -> dict:
        dna = {}
        dna.update(self)
        return dna

    # checks for params similarity, returns True if got similar
    def check_params_sim(
            self,
            params :dict or list,
            lev_dist_diff: int=     1):

        found_any = False

        # look for params not in self.keys
        paramsL = params if type(params) is list else list(params.keys())
        self_paramsL = list(self.keys())
        diff_paramsL = [par for par in paramsL if par not in self_paramsL]

        # prepare dictionary of lists of lowercased underscore splits of params not in self.keys
        diff_paramsD = {}
        for par in diff_paramsL:
            split = par.split('_')
            split_lower = [el.lower() for el in split]
            perm = list(itertools.permutations(split_lower))
            diff_paramsD[par] = [''.join(el) for el in perm]

        self_paramsD = {par: par.replace('_','').lower() for par in self_paramsL} # self params lowercased with removed underscores

        for p_key in diff_paramsD:
            for s_key in self_paramsD:
                sim_keys = False
                s_par = self_paramsD[s_key]
                for p_par in diff_paramsD[p_key]:
                    levD = lev_dist(p_par,s_par)
                    if levD <= lev_dist_diff: sim_keys = True
                    if s_par in p_par or p_par in s_par: sim_keys = True
                if sim_keys:
                    if not found_any:
                        print('\nParaDict was asked to check for params similarity and found:')
                        found_any = True
                    print(f' @@@ ### >>> ACHTUNG: keys \'{p_key}\' and \'{s_key}\' are CLOSE !!!')

        return found_any

    # returns deepcopy of self
    def deepcopy_copy(self): return ParaDict(copy.deepcopy(self))

    def __obj_FN(self): return f'{self.dna_FD}/{self._fpx}{ParaDict.OBJ_SUFFIX}'

    def __txt_FN(self): return f'{self.dna_FD}/{self._fpx}{ParaDict.TXT_SUFFIX}'

    # loads from folder
    def __load(self):
        if os.path.isfile(self.__obj_FN()):
            return r_pickle(self.__obj_FN(), obj_type=ParaDict)
        return {}

    # saves to folder
    def save(
            self,
            save_old=   True):

        prep_folder(self.dna_FD)

        if save_old:
            if os.path.isfile(self.__obj_FN()): shutil.copy(self.__obj_FN(), f'{self.__obj_FN()}_OLD')
            if os.path.isfile(self.__txt_FN()): shutil.copy(self.__txt_FN(), f'{self.__txt_FN()}_OLD')

        w_pickle(self, self.__obj_FN())
        with open(self.__txt_FN(), 'w') as file: file.write(str(self))

    # copies dna from one folder to another
    @staticmethod
    def copy_dnaFD(
            dna_SFD_S: str, # sub folder S
            dna_SFD_T: str, # sub folder T
            top_FD_S: str,  # top folder S
            top_FD_T: str,  # top folder T
            fn_pfx: str=    FN_PREFIX):

        sdna = ParaDict(
            dna_TFD=    top_FD_S,
            dna_SFD=    dna_SFD_S,
            fn_pfx=     fn_pfx)

        if 'name' in sdna: sdna['name'] = dna_SFD_T  # update name

        tdna = ParaDict(
            dna_TFD=    top_FD_T,
            dna_SFD=    dna_SFD_T,
            fn_pfx=     fn_pfx)

        tdna.update(sdna)
        tdna.save()

    # converts dict to nice string
    @staticmethod
    def dict_2str(dct: DNA) -> str:
        if dct:
            s = ''
            max_len_sk = max([len(k) for k in dct.keys()])
            for k, v in sorted(dct.items()): s += f'{str(k):{max_len_sk}s} : {str(v)}\n'
            return s[:-1]
        return '--empty dna--'

    def __str__(self):
        return self.dict_2str(self)


# genetic xrossing for two ParaDict objects (saved already in folders)
def do_GX(
        name_A: str,                    # name of parent A (dna subfolder name)
        name_B: str,                    # name of parent B (dna subfolder name)
        name_child: str,                # name of child (dna subfolder name)

        paspa: PaSpa,                   # parameters space of both points

        folder_A: str,                  # top folder A
        folder_B: str=      None,       # top folder B
        folder_child: str=  None,       # top folder child

        fn_pfx: str=        FN_PREFIX,  # dna filename prefix
        ax_dst=             0.05,
        allow_full_tuple=   True):

    if not folder_B: folder_B = folder_A
    if not folder_child: folder_child = folder_A

    pa_dna = ParaDict(
        dna_TFD=    folder_A,
        dna_SFD=    name_A,
        fn_pfx=     fn_pfx)

    pb_dna = ParaDict(
        dna_TFD=    folder_B,
        dna_SFD=    name_B,
        fn_pfx=     fn_pfx)

    pc_dna = paspa.sample_GX_point(
        pa=                 pa_dna,
        pb=                 pb_dna,
        ax_dst=             ax_dst,
        allow_full_tuple=   allow_full_tuple)

    """
    # TODO: maybe important case when paspa.axes set is only a subset of ParaDict
    # add rest of keys
    for k in pa_dna:
        if k not in paspa.:
            pc_dna[k] = pa_dna[k]
    """

    cfmd = ParaDict(
        dct=        pc_dna,
        dna_TFD=    folder_child,
        dna_SFD=    name_child,
        fn_pfx=     fn_pfx)
    cfmd.save()