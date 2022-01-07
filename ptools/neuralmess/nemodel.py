"""

 2019 (c) piteren

 NEModel class: wraps graph building functions: fwd_func & opt_func with some features

    fwd_func:
        - should build complete model forward graph (FWD) - from PH (placeholders) to loss
        - function should return dict with: PH, tensors, variables lists
            - dict keys should meet naming policy
                - list of special keys to consider while building fwd_func is under SPEC_KEYS
                - OPT part will be build only if there is a "loss" tensor returned
                - dict may contain variables_lists and single variable under keys with 'var' in name
                    - variables returned under 'train_vars' key are optimized (if 'train_vars' key is not present all trainable vars are optimized)
                    - sub-lists of variables will serve for separate savers (saved in subfolders)
    opt_func - rather should not be replaced, but if you have to:
        - should accept train_vars and gradients parameters

    model name is resolved
        self_args_dict >> fwdf_mdict >> mdict >> timestamp

    graph building params may come from (in order of overriding):
        - NEModel defaults (like name, seed, ..)
        - params saved in folder
        - fwd_func & opt_func defaults
        - given kwargs

    - keeps graph params in self (dict)
    - tensors, placeholders, etc... returned by graphs are also kept in self (dict) keys
    - model objects (graph, session, saver ...) are kept as self fields

 NEModel class implements:
    - one folder for all model data (subfolder of save_TFD named with model name)
    - logger (txt file saved into the model folder)
    - GPU management with multi-GPU training (placement of model graph elements across devices)
    - builds optimization (OPT) graph part
        - calculates gradients for every tower >> averages them
        - AVT gradient clipping and scaled LR (warmup, annealing)
    - MultiSaver (with option for saving sub-lists of variables into separate checkpoints)
    - sanity check of many graph elements and dependencies
    - inits session, TB writer, MultiSaver loads variables (or inits them)
    - self as dict with all model parameters, tensors, PH, session, savers
    - baseline methods for training
"""

import numpy as np
import os
from typing import Callable
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ptools.lipytools.logger import set_logger
from ptools.lipytools.little_methods import get_defaults, short_scin, stamp,get_params
from ptools.lipytools.moving_average import MovAvg
from ptools.mpython.mptools import DevicesParam
from ptools.pms.paradict import ParaDict
from ptools.neuralmess.get_tf import tf
from ptools.neuralmess.base_elements import num_var_floats, lr_scaler, gc_loss_reductor, log_vars, mrg_ckpts
from ptools.neuralmess.dev_manager import tf_devices, mask_cuda
from ptools.neuralmess.multi_saver import MultiSaver
from ptools.neuralmess.batcher import Batcher


# restricted keys for fwd_func mdict and return dict (if they appear in mdict, should be named exactly like below)
SPEC_KEYS = [
    'name',                                             # model name
    'seed',                                             # seed for TF nad numpy
    'iLR',                                              # initial learning rate (base)
    'warm_up','ann_base','ann_step','n_wup_off',        # LR management (parameters of LR warmup and annealing)
    'avt_SVal','avt_window','avt_max_upd','do_clip',    # gradients clipping parameters
    'train_vars',                                       # list of variables to train (may be returned, otherwise all trainable are taken)
    'opt_vars',                                         # list of variables returned by opt_func
    'loss',                                             # loss
    'acc',                                              # accuracy
    'f1',                                               # F1
    'opt_class',                                        # optimizer class
    'batch_size',                                       # batch size
    'n_batches',                                        # number of batches for train
    'verb']                                             # fwd_func verbosity


SAVE_TFD =          '_models'
NEMODEL_DNA_PFX =   'nemodel_dna'

# default NEModel function for optimization graph
def opt_graph(
        train_vars,
        gradients,
        opt_class=                  tf.train.AdamOptimizer, # default optimizer, other examples: tf.train.GradientDescentOptimizer, partial(tf.train.AdamOptimizer, beta1=0.7, beta2=0.7)
        iLR=                        3e-4,
        warm_up=                    None,
        ann_base=                   None,
        ann_step=                   1,
        n_wup_off: float=           1,
        avt_SVal=                   1,
        avt_window=                 100,
        avt_max_upd=                1.5,
        do_clip=                    False,
        verb=                       0):

    g_step = tf.get_variable(  # global step variable
        name=           'g_step',
        shape=          [],
        trainable=      False,
        initializer=    tf.constant_initializer(0),
        dtype=          tf.int32)

    iLR_var = tf.get_variable(  # base LR variable
        name=           'iLR',
        shape=          [],
        trainable=      False,
        initializer=    tf.constant_initializer(iLR),
        dtype=          tf.float32)

    scaled_LR = lr_scaler(
        iLR=            iLR_var,
        g_step=         g_step,
        warm_up=        warm_up,
        ann_base=       ann_base,
        ann_step=       ann_step,
        n_wup_off=      n_wup_off,
        verb=           verb)['scaled_LR']

    # updates with: optimizer, gg_norm, avt_gg_norm
    loss_reductorD = gc_loss_reductor(
        optimizer=      opt_class(learning_rate=scaled_LR),
        vars=           train_vars,
        g_step=         g_step,
        gradients=      gradients,
        avt_SVal=       avt_SVal,
        avt_window=     avt_window,
        avt_max_upd=    avt_max_upd,
        do_clip=        do_clip,
        verb=           verb)

    # select OPT vars
    opt_vars = tf.global_variables(scope=tf.get_variable_scope().name)
    if verb>0:
        print(f' ### opt_vars: {len(opt_vars)} floats: {short_scin(num_var_floats(opt_vars))} ({opt_vars[0].device})')
        if verb>1: log_vars(opt_vars)

    rd = {}
    rd.update({
        'g_step':       g_step,
        'iLR_var':      iLR_var,
        'scaled_LR':    scaled_LR,
        'opt_vars':     opt_vars})
    rd.update(loss_reductorD)
    return rd


class NEModel(dict):

    def __init__(
            self,
            fwd_func,                               # function building forward graph (from PH to loss)
            opt_func=                   opt_graph,  # function building OPT graph (from train_vars & gradients to optimizer)
            devices: DevicesParam=      -1,         # check neuralmess.dev_manager.ft_devices for details
            do_optimization: bool=      True,       # add optimization part to the graph (for training)
            name_timestamp=             False,      # adds timestamp to name
                # default train parameters, may be overridden by given with kwargs graph params
            batch_size=                 64,
            n_batches=                  1000,
                # save
            save_TFD: str=              SAVE_TFD,   # top folder of model_FD
            savers_names: tuple=        (None,),    # names of savers for MultiSaver
            load_saver: bool or str=    True,       # for None does not load, for True loads default
            hpmser_mode: bool=          False,      # it will set model to be quiet and fast
            read_only=                  False,      # sets model to be read only - will not save anything to folder
            do_logfile=                 True,       # enables saving log file in save_TFD
            do_TB=                      True,       # runs TensorBard
            silent_TF_warnings=         False,      # turns off TF warnings
                # GPU management
            sep_device=                 True,       # separate first device for variables, gradients_avg, optimizer (otherwise those ar placed on the first FWD calculations tower)
            collocate_GWO=              False,      # collocates gradient calculations with tf.OPs (gradients are calculated on every tower with its operations, but remember that vars are on one device...) (otherwise with first FWD calculations tower)
            **kwargs):                              # here go all other params of graph (fwd / OPT)

        dict.__init__(self) # init self as a dict

        # hpmser_mode overrides
        self.hpmser_mode = hpmser_mode
        if self.hpmser_mode:
            kwargs['verb'] = 0
            read_only = True

        self.read_only = read_only
        # read only overrides
        if self.read_only:
            do_logfile = False
            do_TB = False

        self.do_TB = do_TB

        self.verb = kwargs['verb'] if 'verb' in kwargs else 0

        if silent_TF_warnings:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            warnings.filterwarnings('ignore')

        if self.verb>0: print(f'\n *** NEModel (type: {type(self).__name__}) *** initializes...')

        nemodel_args = {'name':'NEM', 'seed':12321} # NEModel default args

        fwd_func_defaults = get_defaults(function=fwd_func) # params dict with defaults of fwd_func

        # resolve model name (NEModel >> fwd_func_defaults >> mdict >> timestamp)
        resolved_name =                 nemodel_args['name']
        if 'name' in fwd_func_defaults: resolved_name =     fwd_func_defaults['name']
        if 'name' in kwargs:            resolved_name =     kwargs['name']
        if name_timestamp:              resolved_name +=    f'.{stamp()}'
        self.name = resolved_name
        kwargs['name'] = resolved_name
        if self.verb>0: print(f' > NEModel name: {self.name}')

        self.model_dir = f'{save_TFD}/{self.name}'
        if self.verb>0: print(f' > NEModel dir: {self.model_dir}{" read only mode!" if self.read_only else ""}')

        # ParaDict with empty dna, it gets dna from folder (if present)
        folder_para_dict = ParaDict(
            dna_TFD=    save_TFD,
            dna_SFD=    self.name,
            fn_pfx=     NEMODEL_DNA_PFX,
            verb=       self.verb)

        # set logfile
        if do_logfile:
            set_logger(
                log_folder=     self.model_dir,
                custom_name=    self.name,
                verb=           self.verb)

        # resolve graph_dna in proper order
        self.__fwd_graph_dna = self.__get_func_dna(
            func=               fwd_func,
            nemodel_defaults=   nemodel_args,
            dna_from_folder=    folder_para_dict,
            user_dna=           kwargs)
        if self.verb>0: print(f'\n > NEModel fwd_graph_dna : {self.__fwd_graph_dna}')
        self.__opt_graph_dna = self.__get_func_dna(
            func=               opt_func,
            nemodel_defaults=   nemodel_args,
            dna_from_folder=    folder_para_dict,
            user_dna=           kwargs)
        if self.verb>0: print(f' > NEModel opt_graph_dna : {self.__opt_graph_dna}')

        folder_para_dict.update(self.__fwd_graph_dna)
        folder_para_dict.update(self.__opt_graph_dna)

        # train params will be saved in folder also
        train_params = {
            'batch_size':   batch_size,
            'n_batches':    n_batches}
        folder_para_dict.update(train_params)

        # finally update with kwargs given by user not valid for fwd_func nor opt_func
        not_used_kwargs = {}
        for k in kwargs:
            if k not in self.__fwd_graph_dna and k not in self.__opt_graph_dna:
                not_used_kwargs[k] = kwargs[k]
        if self.verb>0 and not_used_kwargs: print(f' > NEModel kwargs not used by any graph : {not_used_kwargs}')
        folder_para_dict.update(not_used_kwargs)

        folder_para_dict.check_params_sim(SPEC_KEYS)  # safety check
        if not self.read_only: folder_para_dict.save()

        # finally update self
        self.update(nemodel_args)
        self.update(train_params)
        self.update(not_used_kwargs)
        self.update(self.__fwd_graph_dna)
        self.update(self.__opt_graph_dna)

        devices = tf_devices(devices, verb=self.verb)

        # mask GPU devices
        devices_other = []
        devices_gpu = []
        for device in devices:
            if 'GPU' in device: devices_gpu.append(device)
            else: devices_other.append(device)
        # print(f'got GPU devices: {devices_gpu} and other: {devices_other}')
        if devices_gpu:
            ids = [dev[12:] for dev in devices_gpu]
            # print('ids: {ids}')
            mask_cuda(ids)
            devices_gpu = [f'/device:GPU:{ix}' for ix in range(len(devices_gpu))]
            # print(f'masked GPU devices: {devices_gpu}')
        devices = devices_other + devices_gpu

        # report devices
        if self.verb>0:
            print()
            if len(devices)==1:
                if 'CPU' in devices[0]: print(f'NEModel builds CPU device setup')
                else:                   print(f'NEModel builds single-GPU setup')
            else:                       print(f'NEModel builds multi-dev setup for {len(devices)} devices')

        if len(devices)<3: sep_device = False # SEP is available for 3 or more devices

        # build FWD graph(s) >> manage variables >> build OPT graph
        self.gFWD = [] # list of dicts of all FWD graphs (from all devices)
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self['seed']) # set graph seed
            np.random.seed(self['seed'])
            if self.verb>0: print(f'\nNEModel set TF & NP seed to {self["seed"]}')

            # builds graph @SEP, this graph wont be run, it is only needed to place variables, if not vars_sep >> variables will be placed with first tower
            if sep_device:
                if self.verb>0: print(f'\nNEModel places VARs on {devices[0]}...')
                with tf.device(devices[0]):
                    fwd_func(**self.__fwd_graph_dna)

            tower_devices = [] + devices
            if sep_device: tower_devices = tower_devices[1:] # trim SEP
            for dev in tower_devices:
                if self.verb>0: print(f'\nNEModel builds FWD graph @device: {dev}')
                with tf.device(dev):
                    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                        self.gFWD.append(fwd_func(**self.__fwd_graph_dna))

            fwd_graph_return_dict = self.gFWD[0]
            if self.verb>0: print(f'dictionary keys returned by fwd_func ({fwd_func.__name__}): {fwd_graph_return_dict.keys()}')

            self.update(fwd_graph_return_dict) # update self with fwd_graph_return_dict

            # get FWD variables returned by fwd_func (4 saver)
            train_vars = [] # variables to train
            saver_vars = {} # dict of variables to save
            for key in self.keys():
                if 'var' in key.lower():
                    if key =='train_vars':
                        train_vars = self[key]
                        if type(train_vars) is not list: train_vars = [train_vars]
                    else:
                        if type(self[key]) is not list: saver_vars[key] = [self[key]]
                        else:                           saver_vars[key] = self[key]
            all_vars = tf.global_variables()

            # there are returned variables >> assert there are all variables returned in lists
            if saver_vars:
                all_vars_returned = []
                for key in saver_vars: all_vars_returned += saver_vars[key]
                there_are_all = True
                for var in all_vars:
                    if var not in all_vars_returned:
                        print(f' *** variable {var.name} not returned by fwd_func')
                        there_are_all = False
                assert there_are_all, 'ERR: there are some variables not returned by fwd_func in lists!'

            else: saver_vars['fwd_vars'] = all_vars # put all

            if self.verb>0:
                print('\nNEModel variables to save from fwd_func:')
                for key in sorted(list(saver_vars.keys())):
                    varList = saver_vars[key]
                    if varList: print(f' ### vars @{key} - num: {len(varList)}, floats: {short_scin(num_var_floats(varList))} ({varList[0].device})')
                    else: print(' ### no vars')
                    if self.verb>1: log_vars(varList)

            if 'loss' not in self:
                do_optimization = False
                if self.verb>0: print('\nthere is no loss in FWD graph, OPT graph wont be build')

            if not do_optimization:
                if self.verb>0: print('\nOPT graph wont be build')
            # build optimization graph
            else:
                if self.verb>0: print(f'\nPreparing OPT part with {self["opt_class"]}')
                # select trainable variables for OPT
                all_tvars = tf.trainable_variables()
                if train_vars:
                    # check if all train_vars are trainable:
                    for var in train_vars:
                        if var not in all_tvars:
                            if self.verb>0: print(f'variable {var.name} is not trainable but is in train_vars, please check the graph!')
                else:
                    for key in saver_vars:
                        for var in saver_vars[key]:
                            if var in all_tvars:
                                train_vars.append(var)
                    assert train_vars, 'ERR: there are no trainable variables at the graph!'
                # log train_vars
                if self.verb>0:
                    print('\nNEModel trainable variables:')
                    print(f' ### train_vars: {len(train_vars)} floats: {short_scin(num_var_floats(train_vars))}')
                    if self.verb>1: log_vars(train_vars)

                # build gradients for towers
                for ix in range(len(self.gFWD)):
                    tower = self.gFWD[ix]
                    tower['gradients'] = tf.gradients(
                        ys=                             tower['loss'],
                        xs=                             train_vars,
                        colocate_gradients_with_ops=    not collocate_GWO) # TF default is False >> calculates gradients where OPS, for True >> where train_vars

                    # log gradients
                    if self.verb>0:
                        nGrad = len(tower['gradients'])

                        # None_as_gradient case
                        device = 'UNKNOWN'
                        for t in tower['gradients']:
                            if t is not None:
                                device = t.device
                                break

                        print(f' > gradients for {ix} tower got {nGrad} tensors ({device})')
                        if self.verb>1:
                            print('NEModel variables and their gradients:')
                            for gix in range(len(tower['gradients'])):
                                grad = tower['gradients'][gix]
                                var = train_vars[gix]
                                print(var, var.device)
                                print(f' > {grad}') # grad as a tensor displays device when printed (unless collocated with OP!)

                self['gradients'] = self.gFWD[0]['gradients']

                # None @gradients check
                none_grads = 0
                for grad in self['gradients']:
                    if grad is None: none_grads += 1
                if none_grads and self.verb>0:
                    print(f'There are None gradients: {none_grads}/{len(self["gradients"])}, some trainVars may be unrelated to loss, please check the graph!')

                # average gradients
                if len(devices) > 1:

                    if self.verb>0: print(f'\nNEModel builds gradients averaging graph with device {devices[0]} for {len(self.gFWD)} towers')
                    with tf.device(devices[0]):
                        towerGrads = [tower['gradients'] for tower in self.gFWD]
                        avgGrads = []
                        for mGrads in zip(*towerGrads):
                            grads = []
                            for grad in mGrads:
                                if grad is not None: # None for variables not used while training now...
                                    expandedG = tf.expand_dims(input=grad, axis=-1)
                                    grads.append(expandedG)
                            if grads:
                                grad = tf.concat(values=grads, axis=-1)
                                grad = tf.reduce_mean(input_tensor=grad, axis=-1)
                                avgGrads.append(grad)
                            else: avgGrads.append(None)

                        self['gradients'] = avgGrads # update with averaged gradients
                        if self.verb>0: print(f' > NEModel averaged gradients ({self["gradients"][0].device})')

                # build OPT graph
                with tf.variable_scope('OPT', reuse=tf.AUTO_REUSE):

                    if self.verb>0: print(f'\nBuilding OPT graph for {self.name} model @device: {devices[0]}')
                    with tf.device(devices[0]):

                        opt_graph_return_dict = opt_func(
                            train_vars=     train_vars,
                            gradients=      self['gradients'],
                            **self.__opt_graph_dna)
                        if self.verb>0: print(f'dictionary keys returned by opt_func ({opt_func.__name__}): {opt_graph_return_dict.keys()}')

                        self.update(opt_graph_return_dict)  # update self with opt_graph_return_dict

                        saver_vars['opt_vars'] = self['opt_vars']

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(
            graph=  self.graph,
            config= config)

        # remove keys with no variables (corner case, for proper saver)
        sKeys = list(saver_vars.keys())
        for key in sKeys:
            if not saver_vars[key]: saver_vars.pop(key)
        # TODO: saver_vars, savers_names, load_saver - need a little refactor!!!
        # add saver then load
        self.__saver = MultiSaver(
            model_name= self.name,
            vars=       saver_vars,
            save_TFD=   save_TFD,
            savers=     savers_names,
            session=    self.session,
            verb=       self.verb)
        if load_saver:
            if type(load_saver) is bool: load_saver = None
            self.__saver.load(saver=load_saver)
            self.update_LR(self['iLR']) # safety update of iLR

        self.__summ_writer = tf.summary.FileWriter(
            logdir=         self.model_dir,
            #graph=          self.graph, # you can call add_graph() later
            flush_secs=     10) if self.do_TB else None

        self.model_data = None
        self.batcher = None

        if self.verb>0: print(f'{self.name} (NEModel) build finished!')
        if self.verb>2: print(self)

    # resolves func dna in proper order
    def __get_func_dna(
            self,
            func: Callable,
            nemodel_defaults: dict,
            dna_from_folder: dict,
            user_dna: dict):

        func_args = get_params(func)
        func_args_without_defaults = func_args['without_defaults']
        func_args_with_defaults = func_args['with_defaults']
        if self.verb>1:
            print(f' >> function {func.__name__} args:')
            print(f' >> without defaults: {func_args_without_defaults}')
            print(f' >> with defaults   : {func_args_with_defaults}')
            print(f' >> given:')
            print(f' >> NEModel_defaults: {nemodel_defaults}')
            print(f' >> dna_from_folder : {dna_from_folder}')
            print(f' >> user_dna        : {user_dna}')

        _dna = {}
        _dna.update(nemodel_defaults)           # 1 update with params defined by NEModel
        _dna.update(func_args_with_defaults)    # 2 update with defaults of func
        _dna.update(dna_from_folder)            # 3 update with params from folder
        _dna.update(user_dna)                   # 4 update with user params

        func_dna = {k: _dna[k] for k in _dna if k in func_args_without_defaults or k in func_args_with_defaults} # filter to get only params accepted by func
        return func_dna

    def __str__(self): return ParaDict.dict_2str(self)

    def save(self):
        assert not self.read_only, f'ERR: cannot save NEModel {self.name} while model is readonly!'
        self.__saver.save()

    # reloads model checkpoint
    def load(self):
        self.__saver.load()

    # updates base LR (iLR) in graph - but not saves it to the checkpoint
    def update_LR(self, lr):
        if 'iLR_var' not in self: print('NEModel: There is no LR variable in graph to update')
        else:
            self['iLR'] = lr
            self.session.run(tf.assign(ref=self['iLR_var'], value=lr))

    def get_summ_writer(self): return self.__summ_writer

    # logs value to TB
    def log_TB(
            self,
            value,
            tag: str,
            step: int):
        assert self.__summ_writer, f'ERR: can not log_TB since there is no summ_writer in NEModel {self.name}'
        vsumm = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.__summ_writer.add_summary(vsumm, step)

    # copies NEModel folder (dna & checkpoints)
    @staticmethod
    def copy_nemodel_FD(
            name_S: str,
            name_T: str,
            folder_S: str=  SAVE_TFD,
            folder_T: str=  SAVE_TFD):

        # copy dna
        ParaDict.copy_dnaFD(
            dna_SFD_S=  name_S,
            dna_SFD_T=  name_T,
            top_FD_S=   folder_S,
            top_FD_T=   folder_T,
            fn_pfx=     NEMODEL_DNA_PFX)

        # copy checkpoints
        nm_SFD = f'{folder_S}/{name_S}'
        ckptL = [cfd for cfd in os.listdir(nm_SFD) if os.path.isdir(os.path.join(nm_SFD, cfd))]
        if 'opt_vars' in ckptL: ckptL.remove('opt_vars')
        for ckpt in ckptL:
            mrg_ckpts(
                ckptA=          ckpt,
                ckptA_FD=       nm_SFD,
                ckptB=          None,
                ckptB_FD=       None,
                ckptM=          ckpt,
                ckptM_FD=       f'{folder_T}/{name_T}',
                replace_scope=  name_T)

    # GX for two NEModels
    @staticmethod
    def do_GX(
            name_A: str,                    # name parent A
            name_B: str,                    # name parent B
            name_child: str,                # name child
            folder_A: str=      SAVE_TFD,
            folder_B: str=      None,
            folder_child: str=  None,
            ratio: float=       0.5,
            noise: float=       0.03):

        if not folder_B: folder_B = folder_A
        if not folder_child: folder_child = folder_A

        mfd = f'{folder_A}/{name_A}'
        ckptL = [dI for dI in os.listdir(mfd) if os.path.isdir(os.path.join(mfd,dI))]
        if 'opt_vars' in ckptL: ckptL.remove('opt_vars')

        for ckpt in ckptL:
            mrg_ckpts(
                ckptA=          ckpt,
                ckptA_FD=       f'{folder_A}/{name_A}/',
                ckptB=          ckpt,
                ckptB_FD=       f'{folder_B}/{name_B}/',
                ckptM=          ckpt,
                ckptM_FD=       f'{folder_child}/{name_child}/',
                replace_scope=  name_child,
                mrgF=           ratio,
                noiseF=         noise)

    # ************************************************************************************************ training baseline

    # loads model data for training, dict should have at least 'train':{} for Batcher
    def load_model_data(self) -> dict:
        raise NotImplementedError('NEModel.load_model_data() should be overridden!')

    # pre training method - may be overridden
    def pre_train(self):
        self.model_data = self.load_model_data()
        self.batcher = Batcher(
            data_TR=    self.model_data['train'],
            data_VL=    self.model_data['valid'] if 'valid' in self.model_data else None,
            data_TS=    self.model_data['test'] if 'test' in self.model_data else None,
            batch_size= self['batch_size'],
            btype=      'random_cov',
            verb=       self.verb)

    # builds feed dict from given batch of data
    def build_feed(self, batch: dict, train=True) -> dict:
        raise NotImplementedError('NEModel.build_feed() should be overridden!')

    # training method, saves max
    def train(
            self,
            test_freq=          100,    # number of batches between tests, model SHOULD BE tested while training
            mov_avg_factor=     0.1,
            save=               True):  # allows to save model while training

        self.pre_train()

        if self.verb>0: print(f'{self.name} - training starts')
        batch_IX = 0
        tr_lssL = []
        tr_accL = []
        ts_acc_max = 0
        ts_acc_mav = MovAvg(mov_avg_factor)

        ts_results = []
        ts_bIX = [bIX for bIX in range(self['n_batches']+1) if not bIX % test_freq] # batch indexes when test will be performed
        assert ts_bIX, 'ERR: model SHOULD BE tested while training!'
        ten_factor = int(0.1*len(ts_bIX)) # number of tests for last 10% of training
        if ten_factor < 1: ten_factor = 1 # we need at least one result
        if self.hpmser_mode: ts_bIX = ts_bIX[-ten_factor:]

        while batch_IX < self['n_batches']:
            batch_IX += 1
            batch = self.batcher.get_batch()

            feed = self.build_feed(batch)
            fetches = self['optimizer']
            if self.do_TB or self.verb>0: fetches = [self['optimizer'], self['loss'], self['acc'], self['gg_norm'], self['avt_gg_norm']]

            run_out = self.session.run(fetches, feed)

            if self.do_TB or self.verb>0:
                _, loss, acc, gg_norm, avt_gg_norm = run_out
                if self.do_TB:
                    self.log_TB(value=loss,        tag='tr/loss',    step=batch_IX)
                    self.log_TB(value=acc,         tag='tr/acc',     step=batch_IX)
                    self.log_TB(value=gg_norm,     tag='tr/gn',      step=batch_IX)
                    self.log_TB(value=avt_gg_norm, tag='tr/gn_avt',  step=batch_IX)
                tr_lssL.append(loss)
                tr_accL.append(acc)

            if batch_IX in ts_bIX:
                ts_acc, ts_loss = self.test()
                acc_mav = ts_acc_mav.upd(ts_acc)
                ts_results.append(ts_acc)
                if self.do_TB:
                    self.log_TB(value=ts_loss, tag='ts/loss',    step=batch_IX)
                    self.log_TB(value=ts_acc,  tag='ts/acc',     step=batch_IX)
                    self.log_TB(value=acc_mav, tag='ts/acc_mav', step=batch_IX)
                if self.verb>0: print(f'{batch_IX:5d} TR: {100*sum(tr_accL)/test_freq:.1f} / {sum(tr_lssL)/test_freq:.3f} -- TS: {100*ts_acc:.1f} / {ts_loss:.3f}')
                tr_lssL = []
                tr_accL = []

                if ts_acc > ts_acc_max:
                    ts_acc_max = ts_acc
                    if not self.read_only and save: self.save() # model is saved for max_ts_acc

        # weighted test value for last 10% test results
        ts_results = ts_results[-ten_factor:]
        ts_wval = 0
        weight = 1
        sum_weight = 0
        for tr in ts_results:
            ts_wval += tr*weight
            sum_weight += weight
            weight += 1
        ts_wval /= sum_weight
        if self.do_TB: self.log_TB(value=ts_wval, tag='ts/ts_wval', step=batch_IX)
        if self.verb>0:
            print(f'model {self.name} finished training')
            print(f' > test_acc_max: {ts_acc_max:.4f}')
            print(f' > test_wval:    {ts_wval:.4f}')

        return ts_wval

    def test(self):
        batches = self.batcher.get_TS_batches()
        acc_loss = []
        acc_acc = []
        for batch in batches:
            feed = self.build_feed(batch, train=False)
            fetches = [self['loss'], self['acc']]
            loss, acc = self.session.run(fetches, feed)
            acc_loss.append(loss)
            acc_acc.append(acc)
        return sum(acc_acc)/len(acc_acc), sum(acc_loss)/len(acc_loss)