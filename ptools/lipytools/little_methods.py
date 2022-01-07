"""

 2018 (c) piteren

    some little methods (but frequently used) for Python

"""

import csv
import inspect
import json
import os
import pickle
import random
import shutil
import string
import time
from typing import List, Callable, Any, Optional


# returns default args (with their values)
def get_defaults(function: Callable):
    arg_dict = {}
    if function:
        specs = inspect.getfullargspec(function)
        args = specs.args
        vals = specs.defaults
        if vals:
            vals = list(vals)
            args.reverse()
            vals.reverse()
            for ix in range(len(vals)):
                arg_dict[args[ix]] = vals[ix]
    return arg_dict

# prepares function parameters dictionary
def get_params(function: Callable):
    params_dict = {'without_defaults':[], 'with_defaults':{}}
    if function:
        specs = inspect.getfullargspec(function)
        params = specs.args
        if not params: params = []
        vals = specs.defaults
        if not vals: vals = ()

        while len(params) > len(vals):
            params_dict['without_defaults'].append(params.pop(0))

        params_dict['with_defaults'] = {k: v for k,v in zip(params,vals)}

    return params_dict

# short(compressed) scientific notation for floats
def short_scin(
        fl: float,
        precision:int=  1):
    sh = f'{fl:.{precision}E}'
    sh = sh.replace('+0','')
    sh = sh.replace('+','')
    sh = sh.replace('-0','-')
    sh = sh.replace('E','e')
    return sh

# returns sting from float, always of given width
def float_to_str(
        num: float,
        width: int= 7):
    if width < 5: width = 5
    scientific_decimals = width-6 if width>6 else 0
    ff = f'{num:.{scientific_decimals}E}'
    if 1000 > num > 0.0001: ff = str(num)[:width]
    if len(ff)<width: ff += '0'*(width-len(ff))
    return ff


# *********************************************************************************************** file readers / writers
# ********************************************* for raise_exception=False each reader will return None if file not found

def r_pickle( # pickle reader
        file_path,
        obj_type=           None, # if obj_type is given checks for compatibility with given type
        raise_exception=    False):
    if not os.path.isfile(file_path):
        if raise_exception: raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    obj = pickle.load(open(file_path, 'rb'))
    if obj_type: assert type(obj) is obj_type, f'ERROR: obj from file is not {str(obj_type)} type !!!'
    return obj

def w_pickle( # pickle writer
        obj,
        file_path):
    pickle.dump(obj, open(file_path, 'wb'))

def r_json( # json reader
        file_path,
        raise_exception=    False):
    if not os.path.isfile(file_path):
        if raise_exception: raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def w_json( # json writer
        data: dict,
        file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def r_jsonl( # jsonl reader
        file_path,
        raise_exception=False):
    if not os.path.isfile(file_path):
        if raise_exception: raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def w_jsonl( # jsonl writer
        data: List[dict],
        file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for d in data:
            json.dump(d, file, ensure_ascii=False)
            file.write('\n')

def r_csv( # csv reader
        file_path,
        raise_exception=    False):
    if not os.path.isfile(file_path):
        if raise_exception: raise FileNotFoundError(f'file {file_path} not exists!')
        return None
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        return [row for row in reader][1:]


# returns timestamp string
def stamp(
        year=       False,
        date=       True,
        letters=    3):
    random.seed(time.time())
    if year:        stp = time.strftime('%y%m%d.%H%M')
    else:           stp = time.strftime('%m%d.%H%M')
    if not date:    stp = ''
    if letters:
        if date:    stp += '.'
        if True:    stp += ''.join([random.choice(string.ascii_letters) for _ in range(letters)])
    return stp

# returns string representation of given list
def list_str(ls: List[Any], limit: Optional[int]=200):
    lstr = [str(e) for e in ls]
    lstr = '; '.join(lstr)
    if limit: lstr = lstr[:limit]
    return lstr

# prints nested dict
def print_nested_dict(dc: dict, ind_scale=2, line_limit=200):

    tpD = {
        dict:   'D',
        list:   'L',
        tuple:  'T',
        str:    'S'}

    def __prn_root(root: dict, ind, ind_scale=2, line_limit=line_limit):

        spacer = ' ' * ind * ind_scale
        for k in sorted(list(root.keys())):
            tp = tpD.get(type(root[k]),'O')
            ln = len(root[k]) if tp in tpD.values() else ''

            exmpl = ''
            if tp!='D':
                exmpl = str(root[k])
                if line_limit:
                    if len(exmpl)>line_limit: exmpl = f'{exmpl[:line_limit]} ..'
                exmpl = f' : {exmpl}'

            print(f'{spacer}{k} [{tp}.{ln}]{exmpl}')

            if type(root[k]) is dict: __prn_root(root[k],ind+1,ind_scale)

    __prn_root(dc,ind=0,ind_scale=ind_scale)


# prepares folder, creates or flushes
def prep_folder(
        folder_path :str, # folder path
        flush_non_empty=    False):
    if flush_non_empty and os.path.isdir(folder_path): shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

# random <0;1> probability function
def prob(p: float) -> bool:
    return random.random() < p

def test_stamp():
    print(stamp())


def test_print_nested_dict():
    dc = {
        'a0': {
            'a1': {
                'a2': ['el1','el2']
            }
        },
        'b0': ['el1','el2','el3']
    }
    print_nested_dict(dc)



if __name__ == '__main__':

    #test_stamp()
    test_print_nested_dict()