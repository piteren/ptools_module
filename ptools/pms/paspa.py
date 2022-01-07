"""

 2018 (c) piteren

    PaSpa - Parameters Space

        - each axis of space has:
            - type (list - continuous, tuple - discrete)
            - range (width, which usually will be normalized to 1)
        - non numeric parameters are handled with distance calculations
        PaSpa is a metric space (https://en.wikipedia.org/wiki/Metric_space)
        PaSpa supports L1 and L2 distance calculations, normalized to 1 (1 is the longest diagonal of space)

        PaSpa consists points (in space - self) {axis: value}

        PaSpa has:
            - dim  - dimensionality (= number of axes)
            - rdim - reduced dimensionality (rdim<=dim - since some axes are simpler(tuples, lists of ints))

    PaSpa is build from PSDD (dict):
        PSDD - {axis(parameter name): list or tuple or value}
            > list of ints or floats defines continuous range
            > tuple may contain elements of any type (even non-numeric), may have one
            > any other type is considered to be a constant (single value)

            example:
            {   'a':    [0.0, 1],               # range of floats
                'b':    (-1,-7,10,15.5,90,30),  # set of num(float) values, num will be sorted
                'c':    ('tat','mam','kot'),    # set of diff values
                'd':    [0,10],                 # range of ints
                'e':    (-2.0,2,None)}          # set of diff values
                'f':    (16.2,)}                # single value

"""

import math
import random
from typing import Any, Dict, List, Tuple

from ptools.lipytools.little_methods import float_to_str


P_VAL = float or int or Any                             # point value (..or Any makes no sense)

POINT = Dict[str, P_VAL]                                # point in th PaSpa {axis: value}

PSDD  = Dict[str, List[P_VAL] or Tuple[P_VAL] or P_VAL] # PaSpa Definition Dictionary


# parameters space
class PaSpa:

    def __init__(
            self,
            psdd :PSDD,             # params space dictionary (lists, tuples, vals)
            distance_L2=    True,   # defines L1 or L2 distance for PaSpa
            verb=           0):

        self.__psdd = psdd
        self.axes = sorted(list(self.__psdd.keys()))
        self.L2 = distance_L2
        self.dim = len(self.__psdd)
        if verb > 0:  print(f'\n*** PaSpa *** (dim: {self.dim}) inits')

        # resolve axis type and width and some safety checks
        self.__axT = {} # axis type, (list,tuple)_(float,int,diff)
        self.__axW = {} # axis width
        for axis in self.axes:

            if not type(self.__psdd[axis]) in (list,tuple): self.__psdd[axis] = (self.__psdd[axis],)
            pdef = self.__psdd[axis]
            tp = 'tuple' if type(pdef) is tuple else 'list'
            if tp == 'list' and len(pdef) != 2:
                assert False, f'ERR: parameter definition with list should have two elements (in list): >{pdef}<!'

            tpn = 'int' # default
            are_flt = False
            are_dif = False
            for el in pdef:
                if type(el) is float: are_flt = True
                if type(el) is not int and type(el) is not float: are_dif = True
            if are_flt: tpn = 'float'  # downgrade
            if are_dif: tpn = 'diff'   # downgrade

            assert not (tp=='list' and tpn=='diff')

            # sort numeric
            if tpn!='diff':
                pdef = sorted(list(pdef))
                self.__psdd[axis] = pdef if tp == 'list' else tuple(pdef) # update type of sorted

            self.__axT[axis] = f'{tp}_{tpn}' # string like 'list_int'
            self.__axW[axis] = pdef[-1] - pdef[0] if tpn != 'diff' else len(pdef) - 1

        self.rdim = self.__rdim()
        if verb > 0:  print(f' > PaSpa rdim: {self.rdim:.1f}')

        # width of str(value) for axes, used for str formatting
        self.__str_width = {}
        for axis in self.axes:
            width = max([len(str(e)) for e in self.__psdd[axis]])
            if 'list_float' in self.__axT[axis]: width = 7
            self.__str_width[axis] = width

    # calculate reduced dimensionality of PaSpa, rdim = log10(∏ axd) where axd=10 for list of floats
    def __rdim(self):
        """
        rdim = log10(∏ sq if sq<10 else 10) for all axes
            sq = 10 for list of floats (axis)
            sq = sqrt(len(axis_elements)) for tuple or list of int
        """
        axd = []
        for axis in self.axes:
            axt = self.__axT[axis]
            if 'val' in axt: continue
            if 'list' in axt: sq = 10 if 'float' in axt else math.sqrt(self.__axW[axis])
            else:             sq = math.sqrt(len(self.__psdd[axis]))
            axd.append(sq if sq<10 else 10)
        mul = 1
        for e in axd: mul *= e
        return math.log10(mul)

    # checks if given value belongs to an axis of space
    def __is_in_axis(self, value: P_VAL, axis: str):
        if axis not in self.__axT:                                      return False # axis not in a space
        if 'list' in self.__axT[axis]:
            if type(value) is float and 'int' in self.__axT[axis]:      return False # type mismatch
            if self.__psdd[axis][0] <= value <= self.__psdd[axis][1]:   return True
            else:                                                       return False # value not in a range
        elif value not in self.__psdd[axis]:                            return False # value not in a tuple
        return True

    # checks if point belongs to a space
    def __is_in_space(self, point: POINT):
        for axis in point:
            if not self.__is_in_axis(value=point[axis], axis=axis): return False
        return True

    # gets random value for axis ...algorithm is a bit complicated, but ensures equal probability for both sides of ref_val
    def __random_value(
            self,
            axis: str,
            ref_val: P_VAL=     None,   # reference value on axis
            ax_dst: float=      None,   # RELATIVE distance on axis from ref_val (on both sides) to sample
            allow_full_tuple=   True):  # for tuple axis type allows to sample from whole axis (forget ref_val and ax_dst)

        axT = self.__axT[axis]
        axW = self.__axW[axis]
        psd = self.__psdd[axis]

        if 'list' in axT:
            a = psd[0]
            b = psd[1]
            if ref_val is not None:
                a = ref_val
                dist = ax_dst * axW
                if 'int' in axT: dist = int(dist)
                if random.random() < 0.5: # left side
                    b = psd[0]
                    if b < ref_val - dist: b = ref_val - dist
                else:
                    b = psd[1]
                    if b > ref_val + dist: b = ref_val + dist
            val = random.uniform(a,b)
            if 'int' in axT: val = int(round(val))

        else:
            if ref_val is None or allow_full_tuple: choice_vals = psd # whole axis
            else:
                # select vals from tuple in range
                sub_vals_L = [] # left
                sub_vals_R = [] # right
                for e in psd:
                    un_ax_dst = ax_dst * axW
                    if 'diff' not in axT:
                        if ref_val - un_ax_dst <= e <= ref_val + un_ax_dst: # in range
                            if e < ref_val: sub_vals_L.append(e)
                            else:           sub_vals_R.append(e)
                    else:
                        psdL = list(psd)
                        eIX = psdL.index(e)
                        rIX = psdL.index(ref_val)
                        if rIX - un_ax_dst <= eIX <= rIX + un_ax_dst:
                            if eIX < rIX: sub_vals_L.append(e)
                            else:         sub_vals_R.append(e)

                # same quantity for both sides (..probability(0.5))
                sh_vals = sub_vals_L
                lg_vals = sub_vals_R
                # swap
                if len(sub_vals_L) > len(sub_vals_R):
                    sh_vals = sub_vals_R
                    lg_vals = sub_vals_L
                if sh_vals:
                    while len(sh_vals) < len(lg_vals):
                        sh_vals.append(random.choice(sh_vals))
                choice_vals = sh_vals + lg_vals
                assert choice_vals

            val = random.choice(choice_vals)

        assert self.__is_in_axis(value=val, axis=axis)
        return val

    # distance between two points in space (normalized to 1 - divided by max space distance)
    def distance(self, pa: POINT, pb: POINT) -> float:

        if self.L2:
            dist_pow_sum = 0
            for axis in pa:
                if self.__axW[axis] > 0:
                    dist = self.__psdd[axis].index(pa[axis]) - self.__psdd[axis].index(pb[axis]) \
                        if 'diff' in self.__axT[axis] else \
                        pa[axis] - pb[axis]
                    dist_pow_sum += (dist / self.__axW[axis])**2
            return  math.sqrt(dist_pow_sum) / math.sqrt(self.dim)
        else:
            dist_abs_sum = 0
            for axis in pa:
                if self.__axW[axis] > 0:
                    dist = self.__psdd[axis].index(pa[axis]) - self.__psdd[axis].index(pb[axis]) \
                        if 'diff' in self.__axT[axis] else \
                        pa[axis] - pb[axis]
                    dist_abs_sum += abs(dist) / self.__axW[axis]
            return dist_abs_sum / self.dim

    # samples (random) point from whole space or from the surroundings of ref_point
    def sample_point(
            self,
            ref_point :POINT=   None,           # reference point
            ax_dst: float=      None,           # relative distance to sample (on axes from ref_point, ..both sides)
            allow_full_tuple=   True) -> POINT: # for tuple_axis_type allows forget ref_point

        if ref_point is None: ref_point = {axis: None for axis in self.axes}
        return {axis: self.__random_value(
            axis=               axis,
            ref_val=            ref_point[axis],
            ax_dst=             ax_dst,
            allow_full_tuple=   allow_full_tuple) for axis in self.axes}

    # samples GeneX point from given two or one
    def sample_GX_point(
            self,
            pa: POINT,
            pb: POINT=          None,
            ax_dst: float=      None,
            full_space_prob=    0.05, # probability of sampling new value for axis from whole paspa
            allow_full_tuple=   True) -> POINT:

        if not pb: sp = pa
        else:
            sp = {}
            for axis in self.axes:
                # mix
                if random.random() < 0.5 and 'list' in self.__axT:
                    val = pa[axis]+pb[axis] / 2
                    if 'int' in self.__axT: val = int(round(val))
                # one of two
                else:
                    val = pa[axis] if random.random() < 0.5 else pb[axis]
                sp[axis] = val

        # replace axis values by full space random values
        p_full_space = self.sample_point()
        for axis in self.axes:
            if random.random() < full_space_prob:
                sp[axis] = p_full_space[axis]

        return self.sample_point(
            ref_point=          sp,
            ax_dst=             ax_dst,
            allow_full_tuple=   allow_full_tuple)

    # samples 2 points distanced with 1 (opposite corner points)
    def sample_corners(self) -> (POINT,POINT):
        pa = {}
        pb = {}
        axes = list(self.__psdd.keys())
        left = [0 if random.random()>0.5 else 1 for _ in range(self.dim)] # left/right
        for aIX in range(len(axes)):
            ax = axes[aIX]
            vl = self.__psdd[ax][0]
            vr = self.__psdd[ax][-1]
            pa[ax] = vl
            pb[ax] = vr
            if left[aIX]:
                pa[ax] = vr
                pb[ax] = vl
        return pa, pb

    # point -> nicely formatted string
    def point_2str(self, p :POINT) -> str:
        s = '{'
        for axis in sorted(list(p.keys())):
            val = p[axis]
            vs = float_to_str(val) if 'list_float' in self.__axT[axis] else str(val)
            #if len(vs) > self.__str_width[axis]: vs = vs[:self.__str_width[axis]]
            s += f'{axis}:{vs:{self.__str_width[axis]}s} '
        s = s[:-1] + '}'
        return s

    # returns info(string) about self
    def __str__(self):
        info = f'*** PaSpa *** (dim: {self.dim}, rdim: {self.rdim:.1f}) parameters space:\n'
        max_ax_l = 0
        max_ps_l = 0
        for axis in self.axes:
            if len(axis)                > max_ax_l: max_ax_l = len(axis)
            if len(str(self.__psdd[axis])) > max_ps_l: max_ps_l = len(str(self.__psdd[axis]))
        if max_ax_l > 40: max_ax_l = 40
        if max_ps_l > 40: max_ps_l = 40

        for axis in self.axes:
            info += f' > {axis:{max_ax_l}s}  {str(self.__psdd[axis]):{max_ps_l}s}  {self.__axT[axis]:11s}  width: {self.__axW[axis]}\n'
        return info[:-1]

    def __eq__(self, other):
        for k in self.__psdd:
            if k not in other.__psdd: return False
            if self.__psdd[k] != other.__psdd[k]: return False
        if self.L2 != other.L2: return False
        return True

def example_paspa1():

    dc = {

        'a':    [0.0,   1.0],
        'b':    [-5.0,  5],
        'c':    [0,     10],
        'd':    [-10,   -5],
        'e':    (-1,-7,10,15.5,90,30),
        'f':    (1,8),
        'g':    (-11,-2,-3,4,5,8,9),
        'h':    (True, False, None),
        'i':    ('tat', 'mam', 'kot'),
        'j':    (0, 1, 2, 3, None)}

    paspa = PaSpa(dc)

    print(f'\n{paspa}')

    print(f'\n### Corners of space:')
    pa, pb = paspa.sample_corners()
    print(paspa.point_2str(pa))
    print(paspa.point_2str(pb))
    print(f'distance: {paspa.distance(pa,pb)}')

    print(f'\n### Random 100 points from space:')
    points = []
    for ix in range(100):
        point = paspa.sample_point()
        points.append(point)
        print(f'{ix:2d}: {paspa.point_2str(point)}')

    print(f'\n### 100 points from space with ref_point and ax_dst:')
    for ix in range(100):
        point_a = points[ix]
        ax_dst = random.random()
        point_b = paspa.sample_point(
            ref_point=          point_a,
            ax_dst=             ax_dst,
            allow_full_tuple=   False)
        print(f'{ix:2d}: requested axis distance {ax_dst:.3f}, resulting space distance: {paspa.distance(point_a, point_b):.3f}')
        print(f'  {paspa.point_2str(point_a)}')
        print(f'  {paspa.point_2str(point_b)}')

# samples close points
def example_paspa2():

    psd = {
        'pe_width':             [0,5],
        'pe_min_pi':            [0.05,1],
        'pe_max_pi':            [1.0,9.9],
        't_drop':               [0.0,0.1],                  #
        'f_drop':               [0.0,0.2],                  #
        'n_layers':             [15,25],                    #
        'lay_drop':             [0.0,0.2],                  #
        'ldrt_scale':           [2,6],                      #
        'ldrt_drop':            [0.0,0.5],                  #
        'drt_nlays':            [0,5],                      #
        'drt_scale':            [2,6],                      #
        'drt_drop':             [0.0,0.6],                  #
        'out_drop':             [0.0,0.5],                  #
        'learning_rate':        (1e-4,1e-3,5e-3),           #
        'weight_f':             [0.1,9.9],                  #
        'scale_f':              [1.0,6],                    #
        'warm_up':              (100,200,500,1000,2000),    #
        'ann_step':             (1,2,3),                    #
        'n_wup_off':            [1,50]}                     #

    paspa = PaSpa(psd)
    print(f'\n{paspa}')

    ref_pt = paspa.sample_point()
    print(f'\nSampled reference point:\n > {paspa.point_2str(ref_pt)}')

    ld = 1
    while ld > 0:
        nref_pt = paspa.sample_point(
            ref_point=  ref_pt,
            ax_dst=     0.1)
        ld = nref_pt['ldrt_drop']
        if ld < ref_pt['ldrt_drop']:
            ref_pt = nref_pt
            print(f' next point ldrt_drop: {ref_pt["ldrt_drop"]}')
    print(f'\nFinal point:\n > {paspa.point_2str(ref_pt)}')


if __name__ == '__main__':

    example_paspa1()
    #example_paspa2()

