# -*- coding: utf-8 -*-
"""
@author: Meng Wang, Jiheng Duan
"""
import numpy as np
from qutip import *
from qusim.PulseGen.pulse_config import PulseConfig

def merge_pulse_chan(pulse_buffer_lst: list, pulse: PulseConfig, Hd_i: list):
    if pulse.pulse_type in pulse_buffer_lst[0]:
        index_type = pulse_buffer_lst[0].index(pulse.pulse_type)
    else: 
        # print(f'{pulse.pulse_index} miss index type')
        index_type = -1
    if pulse.qindex in pulse_buffer_lst[1]:
        index_qi = pulse_buffer_lst[1].index(pulse.qindex)
    else:
        # print(f'{pulse.pulse_index} miss index qi')
        index_qi = -1
    # index_type = np.where(np.array(pulse_buffer_lst[0])==pulse.pulse_type)[0]
    # index_qi = np.where(np.array(pulse_buffer_lst[1])==str(pulse.qindex))[0]
    if index_type>=0 and index_qi>=0:
        t_index = np.intersect1d(index_type, index_qi)
        if len(t_index)>0:
            t_index = int(t_index[0])
            pulse_buffer_lst[2][t_index][1] += Hd_i[1]
        else:
            pulse_buffer_lst[0].append(pulse.pulse_type)
            pulse_buffer_lst[1].append(pulse.qindex)
            pulse_buffer_lst[2].append(Hd_i)
    else:
        pulse_buffer_lst[0].append(pulse.pulse_type)
        pulse_buffer_lst[1].append(pulse.qindex)
        pulse_buffer_lst[2].append(Hd_i)
        
    return pulse_buffer_lst



