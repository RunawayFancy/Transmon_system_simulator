# -*- coding: utf-8 -*-
"""
@author: Jiheng Duan
"""

from qusim.PulseGen.pulse_config import PulseConfig
from qusim.PulseGen.simulation_option import SimulationOption
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import pi as PI
from collections import defaultdict as ddict

def plot_pulse_sequence(pseq: list[PulseConfig], sim_opts: SimulationOption):
    
    tlist = sim_opts.tlist    
    channel_dic = ddict(list)
    q_index_list = []

    for pulse in pseq:
        waveform_y = pulse.get_pulse(sim_opts)/2/PI
        if pulse.qindex not in q_index_list:
            q_index_list.append(pulse.qindex)
        
        # Rescale the pulses by factor 1/1.2
        # for i in range(len(t_list)):
        #     if waveform_y[i] != None and pulse_amp != 0:
        #         waveform_y[i] /= np.abs(pulse_amp)

        channel_name = f"{pulse.pulse_type}{pulse.qindex}"
        channel_dic[channel_name].append(waveform_y)

    # print(type(channel_dic['XY0']))

    # Define a custom sorting function to sort the channel names as desired
    def custom_sort(channel_name):
        channel_type = channel_name[:-1]
        channel_index = channel_name[-1]
        if channel_index.isdigit():
            return int(channel_index), channel_type
        else:
            return float('inf'), channel_type

    # Sort the channel names based on the custom sorting function
    sorted_channels = sorted(channel_dic.keys(), key=custom_sort)

    # Convert the waveform data to NumPy arrays
    channel_dic = {k: np.array(v) for k, v in channel_dic.items()}

    # Create the figure and axes
    fig, ax1 = plt.subplots()
    fig.canvas.manager.set_window_title('pulse_sequence')
    # Set the vertical spacing between channel lines
    vertical_spacing = 2

    # Iterate over the channels and plot them vertically separated
    for i, channel_name in enumerate(sorted_channels):
        index = 1
        for waveform in channel_dic[channel_name]:
            vertical_offset = i * vertical_spacing
            # print(np.max(waveform))
            waveform_offset = waveform + vertical_offset
            for _j, _off in enumerate(waveform_offset):
                if _off == vertical_offset: 
                    waveform_offset[_j] = None
            
            ax1.plot(tlist, waveform_offset, label=f'{channel_name}{index}')
            index += 1

    # Set the y-axis tick labels and limits for channel plot
    # print(len(channel_dic))
    ax1.set_yticks(np.arange(len(channel_dic)) * vertical_spacing)
    ax1.set_yticklabels(sorted_channels)
    ax1.set_ylim(-vertical_spacing/2, len(channel_dic) * vertical_spacing)
    ax1.set_xlim(0, sim_opts.simulation_time)
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Channel name")
    # Create a twin axes for the pulse amplitude
    ax2 = ax1.twinx()
    ax2.set_ylim(-vertical_spacing/2, len(channel_dic) * vertical_spacing)
    ax2.set_ylabel("Pulse Amplitude (a.u.)")
    plt.grid()
    plt.show()

def plot_population_evolution(_system, result_list, sim_opts: SimulationOption, interested_state, interested_state_label, initial_state_label):
    num_subplots = len(result_list)
    # Calculate the number of rows and columns for the subplot layout
    num_rows = int(num_subplots ** 0.5)
    num_cols = (num_subplots + num_rows - 1) // num_rows
    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.canvas.manager.set_window_title('Pop. vs. time')
    # Flatten the axs array if it's 1D (for the case when there's only one subplot)
    time_list = sim_opts.tlist

    if num_subplots in [2,3]:
        axs = np.array([axs])
    if num_subplots == 1:
        axs = np.array([[axs]])
    for index, result in enumerate(result_list):
        row_idx = index // num_cols
        col_idx = index % num_cols
        label_list = interested_state_label[index]
        data_list = _system.get_data_list(result, sim_opts, interested_state[index])
        for jndex in range(len(data_list)):
            axs[row_idx, col_idx].plot(time_list, data_list[jndex], label = label_list[jndex])
            axs[row_idx, col_idx].set_xlabel("Time (ns)")
            axs[row_idx, col_idx].set_ylabel("Population")
            axs[row_idx, col_idx].legend()
        axs[row_idx, col_idx].set_title(f"Initial state: {initial_state_label[index]}")
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()
    

def plot_zz_sweep(x_list:list, y_list:list, zz_list:list, x_label:str, y_label:str, fn:str=None):
    nrm1 = matplotlib.colors.LogNorm()  
    fig1, ax1 = plt.subplots()
    p1 = ax1.pcolor(x_list, y_list, zz_list, cmap=matplotlib.cm.viridis_r,norm=nrm1)
    cb1 = fig1.colorbar(p1, ax=ax1,label='ZZ', extend='both')
    ax1.set_xlabel('${}$'.format(x_label))
    ax1.set_ylabel('${}$'.format(y_label))
    if fn:
        plt.title(fn)
    else:
        plt.title('ZZ')
    plt.show()


def plot_Elevel_dynamics(w_scan_space, energy_level_list, num_to_plot, xlabel:str, xrange = [], yrange = [], legend=[]):
    if isinstance(num_to_plot, int):
        plot_list = range(num_to_plot)
    if isinstance(num_to_plot, list):
        if len(num_to_plot) ==2:
            if num_to_plot[1] - num_to_plot[0] != 1:
                plot_list = range(num_to_plot[0], num_to_plot[1])
            else:
                plot_list = num_to_plot
        else:
            plot_list = num_to_plot
    label_list,label_list_trg = get_label_list(num_to_plot, legend)
    for ii,j in enumerate(plot_list):
        plt.plot(w_scan_space, [v[j] for v in energy_level_list], label=f'{label_list[ii]}')
    if len(xrange) > 1: plt.xlim(xrange)
    if len(yrange) > 1: plt.ylim(yrange)
    plt.xlabel('${}$/GHz'.format(xlabel))
    plt.ylabel('Energy (GHz)')
    plt.title("Energy level")
    if label_list_trg:
        plt.legend()
    plt.show()

def get_label_list(num_to_plot, legend):
    
    label_list_trg = True
    if isinstance(num_to_plot, int):
        label_list = range(num_to_plot)
    if isinstance(num_to_plot, list):
        if len(legend)==len(num_to_plot):
            label_list = legend
        else:
            print('Miss matching between num_to_plot and lengend')
            num_to_plot = range(len(num_to_plot))
            label_list = [i for i in range(num_to_plot)]
    if len(legend)==0:
        label_list_trg = False
    return label_list,label_list_trg