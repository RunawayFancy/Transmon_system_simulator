# -*- coding: utf-8 -*-
"""
@author: Pan Shi, Meng Wang, Jiheng Duan
"""
from typing import List, Optional, Tuple, Union
import numpy as np
import copy
import qutip_jax
import jax
import jax.numpy as jnp
from qutip import *
from qusim.PulseGen.pulse_config import PulseConfig
from qusim.PulseGen.pulse_buffer import merge_pulse_chan
from qusim.PulseGen.simulation_option import SimulationOption
import qusim.Instruments.tools as tool
from qusim.Instruments.angle import get_angle
from qusim.PulseGen.noise_config import * 
from qusim.PulseGen.noise_gen import noise_gen


class TransmonSys:
    """
    A class of a multiple qubit (transmon) interacting system
    N: int
        Dimension of the Hilbert space of every qubit (number of energy level considered in the simulation)

    w: array like
        One dimensional array, encoding frequency of each qubit

    alpha: array like
        One dimensional array, encoding anharmonicity of each qubit
        For qubit, let alpha < 0
        For resonator, let alpha = 0
    
    r: array like
        Two dimensional array, encoding
        the coupling strength between each
        two qubit

    q_dim: array like
        One dimensional list, defined by
        `[qubit_dim for _ in range(len(w))]`
    
    gamma_list: array like
        One dimensional list, containing multiple dictionaries
        with numbers equal to the number of qubits.

    g_freq = False
        Frequency dependency of coupling between qubits. Default is False.
    """

    def __init__(self, N: int|None, q_dim: List[float], w: List[float], alpha: List[float], r=0, gamma_list = None, g_freq = False):
        self.w = w
        self.num_q = len(self.w)  # Number of qubits
        self.q_dim = q_dim # Qubit dimension
        if N != None: self.N = N # Turn on max photon num
        else: self.N = self.num_q * max(self.q_dim) + 1 # Turn off max photon num
        self.alpha = alpha
        if self.num_q > 1: self.r = r
        else: self.r = 0
        self.g = self.r
        self.g_freq = g_freq
        self.a_list = self.get_a_list()  # Define the second quantization field operator
        self.a_dagger_list = [a.dag() for a in self.a_list]
        self.H_q, self.H_a = self.get_Hq_Ha()
        self.H_inter = self.get_H_inter()
        self.H = 2 * np.pi * (self.H_q + self.H_a + self.H_inter)
        self.gamma_list = gamma_list
        self.state_dic = enr_state_dictionaries(self.q_dim, self.N)
        self.qchannel = self.get_qchannel()
        self.pulse_type_mapping={
            "XY": self.get_H_XY_drive,
            "Z": self.get_H_Z_bias,
        }
        

    def get_qchannel(self):
        """
        Channel name: tuple
            ("XY", 1)
            ("Z", 1)
        """ 
        qchannel_single = [gate for _qidx in range(self.num_q) for gate in [("XY", _qidx), ("Z", _qidx)]]
        qchannel = qchannel_single
        
        return qchannel


    def get_a_list(self):
        a_list = enr_destroy(self.q_dim, excitations=self.N)
        row_shape, col_shape = a_list[0].shape[0], a_list[0].shape[1]
        for a in a_list:
            a.dims = [[row_shape], [col_shape]]
        return a_list


    def get_Hq_Ha(self):
        """
        Calculate the qubit and anharmonicity Hamiltonian
        H_q: qubit Hamiltonian
        H_a: anharmonicity

        _q: variable for H_q
        _a: variable for anharmonicity
        """
        # Define the qubit and anharmo Hamiltonian
        H_q, H_a = 0, 0
        for q_index in range(self.num_q):
            H_q += self.w[q_index] * self.a_dagger_list[q_index] * self.a_list[q_index]
            H_a += self.alpha[q_index]/2 * self.a_dagger_list[q_index] * self.a_dagger_list[q_index] * self.a_list[q_index] * self.a_list[q_index]
        
        return H_q, H_a


    def get_H_inter(self):
        H_inter = 0
        if self.g == 0: return 0
        if self.num_q > 1:
            for q_index1 in range(self.num_q - 1):
                for q_index2 in range(q_index1 + 1, self.num_q): 
                    if self.g_freq:
                        H_inter += self.g[q_index1][q_index2] * np.sqrt(self.w[q_index1] * self.w[q_index2]) * (self.a_list[q_index1] + self.a_dagger_list[q_index1]) * (self.a_list[q_index2] + self.a_dagger_list[q_index2])
                        # print('1')
                    else:
                        H_inter += self.g[q_index1][q_index2] * (self.a_list[q_index1] + self.a_dagger_list[q_index1]) * (self.a_list[q_index2] + self.a_dagger_list[q_index2])
                        opr = (self.a_list[q_index1] + self.a_dagger_list[q_index1]) * (self.a_list[q_index2] + self.a_dagger_list[q_index2])
                        # print(self.g[q_index1][q_index2])
                        # print(opr.check_herm())
        return H_inter


    def get_state_index(self, n, freq_threshold = 1e-6, deg_threshold = 5e-3, deg_round = 7):
        '''
        n: tuple
            e.g., (0,0,0), (1,0,1)
        
        freq_threshold: float, double
            The threshold of qubit frequency difference that 
            is recognized as degeneracies happening
        
        deg_threshold: float, double
            The threshold of probability amplitude that between
            superposition states constructed by energy
            degenerated states
        
        deg_round: int
            The number of decimal number that will be rounded in 
            estimating the probability amplitude of each degenerated 
            state.
        '''
        state_index = self.state_dic[1][n]
        eigen_list = [np.abs(arr[state_index][0]) for arr in self.H.eigenstates()[1]]

        max_value = max(eigen_list)
        max_index = np.argmax(np.array(eigen_list))
        # print(len(eigen_list))
        # print('eigen_list = {}'.format(eigen_list))
        # print('max_value = {}',format(max_value))
        # print('max_index = {}'.format(max_index))

        sim_index = tool.find_similar_indices(np.array(self.w), freq_threshold)
        if len(sim_index) > 0: 
            degenerate_index = []
            for index, value in enumerate(eigen_list):
                if index == max_index: continue
                if np.abs(value - max_value) < deg_threshold: # Not sufficient to say degeneracy is appears
                    prob_amp_list = []

                    # print(value)
                    # Exam the state 
                    for row in self.H.eigenstates()[1][index]:
                        prob_amp_list.append(np.round(np.abs(row[0]), deg_round))

                    # Count the number of equal array elements
                    deg_prob_amp_list, num_degen_list = np.unique(np.array(prob_amp_list), return_counts=True)
                    
                    # print(deg_prob_amp_list)
                    # print(num_degen_list)
                    # Extracting the maximum degenerate
                    i_max = np.argmax(deg_prob_amp_list)
                    num_degen = num_degen_list[i_max]
                    deg_prob_amp = deg_prob_amp_list[i_max]
                    
                    # print(deg_prob_amp)
                    # print(num_degen)

                    if num_degen > 1:
                        degenerate_index.append(index)

            # print(degenerate_index)
            if len(degenerate_index) > 0:
                degenerate_index.append(max_index)
                deg_index_arr = np.sort(np.array(degenerate_index))

                # Effective excitation number
                num_excit = 0
                for wi in sim_index:
                    num_excit += n[wi]

                count_n = 0
                for ii,wi in enumerate(sim_index):
                    if n[wi] != 0:
                        count_n += 1 * 2**(ii)
                max_index = deg_index_arr[count_n-1]
        # print(max_index)

        return max_index


    def get_eigenstates_energy(self, n, freq_threshold = 1e-6, deg_threshold = 5e-3, deg_round = 7):
        '''
        n: tuple
            e.g., (0,0,0), (1,0,1)
        
        freq_threshold: float, double
            The threshold of qubit frequency difference that 
            is recognized as degeneracies happening
        
        deg_threshold: float, double
            The threshold of probability amplitude that between
            superposition states constructed by energy
            degenerated states
        
        deg_round: int
            The number of decimal number that will be rounded in 
            estimating the probability amplitude of each degenerated 
            state.
        '''
        state_index = self.get_state_index(n, freq_threshold = 1e-6, deg_threshold = 5e-3, deg_round = 7)
        
        # state_index = self.state_dic[1][n]
        eigen_val_state = self.H.eigenstates()
        eigenstates, eigenenergies = eigen_val_state[1][state_index], eigen_val_state[0][state_index]/2/np.pi
        
        # Return a qobj eigenstate, energy level magnitude, and the index of the energy  level
        return eigenstates, eigenenergies.real, state_index
    

    def co_list(self):
        co_list = []
        if self.gamma_list is None:
            return co_list
        for q_index in range(self.num_q):
            # Get collapse up operator
            gamma_up = self.gamma_list[q_index].get("up", 0)
            gamma_down = self.gamma_list[q_index].get("down", 0)
            gamma_z = self.gamma_list[q_index].get("z", 0)
            if gamma_up != 0:
                co_list.append(np.sqrt(self.gamma_list[q_index]["up"]) * self.a_dagger_list[q_index])
            # Get collapse down operator
            if gamma_down != 0:
                co_list.append(np.sqrt(self.gamma_list[q_index]["down"]) * self.a_list[q_index])
            # Get collapse z operator
            # Question marks: L_z = sqrt(2 Gamma_Z) a^dagger a
            if gamma_z != 0:
                co_list.append(np.sqrt(self.gamma_list[q_index]["z"] / 2) * self.a_dagger_list[q_index] * self.a_list[q_index])

        return co_list
    

    def get_H_d(self, 
            pseq: List[PulseConfig],
            sim_opts: SimulationOption,
            channel_noise: Optional[List[Tuple[tuple, List[Union[GaussianNoiseConfig, RandomTeleNoiseConfig, JNNoiseConfig, OneOverFNoiseConfig]]]]] = None,
            jax=False
        ):

        H_d = []
        if jax:
            H_0 = self.H.to('jax')
            print(H_0.dtype)
        else:
            H_0 = self.H
        H_d.append([H_0, jnp.array(sim_opts.tlist)])

        pulse_buffer_lst = [[] for _ in range(3)]
        for pulse in pseq:
            pulse_buffer_lst = merge_pulse_chan(pulse_buffer_lst, pulse, self.send_pulse(pulse, sim_opts))

        pulse_buffer_lst = self.add_channel_noise(pulse_buffer_lst, sim_opts, channel_noise)

        for Hd_i in pulse_buffer_lst[2]:
            if jax:
                Hd_i[0] = Hd_i[0].to('jax')
                Hd_i[1] = jnp.array(Hd_i[1])
                print(Hd_i[0].dtype)
            H_d.append(Hd_i)

        return H_d
    
    def get_Hd_channel(self, pulse_buffer_lst:list):
        Hd_channel_list = []
        for _i in range(len(pulse_buffer_lst[0])):
            chan_name = (pulse_buffer_lst[0][_i], pulse_buffer_lst[1][_i])
            Hd_channel_list.append(chan_name)

        return Hd_channel_list
    
    def add_channel_noise(self,
            pulse_buffer_lst: list,
            sim_opts: SimulationOption,
            channel_noise: Optional[List[Tuple[tuple, List[Union[GaussianNoiseConfig, RandomTeleNoiseConfig, JNNoiseConfig, OneOverFNoiseConfig]]]]] = None
        ):
        if not channel_noise:
            return pulse_buffer_lst
        for chan_name, noise_config_list in channel_noise:
            Hd_channel_list = self.get_Hd_channel(pulse_buffer_lst)
            assert chan_name in self.qchannel
            if chan_name in Hd_channel_list:
                chan_idx = Hd_channel_list.index(chan_name)

                waveform = copy.deepcopy(pulse_buffer_lst[2][chan_idx][1])
                for noise_config in noise_config_list:
                    if noise_config.methods == 'sum':
                        pulse_buffer_lst[2][chan_idx][1] += np.real(noise_gen(noise_config, waveform))
                    elif noise_config.methods == 'multiply':
                        pulse_buffer_lst[2][chan_idx][1] *= np.real(noise_gen(noise_config, waveform))
                    else:
                        raise AttributeError('Missing attributes methods in noise config.')
            else:
                opeartor = self.pulse_type_mapping[chan_name[0]](chan_name[1])
                
                # if isinstance(chan_name[1], int): # for XY and Z pulse
                #     qeye_list = self.qeye_list.copy()
                #     qeye_list[chan_name[1]] = opeartor_single
                #     opeartor = cal_tensor(qeye_list)
                # else:
                #     opeartor  =  opeartor_single # for INT pulse

                y = np.zeros_like(sim_opts.tlist, dtype="float64")
                waveform = copy.deepcopy(y)
                for noise_config in noise_config_list:
                    if noise_config.methods == 'sum':
                        y += np.real(noise_gen(noise_config, waveform))
                    elif noise_config.methods == 'multiply':
                        y *= np.real(noise_gen(noise_config, waveform))
                    else:
                        raise AttributeError('Missing attributes methods in noise config.')
                pulse_buffer_lst[0].append(chan_name[0])
                pulse_buffer_lst[1].append(chan_name[1])
                pulse_buffer_lst[2].append([opeartor, y])

        return pulse_buffer_lst


    def system_dynamics_mesolve(self, 
            pseq: List[PulseConfig], 
            sim_opts: SimulationOption, 
            channel_noise: Optional[List[Tuple[tuple, List[Union[GaussianNoiseConfig, RandomTeleNoiseConfig, JNNoiseConfig, OneOverFNoiseConfig]]]]] = None,
            option: Options = Options(rtol=1e-8),
            jax=False
        ):
        """
        A method to convert your defined system into the master equation solver in qutip.

        jax bool
            If True, the system will be converted into jax format.
        
        """
        state_list = sim_opts.initial_state
        result_list, angle_list = [], []
        H_d = self.get_H_d(pseq, sim_opts, channel_noise, jax)
        for state in state_list:
            # H_d = []; pulse_buffer_list = [[] for _ in range(3)]
            # H_d.append(self.H)
            # for pulse in pseq:
            #     pulse_buffer_list = merge_pulse_chan(pulse_buffer_list, pulse, self.send_pulse(pulse, sim_opts))
            # for Hd_i in pulse_buffer_list[2]:
            #     H_d.append(Hd_i)

            # print(initial_state)
            # Set up master equation solver
            result, angle = self.master_eq_solver(H_d, sim_opts.tlist, copy.deepcopy(state), option, jax)
            result_list.append(result)
            angle_list.append(angle)
            
        return result_list, angle_list
    

    def master_eq_solver(self, H_d, tlist: np.ndarray, initial_state: List[Qobj], option = Options(rtol=1e-8), jax=False):
        if jax:
            initial_state = initial_state.to('jax')
            print(initial_state.dtype)
            c_ops = [c_op.to('jax') for c_op in self.co_list()]
            tlist = jnp.array(tlist)

            print('==============')
            for Hdi in H_d:
                if isinstance(Hdi, list):
                    print(Hdi[0].dtype)
                    print(isinstance(Hdi[1], jnp.ndarray))
                else:
                    print(Hdi.dtype)
            print(isinstance(tlist, jnp.ndarray))
            print(initial_state.dtype)

            
        else:
            c_ops = self.co_list()
        result = mesolve(H_d, initial_state, tlist, c_ops = c_ops, options = option) 
        angle = get_angle(initial_state, result)
        
        return result, angle
    

    def system_dynamics_propagator(self, 
            pseq: List[PulseConfig], 
            sim_opts: SimulationOption,
            channel_noise: Optional[List[Tuple[tuple, List[Union[GaussianNoiseConfig, RandomTeleNoiseConfig, JNNoiseConfig, OneOverFNoiseConfig]]]]] = None,
            option: Options = Options(rtol=1e-8), 
            do_parallel = True, 
            do_progress_bar=None
        ):
        # H_d = []; pulse_buffer_list = [[] for _ in range(3)]
        # H_d.append(self.H)
        # for pulse in pseq:
        #     pulse_buffer_list = merge_pulse_chan(pulse_buffer_list, pulse, self.send_pulse(pulse, sim_opts))
        # for Hd_i in pulse_buffer_list[2]:
        #     H_d.append(Hd_i)
        H_d = self.get_H_d(pseq, sim_opts, channel_noise)
        result = propagator(H_d, sim_opts.tlist, self.co_list(), {} , option, progress_bar=True)
        
        return result
    

    def send_pulse(self, pulse: PulseConfig, sim_opts: SimulationOption):
        """
        Construct dynamical component of the Hamiltonian H_d
        """
        assert pulse.qindex <= self.num_q -1
        
        if pulse.pulse_type == "XY":
            H_drive = self.H_XY_drive(pulse, sim_opts)
        elif pulse.pulse_type == "Z":
            H_drive = self.H_Z_bias(pulse, sim_opts)
        else:
            raise ValueError(f"Invalid pulse type: pulse index {pulse.pulse_index}, qubit index {pulse.qindex}, pulse type {pulse.pulse_type}")

        return H_drive
    

    def get_H_XY_drive(self, qindex: int):
        """
        Define the Hamiltonian of the system under XY pulse driving
        """
        return -1j*self.a_dagger_list[qindex] + 1j*self.a_list[qindex]
    

    def H_XY_drive(self, pulse: PulseConfig, sim_opts: SimulationOption):
        """
        Define the Hamiltonian of the system under XY pulse driving
        """
        # Get pulse
        XY_pulse = pulse.get_pulse(sim_opts)

        return [self.get_H_XY_drive(pulse.qindex), XY_pulse]


    def get_H_Z_bias(self, qindex: int):
        """
        Define the Hamiltonian of the system under XY pulse driving
        """
        return self.a_dagger_list[qindex] * self.a_list[qindex]
                

    def H_Z_bias(self, pulse: PulseConfig, sim_opts: SimulationOption):
        """
        Define the Hamiltonian of the system under Z pulse biasing
        """
        flux_pulse = pulse.get_pulse(sim_opts)

        return [self.get_H_Z_bias(pulse.qindex), flux_pulse]


    def get_data_list(self, result, sim_opts: SimulationOption, state_list):
        data_list = []
        for state in state_list:
            data_list_dummy = []
            for ii in range(0, sim_opts.simulation_point):
                data_list_dummy.append(np.abs(((result.states[ii]).dag()*state))**2)
            data_list.append(data_list_dummy)

        return data_list
    

    def get_data_list_density(self, result, sim_opts: SimulationOption, state_list: List[Qobj]):
        data_list = []

        for index, state in enumerate(state_list):
            data_list_dummy = []
            for ii in range(0, sim_opts.simulation_point):
                data_list_dummy.append(((result.states[ii] * result[index].states[ii].dag()) * (state * state.dag())).tr())
            data_list.append(data_list_dummy)

        return data_list
