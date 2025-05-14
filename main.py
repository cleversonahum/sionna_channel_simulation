import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf

import sionna
from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    LSChannelEstimator,
    LMMSEEqualizer,
    OFDMModulator,
    OFDMDemodulator,
    RZFPrecoder,
    RemoveNulledSubcarriers,
)
from sionna.phy.channel.tr38901 import AntennaArray, UMa, PanelArray, CDL
from sionna.phy.channel import (
    subcarrier_frequencies,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    time_lag_discrete_time_channel,
    ApplyOFDMChannel,
    ApplyTimeChannel,
    OFDMChannel,
    TimeChannel,
)
from sionna.phy.channel import gen_single_sector_topology
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, sim_ber, compute_ber


num_ut = 2
num_bs = 1
num_ut_ant = 1
RB_number = 132
subcarrier_spacing = 120e3
carrier_frequency = 28e9  # Carrier frequency in Hz.
num_ofdm_symbols = 14
subcarriers_per_rb = 12

num_streams_per_tx = num_ut_ant


rx_tx_association = np.array([[1, 1]])

sm = StreamManagement(rx_tx_association, num_streams_per_tx)

rg = ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=subcarriers_per_rb
    * RB_number,  # TODO check if FFT size must obbey to 3GPP
    subcarrier_spacing=subcarrier_spacing,
    num_tx=num_bs,
    num_streams_per_tx=num_streams_per_tx,
    cyclic_prefix_length=6,
    num_guard_carriers=[5, 6],
    dc_null=True,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[2, 11],
)


bs_array = PanelArray(
    num_rows_per_panel=4,
    num_cols_per_panel=4,
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency,
)

ut_array = PanelArray(
    num_rows_per_panel=1,
    num_cols_per_panel=1,
    polarization="single",
    polarization_type="V",
    antenna_pattern="omni",
    carrier_frequency=carrier_frequency,
)

channel_model = UMa(
    carrier_frequency=3.5e9,
    o2i_model="low",
    ut_array=ut_array,
    bs_array=bs_array,
    direction="downlink",
)

# Set topology
topology = gen_single_sector_topology(
    batch_size=1000, num_ut=num_ut, scenario="uma", min_ut_velocity=5, max_ut_velocity=5
)
channel_model.set_topology(*topology)
channel_model.show_topology(batch_index=0)
channel_model.show_topology(batch_index=999)

a, tau, *_ = channel_model(num_time_samples=2, sampling_frequency=2e3)
frequencies = subcarrier_frequencies(
    subcarriers_per_rb * RB_number, rg.subcarrier_spacing
)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)

plt.show()
print("")
