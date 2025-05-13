import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf

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
from sionna.phy.channel.tr38901 import AntennaArray, UMa, PanelArray
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
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, sim_ber, compute_ber


# Define the number of UT and BS antennas.
# For the CDL model, that will be used in this notebook, only
# a single UT and BS are supported.
num_ut = 2
num_bs = 1
num_ut_ant = 1

# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
num_streams_per_tx = num_ut_ant

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.array([[1, 1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)

rg = ResourceGrid(
    num_ofdm_symbols=14,
    fft_size=76,
    subcarrier_spacing=15e3,
    num_tx=1,
    num_streams_per_tx=num_streams_per_tx,
    cyclic_prefix_length=6,
    num_guard_carriers=[5, 6],
    dc_null=True,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[2, 11],
)
rg.show()


carrier_frequency = 28e9  # Carrier frequency in Hz.


bs_array = PanelArray(
    num_rows_per_panel=4,
    num_cols_per_panel=4,
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency,
)
bs_array.show()

ut_array = PanelArray(
    num_rows_per_panel=1,
    num_cols_per_panel=1,
    polarization="single",
    polarization_type="V",
    antenna_pattern="omni",
    carrier_frequency=carrier_frequency,
)
ut_array.show()

channel_model = UMa(
    carrier_frequency=3.5e9,
    o2i_model="low",
    ut_array=ut_array,
    bs_array=bs_array,
    direction="uplink",
)

ut_loc = tf.constant(
    [[[100.0, 0.0, 1.5], [100.0, 50.0, 1.5]]], dtype=tf.float32
)  # [1, 2, 3]
bs_loc = tf.constant([[[0.0, 0.0, 10.0]]], dtype=tf.float32)  # [1, 1, 3]
ut_vel = tf.constant(
    [[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]], dtype=tf.float32
)  # [1, 2, 3]
ut_orient = tf.constant(
    [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=tf.float32
)  # [1, 2, 3]
bs_orient = tf.constant([[[0.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]

# Set topology
channel_model.set_topology(ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state=False)
channel_model.show_topology()
plt.show()
