import os

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0  # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sionna
import csv
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid,
)
from sionna.phy.channel.tr38901 import UMa, PanelArray
from sionna.phy.channel import (
    subcarrier_frequencies,
    cir_to_ofdm_channel,
)
from sionna.phy.channel import gen_single_sector_topology

sionna.phy.config.seed = 40
num_ut = 10
num_bs = 1
num_bs_ant = 16  # Must be a perfect square
num_ut_ant = 1
number_rbs = 528
subcarrier_spacing = 15e3
carrier_frequency = 3.7e9  # Carrier frequency in Hz.
num_ofdm_symbols = 14
subcarriers_per_rb = 12
precoding_technique = (
    "mrt"  # "mrt" = Maximum Ratio Transmission and "none" = equal power allocation
)
p_tx = 40.0  # Watts
n0 = 1e-9  # Noise power spectral density
batch_size = 10
time_steps_per_batch = 2
ues_velocity = 5  # m/s
num_streams_per_tx = num_ut_ant
rx_tx_association = np.ones((num_bs, num_ut))
plot_ue_se = False  # Plot UE SE per RB for the first batch and num_time_steps element
write_to_file = True  # Write the results to a file

sm = StreamManagement(rx_tx_association, num_streams_per_tx)

rg = ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=subcarriers_per_rb * number_rbs,
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
    num_rows_per_panel=int(np.sqrt(num_bs_ant)),
    num_cols_per_panel=int(np.sqrt(num_bs_ant)),
    polarization="single",
    polarization_type="V",
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
    batch_size=batch_size,
    num_ut=num_ut,
    scenario="uma",
    min_ut_velocity=ues_velocity,
    max_ut_velocity=ues_velocity,
    indoor_probability=0.0,
)
channel_model.set_topology(*topology)

a, tau, *_ = channel_model(
    num_time_samples=time_steps_per_batch,
    sampling_frequency=1 / rg.ofdm_symbol_duration,
)
frequencies = subcarrier_frequencies(
    subcarriers_per_rb * number_rbs, rg.subcarrier_spacing
)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)


if precoding_technique == "mrt":  # Maximal Ratio Transmission
    channel_norms = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(h_freq), 2), 4))
    snr = (
        tf.pow(channel_norms, 2)
        * p_tx
        / (n0 * subcarriers_per_rb * number_rbs * rg.subcarrier_spacing)
    )
elif precoding_technique == "none":  # Equal power allocation
    snr = tf.reduce_sum(
        tf.pow(tf.abs(h_freq), 2)
        * (p_tx / num_bs_ant)
        / (n0 * subcarriers_per_rb * number_rbs * rg.subcarrier_spacing),
        4,
    )
else:
    raise ValueError("Invalid precoding technique. Choose either 'mrt' or 'none'.")
avg_snr_rb = tf.reduce_mean(
    tf.reshape(
        snr,
        (
            batch_size,
            num_ut,
            num_ut_ant,
            num_bs,
            time_steps_per_batch,
            number_rbs,
            subcarriers_per_rb,
        ),
    ),
    axis=-1,
)
avg_se_rb = np.log2(1 + avg_snr_rb)

if plot_ue_se:  # Plot UE SE per RB for the first batch and num_time_steps element
    plt.figure()
    for ue in range(num_ut):
        plt.plot(avg_se_rb[0, ue, 0, 0, 0, :], label=f"UE {ue+1}")
    plt.grid()
    plt.legend()
    plt.show()
    print(
        f"Average achievable throughput per UE (Mbps): {np.mean(avg_se_rb[0, :, 0, 0, 0, :], axis=-1) * subcarriers_per_rb * number_rbs * subcarrier_spacing / 1e6}"
    )
if write_to_file:  # Write the results to a file
    for ue in range(num_ut):
        with open(f"channel_files/ue_{ue+1}_se.csv", "w") as file:
            for batch_idx in range(batch_size):
                for time_step in range(time_steps_per_batch):
                    writer = csv.writer(file)
                    writer.writerow(
                        avg_se_rb[batch_idx, ue, 0, 0, time_step, :]
                    )  # Write the entire list as a single row
