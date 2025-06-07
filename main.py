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
from sionna.phy.channel.tr38901 import UMa, PanelArray
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, utils
from utils import gen_custom_topology, plot_ut_trajectories


sionna.phy.config.seed = 40
plot_ut_se = False  # Plot ut SE per RB for the first batch and num_time_steps element
write_to_file = False  # Write the results to a file
plot_all_ut_trajectories = True  # Plot the trajectories of the uts
num_ut = 2
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
time_steps_per_batch = 1
num_episodes = 5  # Number of episodes to simulate (each episode contains bath_size*time_steps_per_batch time steps)
uts_min_velocity = 5  # m/s
uts_max_velocity = 5  # m/s
num_streams_per_tx = num_ut_ant
min_bs_ut_dis = None  # Meters (None to use the default UMA values)
max_bs_ut_dis = None  # Meters (None to use the default UMA values)
bs_height = None  # Meters (None to use the default UMA values)
min_ut_height = None  # Meters (None to use the default UMA values)
max_ut_height = None  # Meters (None to use the default UMA values)
ut_indoor_probability = (
    0.0  # Probability of the UT being indoors (0.0 for outdoor only)
)
rx_tx_association = np.ones((num_bs, num_ut))

# Generate parameters according to UMa scenario
(
    min_bs_ut_dis,
    max_bs_ut_dis,
    bs_height,
    min_ut_height,
    max_ut_height,
    indoor_probability,
    min_ut_velocity,
    max_ut_velocity,
) = utils.set_3gpp_scenario_parameters(
    "uma",
    min_bs_ut_dis,
    max_bs_ut_dis,
    bs_height,
    min_ut_height,
    max_ut_height,
    ut_indoor_probability,
    uts_min_velocity,
    uts_max_velocity,
)

bs_array = PanelArray(
    num_rows_per_panel=int(np.sqrt(num_bs_ant)),
    num_cols_per_panel=int(np.sqrt(num_bs_ant)),
    polarization="single",
    polarization_type="V",
    antenna_pattern="omni",
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
topology = None  # Initialize topology variable
all_uts_pos = tf.zeros([0, num_ut, 3], dtype=tf.float32)
all_uts_indoor = tf.zeros([0, num_ut], dtype=tf.bool)
for episode in range(num_episodes):
    if topology is None:
        # Set topology in the first iteration
        topology = gen_custom_topology(
            batch_size,
            num_ut,
            min_bs_ut_dis=min_bs_ut_dis,
            max_bs_ut_dis=max_bs_ut_dis,
            bs_height=bs_height,
            min_ut_height=min_ut_height,
            max_ut_height=max_ut_height,
            indoor_probability=ut_indoor_probability,
            min_ut_velocity=uts_min_velocity,
            max_ut_velocity=uts_max_velocity,
            precision=None,
            time_between_batches=10,  # TODO
        )
    else:
        # Update topology in subsequent iterations TODO
        initial_pos_uts = topology[0][-1]
        topology = gen_custom_topology(
            batch_size,
            num_ut,
            min_bs_ut_dis=min_bs_ut_dis,
            max_bs_ut_dis=max_bs_ut_dis,
            bs_height=bs_height,
            min_ut_height=min_ut_height,
            max_ut_height=max_ut_height,
            indoor_probability=ut_indoor_probability,
            min_ut_velocity=uts_min_velocity,
            max_ut_velocity=uts_max_velocity,
            precision=None,
            time_between_batches=10,  # TODO
            initial_ut_loc=initial_pos_uts,
        )
    channel_model.set_topology(*topology)
    all_uts_pos = tf.concat([all_uts_pos, topology[0]], axis=0)
    all_uts_indoor = tf.concat([all_uts_indoor, topology[5]], axis=0)
    a, tau, *_ = channel_model(
        num_time_samples=time_steps_per_batch,
        sampling_frequency=subcarrier_spacing,
    )
    frequencies = subcarrier_frequencies(
        subcarriers_per_rb * number_rbs, subcarrier_spacing
    )
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)

    if precoding_technique == "mrt":  # Maximal Ratio Transmission
        channel_norms = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(h_freq), 2), 4))
        snr = (
            tf.pow(channel_norms, 2)
            * p_tx
            / (n0 * subcarriers_per_rb * number_rbs * subcarrier_spacing)
        )
    elif precoding_technique == "none":  # Equal power allocation
        snr = tf.reduce_sum(
            tf.pow(tf.abs(h_freq), 2)
            * (p_tx / num_bs_ant)
            / (n0 * subcarriers_per_rb * number_rbs * subcarrier_spacing),
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

    if plot_ut_se:  # Plot ut SE per RB for the first batch and num_time_steps element
        plt.figure()
        for ut in range(num_ut):
            plt.plot(avg_se_rb[0, ut, 0, 0, 0, :], label=f"ut {ut+1}")
        plt.grid()
        plt.legend()
        plt.show()
        print(
            f"Average achievable throughput per ut (Mbps): {np.mean(avg_se_rb[0, :, 0, 0, 0, :], axis=-1) * subcarriers_per_rb * number_rbs * subcarrier_spacing / 1e6}"
        )
    if write_to_file:  # Write the results to a file
        file_mode = "w" if episode == 0 else "a"
        for ut in range(num_ut):
            with open(f"channel_files/ut_{ut+1}_se.csv", file_mode) as file:
                for batch_idx in range(batch_size):
                    for time_step in range(time_steps_per_batch):
                        writer = csv.writer(file)
                        writer.writerow(
                            avg_se_rb[batch_idx, ut, 0, 0, time_step, :]
                        )  # Write the entire list as a single row

if plot_all_ut_trajectories:  # Plot the trajectories of the uts
    plot_ut_trajectories(
        all_uts_pos,
        all_uts_indoor,
        np.zeros(2),
        bs_height,
        min_bs_ut_dis,
        max_bs_ut_dis,
    )
print("Simulation completed successfully.")
