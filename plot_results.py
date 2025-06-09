import numpy as np
from utils import (
    plot_ut_trajectories,
    plot_se_cdf_for_tti,
    plot_se_per_rb,
    plot_se_boxplot_per_ue,
    plot_se_statistics_per_rb,
)

plot_positions = True
plot_cdf_se = True
plot_se_rb = True
plot_se_boxplot = True
plot_aggregated_se = True
cdf_se_tti = 10
channel_folder = "channel_files/"
results_folder = "plot_results/"


if plot_positions:
    # Load the data
    data = np.load("channel_files/uts_pos.npz")
    all_uts_pos = data["all_uts_pos"]
    all_uts_indoor = data["all_uts_indoor"]
    min_bs_ut_dis = data["min_bs_ut_dis"]
    max_bs_ut_dis = data["max_bs_ut_dis"]
    bs_height = data["bs_height"]

    # Plot the trajectories of the UEs
    plot_ut_trajectories(
        all_uts_pos,
        all_uts_indoor,
        np.zeros(2),  # Base station position
        bs_height,
        min_bs_ut_dis,
        max_bs_ut_dis,
    )

if plot_cdf_se:
    plot_se_cdf_for_tti(
        cdf_se_tti,
        channel_folder,
        results_folder,
    )

if plot_se_rb:
    plot_se_per_rb(
        cdf_se_tti,
        channel_folder,
        results_folder,
    )

if plot_se_boxplot:
    plot_se_boxplot_per_ue(
        cdf_se_tti,
        channel_folder,
        results_folder,
    )

if plot_aggregated_se:
    plot_se_statistics_per_rb(
        cdf_se_tti,
        channel_folder,
        results_folder,
    )
