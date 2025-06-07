import tensorflow as tf
from sionna.phy import PI, config, dtypes
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def gen_custom_topology(
    batch_size,
    num_ut,
    min_bs_ut_dis=None,
    max_bs_ut_dis=None,
    bs_height=None,
    min_ut_height=None,
    max_ut_height=None,
    indoor_probability=None,
    min_ut_velocity=None,
    max_ut_velocity=None,
    precision=None,
    time_between_batches=1e-3,
    initial_ut_loc=None,
):
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Setting BS to (0,0,bs_height)
    bs_loc = tf.stack(
        [
            tf.zeros([batch_size, 1], rdtype),
            tf.zeros([batch_size, 1], rdtype),
            tf.fill([batch_size, 1], bs_height),
        ],
        axis=-1,
    )

    # Setting the BS orientation to zero since it is omnidirectional
    # and the BS is located at the origin
    bs_orientation = tf.stack(
        [
            tf.fill([batch_size, 1], 0.0),
            tf.fill([batch_size, 1], 0.0),
            tf.zeros([batch_size, 1], rdtype),
        ],
        axis=-1,
    )

    ut_topology = generate_ut_trajectories(
        batch_size,
        num_ut,
        np.zeros([2]),
        min_bs_ut_dis.numpy(),
        max_bs_ut_dis.numpy(),
        min_ut_height.numpy(),
        max_ut_height.numpy(),
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
        time_between_batches,
        initial_ut_loc=initial_ut_loc,
    )
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state


# def generate_ut_trajectories(
#     batch_size: int,
#     num_ut: int,
#     cell_loc_xy: np.ndarray,
#     min_bs_ut_dist: float,
#     max_bs_ut_dist: float,
#     min_ut_height: float,
#     max_ut_height: float,
#     indoor_probability: float,
#     min_ut_velocity: float,
#     max_ut_velocity: float,
#     time_between_batches: float,
#     precision: str = "float32",
#     initial_ut_loc: np.ndarray = None,
# ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
#     """
#     Generate UT trajectories across multiple batches with specified constraints.

#     Args:
#         batch_size: Number of batches (time steps)
#         num_ut: Number of UTs
#         cell_loc_xy: BS location [x, y] (meters)
#         min_bs_ut_dist: Minimum distance from BS (meters)
#         max_bs_ut_dist: Maximum distance from BS (meters)
#         min_ut_height: Minimum UT height (meters)
#         max_ut_height: Maximum UT height (meters)
#         indoor_probability: Probability of UT being indoor [0,1]
#         min_ut_velocity: Minimum UT velocity (m/s)
#         max_ut_velocity: Maximum UT velocity (m/s)
#         time_between_batches: Time between batches (seconds)
#         precision: Numerical precision ('float32' or 'float64')

#     Returns:
#         ut_loc: UT locations [batch_size, num_ut, 3] (x,y,z)
#         ut_orientations: UT orientations [batch_size, num_ut, 3] (yaw,pitch,roll)
#         ut_velocities: UT velocities [batch_size, num_ut, 3] (vx,vy,vz)
#         in_state: Indoor state [batch_size, num_ut] (bool)
#     """
#     # Convert inputs to tensor with specified precision
#     dtype = tf.float32 if precision == "float32" else tf.float64
#     cell_loc = tf.constant([cell_loc_xy[0], cell_loc_xy[1], 0.0], dtype=dtype)

#     # Initialize arrays
#     ut_loc = tf.TensorArray(dtype, size=batch_size)
#     ut_orientations = tf.TensorArray(dtype, size=batch_size)
#     ut_velocities = tf.TensorArray(dtype, size=batch_size)
#     in_state = tf.TensorArray(tf.bool, size=batch_size)

#     # Generate initial positions
#     angles = tf.random.uniform([num_ut], 0, 2 * np.pi, dtype=dtype)
#     distances = tf.random.uniform([num_ut], min_bs_ut_dist, max_bs_ut_dist, dtype=dtype)

#     # Initial positions (x,y)
#     xy = (
#         tf.stack([distances * tf.cos(angles), distances * tf.sin(angles)], axis=1)
#         + cell_loc_xy
#     )

#     # Initial heights
#     z = tf.random.uniform([num_ut], min_ut_height, max_ut_height, dtype=dtype)

#     current_pos = (
#         tf.concat([xy, tf.expand_dims(z, 1)], axis=1)
#         if initial_ut_loc is None
#         else initial_ut_loc
#     )

#     # Initial velocities (random direction with random speed)
#     speeds = tf.random.uniform([num_ut], min_ut_velocity, max_ut_velocity, dtype=dtype)
#     velocity_dirs = tf.random.normal([num_ut, 3], dtype=dtype)
#     velocity_dirs = velocity_dirs / tf.norm(velocity_dirs, axis=1, keepdims=True)
#     current_vel = velocity_dirs * tf.expand_dims(speeds, 1)

#     # Indoor states
#     is_indoor = tf.random.uniform([num_ut]) < indoor_probability

#     for t in range(batch_size):
#         # Store current state
#         ut_loc = ut_loc.write(t, current_pos)
#         ut_orientations = ut_orientations.write(
#             t, tf.random.uniform([num_ut, 3], -np.pi, np.pi, dtype=dtype)
#         )
#         ut_velocities = ut_velocities.write(t, current_vel)
#         in_state = in_state.write(t, is_indoor)

#         # Update positions (Euler integration)
#         if t < batch_size - 1:
#             displacement = current_vel * time_between_batches
#             new_pos = current_pos + displacement

#             # Enforce distance constraints
#             vec_to_bs = (new_pos - cell_loc).numpy()
#             dist_to_bs = tf.norm(vec_to_bs[:, :2], axis=1).numpy()

#             # Adjust positions that violate constraints
#             too_close = dist_to_bs < min_bs_ut_dist
#             too_far = dist_to_bs > max_bs_ut_dist

#             if tf.reduce_any(too_close):
#                 # Push away from BS
#                 adjust_dir = vec_to_bs[too_close, :2]
#                 adjust_dir = adjust_dir / tf.norm(adjust_dir, axis=1, keepdims=True)
#                 new_pos_xy = cell_loc_xy + min_bs_ut_dist * adjust_dir
#                 new_pos = tf.tensor_scatter_nd_update(
#                     new_pos,
#                     tf.where(too_close),
#                     tf.concat(
#                         [new_pos_xy, tf.expand_dims(new_pos.numpy()[too_close, 2], 1)],
#                         axis=1,
#                     ),
#                 )

#             if tf.reduce_any(too_far):
#                 # Pull toward BS
#                 adjust_dir = -vec_to_bs[too_far, :2]
#                 adjust_dir = adjust_dir / tf.norm(adjust_dir, axis=1, keepdims=True)
#                 new_pos_xy = cell_loc_xy + max_bs_ut_dist * adjust_dir
#                 new_pos = tf.tensor_scatter_nd_update(
#                     new_pos,
#                     tf.where(too_far),
#                     tf.concat(
#                         [new_pos_xy, tf.expand_dims(new_pos.numpy()[too_far, 2], 1)],
#                         axis=1,
#                     ),
#                 )

#             # Enforce height constraints
#             new_z = tf.clip_by_value(new_pos[:, 2], min_ut_height, max_ut_height)
#             new_pos = tf.concat([new_pos[:, :2], tf.expand_dims(new_z, 1)], axis=1)

#             current_pos = new_pos

#     # Convert to dense tensors
#     ut_loc = ut_loc.stack()
#     ut_orientations = ut_orientations.stack()
#     ut_velocities = ut_velocities.stack()
#     in_state = in_state.stack()

#     return ut_loc, ut_orientations, ut_velocities, in_state


def generate_ut_trajectories(
    batch_size: int,
    num_ut: int,
    cell_loc_xy: np.ndarray,
    min_bs_ut_dist: float,
    max_bs_ut_dist: float,
    min_ut_height: float,
    max_ut_height: float,
    indoor_probability: float,
    min_ut_velocity: float,
    max_ut_velocity: float,
    time_between_batches: float,
    precision: str = "float32",
    initial_ut_loc: np.ndarray = None,
    direction_change_prob: float = 0.2,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Generate UT trajectories with realistic boundary handling and periodic direction changes.

    Args:
        direction_change_prob: Probability (0-1) of changing direction each time step
        ... (other params same as before)
    """
    dtype = tf.float32 if precision == "float32" else tf.float64
    cell_loc = tf.constant([cell_loc_xy[0], cell_loc_xy[1], 0.0], dtype=dtype)

    # Initialize arrays
    ut_loc = tf.TensorArray(dtype, size=batch_size)
    ut_orientations = tf.TensorArray(dtype, size=batch_size)
    ut_velocities = tf.TensorArray(dtype, size=batch_size)
    in_state = tf.TensorArray(tf.bool, size=batch_size)

    # Generate initial positions
    if initial_ut_loc is None:
        angles = tf.random.uniform([num_ut], 0, 2 * np.pi, dtype=dtype)
        distances = tf.random.uniform(
            [num_ut], min_bs_ut_dist, max_bs_ut_dist, dtype=dtype
        )
        xy = (
            tf.stack([distances * tf.cos(angles), distances * tf.sin(angles)], axis=1)
            + cell_loc_xy
        )
        z = tf.random.uniform([num_ut], min_ut_height, max_ut_height, dtype=dtype)
        current_pos = tf.concat([xy, tf.expand_dims(z, 1)], axis=1)
    else:
        current_pos = tf.convert_to_tensor(initial_ut_loc, dtype=dtype)

    # Initial velocities
    speeds = tf.random.uniform([num_ut], min_ut_velocity, max_ut_velocity, dtype=dtype)
    velocity_dirs = tf.random.normal([num_ut, 3], dtype=dtype)
    velocity_dirs = velocity_dirs / tf.norm(velocity_dirs, axis=1, keepdims=True)
    current_vel = velocity_dirs * tf.expand_dims(speeds, 1)

    # Indoor states
    is_indoor = tf.random.uniform([num_ut]) < indoor_probability

    for t in range(batch_size):
        # Store current state
        ut_loc = ut_loc.write(t, current_pos)
        ut_orientations = ut_orientations.write(
            t, tf.random.uniform([num_ut, 3], -np.pi, np.pi, dtype=dtype)
        )
        ut_velocities = ut_velocities.write(t, current_vel)
        in_state = in_state.write(t, is_indoor)

        if t < batch_size - 1:
            # Random direction changes
            change_dir = tf.random.uniform([num_ut]) < direction_change_prob
            if tf.reduce_any(change_dir):
                num_changing = tf.reduce_sum(tf.cast(change_dir, tf.int32))
                new_dirs = tf.random.normal(
                    [num_changing, 3], dtype=dtype
                )  # Fixed parenthesis here
                new_dirs = new_dirs / tf.norm(new_dirs, axis=1, keepdims=True)
                current_vel = tf.tensor_scatter_nd_update(
                    current_vel,
                    tf.where(change_dir),
                    new_dirs
                    * tf.expand_dims(tf.gather(speeds, tf.where(change_dir)[:, 0]), 1),
                )

            # Update positions
            displacement = current_vel * time_between_batches
            new_pos = current_pos + displacement

            # Enforce distance constraints with bounce physics
            vec_to_bs = new_pos - cell_loc
            dist_to_bs = tf.norm(vec_to_bs[:, :2], axis=1)
            too_close = dist_to_bs < min_bs_ut_dist
            too_far = dist_to_bs > max_bs_ut_dist

            # Handle boundary violations
            if tf.reduce_any(too_close) or tf.reduce_any(too_far):
                violating = too_close | too_far
                violating_pos = tf.gather(new_pos, tf.where(violating)[:, 0])
                violating_vel = tf.gather(current_vel, tf.where(violating)[:, 0])
                vec_to_bs_violating = violating_pos[:, :2] - cell_loc_xy

                # Calculate surface normal and bounce direction
                normals = vec_to_bs_violating / tf.norm(
                    vec_to_bs_violating, axis=1, keepdims=True
                )
                vel_parallel = (
                    tf.reduce_sum(violating_vel[:, :2] * normals, axis=1, keepdims=True)
                    * normals
                )
                vel_perpendicular = violating_vel[:, :2] - vel_parallel

                # New velocity after bounce (with 80% energy retention)
                new_vel_xy = -vel_parallel * 0.8 + vel_perpendicular

                # Adjust position to boundary
                target_dist = tf.where(
                    tf.gather(too_close, tf.where(violating)[:, 0]),
                    min_bs_ut_dist,
                    max_bs_ut_dist,
                )
                new_pos_xy = (
                    cell_loc_xy
                    + (
                        vec_to_bs_violating
                        / tf.norm(vec_to_bs_violating, axis=1, keepdims=True)
                    )
                    * target_dist
                )

                # Update both position and velocity
                new_pos = tf.tensor_scatter_nd_update(
                    new_pos,
                    tf.where(violating),
                    tf.concat(
                        [new_pos_xy, tf.expand_dims(violating_pos[:, 2], 1)], axis=1
                    ),
                )

                current_vel = tf.tensor_scatter_nd_update(
                    current_vel,
                    tf.where(violating),
                    tf.concat(
                        [new_vel_xy, tf.expand_dims(violating_vel[:, 2], 1)], axis=1
                    ),
                )

            # Enforce height constraints
            new_z = tf.clip_by_value(new_pos[:, 2], min_ut_height, max_ut_height)
            new_pos = tf.concat([new_pos[:, :2], tf.expand_dims(new_z, 1)], axis=1)

            current_pos = new_pos

    return (
        ut_loc.stack(),
        ut_orientations.stack(),
        ut_velocities.stack(),
        in_state.stack(),
    )


def plot_ut_trajectories(
    ut_loc,
    in_state,
    cell_loc_xy,
    cell_height,
    min_bs_ut_dist,
    max_bs_ut_dist,
):
    """
    Plot UT trajectories with BS position and cell boundaries.

    Args:
        ut_loc: UT locations [batch_size, num_ut, 3] (x,y,z)
        ut_orientations: UT orientations (not used in plot)
        ut_velocities: UT velocities (not used in plot)
        in_state: Indoor state [batch_size, num_ut] (bool)
        cell_loc_xy: BS location [x, y] (meters)
        min_bs_ut_dist: Minimum cell radius (meters)
        max_bs_ut_dist: Maximum cell radius (meters)
    """
    # Convert tensors to numpy if needed
    if hasattr(ut_loc, "numpy"):
        ut_loc = ut_loc.numpy()
    if hasattr(in_state, "numpy"):
        in_state = in_state.numpy()

    batch_size, num_ut, _ = ut_loc.shape

    # Create figure with 2D and 3D subplots
    fig = plt.figure(figsize=(18, 8))

    # 2D plot (top-down view)
    ax1 = fig.add_subplot(121)
    ax1.set_aspect("equal")

    # Plot cell boundaries
    circle_min = plt.Circle(
        cell_loc_xy, min_bs_ut_dist, color="r", alpha=0.1, label="Min Radius"
    )
    circle_max = plt.Circle(
        cell_loc_xy, max_bs_ut_dist, color="b", alpha=0.1, label="Max Radius"
    )
    ax1.add_patch(circle_min)
    ax1.add_patch(circle_max)

    # Plot BS position
    ax1.scatter(
        cell_loc_xy[0], cell_loc_xy[1], s=200, marker="^", color="red", label="BS"
    )

    # Plot UT trajectories
    for i in range(num_ut):
        # Different colors for indoor/outdoor UTs
        color = "purple" if in_state[0, i] else "blue"

        x = ut_loc[:, i, 0]
        y = ut_loc[:, i, 1]
        ax1.plot(
            x,
            y,
            "o-",
            markersize=4,
            linewidth=1,
            color=color,
            label=f"UT {i+1}" if i < 5 else None,
        )

        # Add start and end markers
        ax1.scatter(x[0], y[0], s=50, marker=">", color=color)
        ax1.scatter(x[-1], y[-1], s=50, marker="s", color=color)

    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("Top-Down View of UT Trajectories")
    ax1.grid(True)

    # Limit legend entries to avoid duplicates
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    # 3D plot
    ax2 = fig.add_subplot(122, projection="3d")

    # Plot BS position
    ax2.scatter(
        cell_loc_xy[0], cell_loc_xy[1], cell_height, s=200, marker="^", color="red"
    )

    # Plot UT trajectories in 3D
    for i in range(num_ut):
        color = "purple" if in_state[0, i] else "blue"
        x = ut_loc[:, i, 0]
        y = ut_loc[:, i, 1]
        z = ut_loc[:, i, 2]

        ax2.plot(x, y, z, "o-", markersize=4, linewidth=1, color=color)

        # Add start and end markers
        ax2.scatter(x[0], y[0], z[0], s=50, marker=">", color=color)
        ax2.scatter(x[-1], y[-1], z[-1], s=50, marker="s", color=color)

    # Add cylindrical cell boundaries
    theta = np.linspace(0, 2 * np.pi, 100)
    x_min = cell_loc_xy[0] + min_bs_ut_dist * np.cos(theta)
    y_min = cell_loc_xy[1] + min_bs_ut_dist * np.sin(theta)
    x_max = cell_loc_xy[0] + max_bs_ut_dist * np.cos(theta)
    y_max = cell_loc_xy[1] + max_bs_ut_dist * np.sin(theta)

    z_min = np.linspace(0, np.max(ut_loc[:, :, 2]), 2)
    theta_grid, z_grid = np.meshgrid(theta, z_min)

    ax2.plot_surface(
        cell_loc_xy[0] + min_bs_ut_dist * np.cos(theta_grid),
        cell_loc_xy[1] + min_bs_ut_dist * np.sin(theta_grid),
        z_grid,
        color="r",
        alpha=0.1,
    )

    ax2.plot_surface(
        cell_loc_xy[0] + max_bs_ut_dist * np.cos(theta_grid),
        cell_loc_xy[1] + max_bs_ut_dist * np.sin(theta_grid),
        z_grid,
        color="b",
        alpha=0.1,
    )

    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.set_zlabel("Height (m)")
    ax2.set_title("3D View of UT Trajectories")

    plt.tight_layout()
    plt.show()
