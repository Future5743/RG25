########################################################################################################################
##################################################### IMPORTS ##########################################################
########################################################################################################################
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import messagebox

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################

def calculate_distance(pos1, pos2, pixel_size):
    '''
    This function computes the real-world distance in meters between two points.

    Parameters:
    -----------
    pos1: list, array, or tuple
        Relative coordinates of the first point

    pos2: list, array, or tuple
        Relative coordinates of the second point

    pixel_size: int
        Size of a pixel on the terrain (in meters)

    Returns:
    --------
    float
        Distance between the two points in meters
    '''
    pixel_dist = np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
    return pixel_dist * pixel_size


def find_alt_point_indices(profile, alt_points):
    '''
    Finds the indices of specific altitude values in a profile.

    Parameters:
    -----------
    profile: list of float
        Original elevation profile (not cleaned)

    alt_points: list
        List of two sublists [min, max] for each half-profile

    Returns:
    --------
    list
        List of indices where the altitudes match the ones given as input
    '''

    alt_1 = alt_points[0][-1]
    idx1 = np.where(profile == alt_1)[0]

    alt_2 = alt_points[1][-1]
    idx2 = np.where(profile == alt_2)[0]

    return [int(idx1), int(idx2)]


def calculate_cumulative_distances(coords_x, coords_y, pixel_size_tb):
    '''
    Computes the cumulative distance from the starting point for each point in a profile.

    Parameters:
    -----------
    coords_x: array
        X coordinates of each point in the profile

    coords_y: array
        Y coordinates of each point in the profile

    pixel_size_tb: int
        Terrain pixel size (in meters)

    Returns:
    --------
    list
        Cumulative distance from the first point to each point in the profile
    '''

    start_point = [coords_x[0], coords_y[0]]
    return [
        calculate_distance(start_point, [coords_x[p], coords_y[p]], pixel_size_tb)
        for p in range(len(coords_x))
    ]


def process_profile(demi_profils_value, demi_profils_coords_relatives, i, pixel_size_tb, min_X):
    '''
    Processes an individual full profile based on a 10° angular interval.

    Parameters:
    -----------
    demi_profils_value: list
        Elevation values for each semi-profile

    demi_profils_coords_relatives: list
        Relative coordinates for each point in the semi-profiles

    i: int
        Profile index (usually angle index)

    pixel_size_tb: int
        Terrain pixel size (in meters)

    min_X: list
        The shortest x-axis profile (used for alignment)

    Returns:
    --------
    full_profile: list
        Combined elevation values for the full profile

    X: list
        Distances of each point from the origin

    min_X: list
        Updated shortest x-axis profile

    '''
    # Process coordinates for both halves
    x1 = demi_profils_coords_relatives[i][0][::-1][:-1]
    y1 = demi_profils_coords_relatives[i][1][::-1][:-1]
    x2 = demi_profils_coords_relatives[i + 18][0]
    y2 = demi_profils_coords_relatives[i + 18][1]

    # Concatenate coordinates
    coords_x = np.concatenate((x1, x2))
    coords_y = np.concatenate((y1, y2))

    # Compute cumulative distances
    X = calculate_cumulative_distances(coords_x, coords_y, pixel_size_tb)

    # Keep the shortest X (for alignment with other profiles)
    min_X = min(min_X, X, key=len)

    # Combine elevation values
    full_profile = demi_profils_value[i][::-1][:-1] + demi_profils_value[i + 18]

    # Replace invalid values with NaN
    full_profile = [val if isinstance(val, np.float32) else np.nan for val in full_profile]

    return full_profile, X, min_X


def profile_derivative(full_profile, X):
    '''
    Computes the second derivative of a topographic profile using cubic splines.

    Parameters:
    -----------
    full_profile: list
        Elevation values of the profile

    X: list
        Corresponding x-axis values (distance)

    Returns:
    --------
    list
        Second derivative values of the profile
    '''
    full_profile = np.array(full_profile)
    X = np.array(X)

    # Handle missing values by forward/backward filling
    if not np.all(np.isfinite(full_profile)):
        index_nan = np.where(np.isnan(full_profile))
        for i in index_nan[0]:
            a = i
            if i < len(full_profile)/2:
                while np.isnan(full_profile[i]):
                    a += 1
                    full_profile[i] = full_profile[a]
            else:
                while np.isnan(full_profile[i]):
                    a -= 1
                    full_profile[i] = full_profile[a]

    # Create cubic spline interpolation
    spline = CubicSpline(X, full_profile)

    # Get second derivative
    spline_derivative = spline.derivative(nu=2)

    # Evaluate second derivative at X
    dy_dx = spline_derivative(X)

    return dy_dx


def find_right_point(points, point_min):
    '''
    This function finds and returns the index of the point to be considered as the **right boundary**
    of the crater floor on the profile.

    To be selected, the point must meet the following criteria:
        * It cannot be at the extremes of the semi-profile (to avoid edge effects and the lowest point).
        * It must be the highest value (second derivative) following the previous criteria.

    Parameters:
    -----------
    points: list
        Contains the values of the second derivative.

    point_min: int
        Index of the lowest point (crater center) in the profile.

    Returns:
    --------
    max_idx: int
        Index of the selected point, considered the right boundary of the crater floor.
    '''

    start = point_min + 1
    available = len(points) - start

    if available <= 1:
        return len(points) - 2

    n_tranche = max(1, available)
    end = min(start + n_tranche, len(points) - 1)

    max_val = float('-inf')
    max_idx = start
    for i in range(start, end):
        if points[i] > max_val:
            max_val = points[i]
            max_idx = i

    return max_idx


def find_left_point(points, point_min):
    '''
    This function finds and returns the index of the point to be considered as the **left boundary**
    of the crater floor on the profile.

    To be selected, the point must meet the following criteria:
        * It cannot be at the extremes of the semi-profile (to avoid edge effects and the lowest point).
        * It must be the highest value (second derivative) following the previous criteria.

    Parameters:
    -----------
    points: list
        Contains the values of the second derivative.

    point_min: int
        Index of the lowest point (crater center) in the profile.

    Returns:
    --------
    max_idx: int
        Index of the selected point, considered the left boundary of the crater floor.
    '''

    end = point_min  # include the point right before the minimum
    available = end - 1  # avoid using points[0]

    if available < 1:
        return 1  # fallback: return second point

    n_tranche = max(1, available)
    start = max(1, end - n_tranche)

    max_val = float('-inf')
    max_idx = start
    for i in range(start, end):  # end is excluded
        if points[i] > max_val:
            max_val = points[i]
            max_idx = i

    return max_idx


def pseudo_floor(X, derivate, point_min):
    '''
    This function computes the estimated crater floor limits based on the second derivative values.

    Parameters:
    -----------
    X: list
        The x-axis (distance values) of the profile.

    derivate: list
        Contains the second derivative values of the profile.

    point_min: int
        Index of the lowest point in the profile (center of the crater).

    Returns:
    --------
    X_left: float
        X-coordinate of the selected point considered as the left boundary of the crater floor.

    X_right: float
        X-coordinate of the selected point considered as the right boundary of the crater floor.
    '''

    index_left = find_left_point(derivate, point_min)
    index_right = find_right_point(derivate, point_min)

    X_left = X[index_left]
    X_right = X[index_right]

    return X_left, X_right


def build_save_path(zone, swirl_on_or_off, crater_id, i, suffix):
    '''
    This function builds and returns the file path to save a profile figure.

    Parameters:
    -----------
    zone: str
        Crater study zone (can be 1, 2, 3, 4, 5, 6, or 7).

    swirl_on_or_off: str
        Indicates whether the crater is in a swirl region ("on" or "off").

    crater_id: int
        ID of the crater being studied.

    i: int
        Loop iteration index, representing the angle interval of the profile.

    suffix: str
        Suffix indicating the type of figure (e.g., "_second_derivative").

    Returns:
    --------
    str
        Full path to the location where the figure will be saved.
    '''

    base_path = f'results/RG{zone}/profiles/{swirl_on_or_off}/{crater_id}'
    os.makedirs(base_path, exist_ok=True)
    filename = f"Profile_{i * 10}_{(i + 18) * 10}{suffix}.png"
    return os.path.join(base_path, filename)


def save_and_plot_profile(full_profile, X, i, zone, crater_id, swirl_on_or_off,
                          alt_points_inner_20):
    '''
    This function plots and saves a profile and its second derivative.
    It also computes an estimation of the crater floor.
    If the profile length is greater than 400m, the crater floor estimation will be interactive; otherwise, it is
    fully automated.

    How to use the interactive mode:
        * The interactive mode will open automatically if the crater profile length is greater than 400m.
        * It uses a matplotlib interface.
        * Two red dots are displayed in addition to the profile: these are the points the automatic algorithm
          computed for the profile.
        * Two options are available:
            ** If the automatically calculated red dots suit you: just click the "Validate Input" button.
            ** If the automatically calculated red dots don’t suit you: click two points on the profile to select better
               points delimiting the crater floor (the clicked points will appear in orange), then click the "Validate Input" button.
            ** Generally, 18 profiles will be plotted (but not always).

    Parameters:
    -----------
    full_profile: list
        Contains all elevation values of the profile.

    X: list
        The x-axis coordinates.

    i: int
        The current loop iteration.

    zone: str
        Indicates the crater study zone (can be 1, 2, 3, 4, 5, 6, or 7).

    crater_id: int
        The ID of the studied crater.

    swirl_on_or_off: str
        Indicates whether the crater is in swirl mode or not.

    alt_points_inner_20: list
        Indicates the points at an elevation value corresponding to 20% of the total depth.

    Returns:
    --------
    int, int
        Returns two integers:
            - The index in the semi-profile considered as the left delimitation of the crater floor.
            - The index in the semi-profile considered as the right delimitation of the crater floor.
    '''

    limit_profile = np.where(full_profile == np.nanmin(full_profile))[0]

    if len(limit_profile) > 1:
        limit_profile = limit_profile[0]

    limit_profile = int(limit_profile)

    def build_save_path(zone, swirl_on_or_off, crater_id, i, suffix):
        base_path = f'results/RG{zone}/profiles/{swirl_on_or_off}/{crater_id}'
        os.makedirs(base_path, exist_ok=True)
        filename = f"Profile_{i * 10}_{(i + 18) * 10}{suffix}.png"
        return os.path.join(base_path, filename)

    indices_20 = find_alt_point_indices(full_profile, alt_points_inner_20)

    start_idx = indices_20[0]
    end_idx = indices_20[1]

    X_slice = X[start_idx:end_idx + 1]
    if len(X_slice) <= 2:
        return limit_profile - start_idx, end_idx - limit_profile

    # ---------------- AUTOMATIC METHOD ----------------
    profile_slice = full_profile[start_idx:end_idx + 1]
    second_derivative = profile_derivative(profile_slice, X_slice)

    point_min = np.where(X_slice == X[limit_profile])[0]

    if len(point_min) == 1:
        point_min = int(point_min)
    else:
        point_min = int(point_min[0])

    floor_left, floor_right = pseudo_floor(X_slice, second_derivative, point_min)

    index_left = int(np.argmin(np.abs(np.array(X) - floor_left)))
    index_right = int(np.argmin(np.abs(np.array(X) - floor_right)))

    automatic_indices = [index_left, index_right]

    # ---------------- INTERACTIVE METHOD ----------------

    selected_points = []
    selected_indices = []

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(X, full_profile, color='blue', marker='x', label='Topographic profile')
    ax.set_title(f'Select two points for crater {crater_id} (angle {i * 10}°)')
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Altitude")
    ax.grid(True)

    # Display automatic points (red)
    for idx in automatic_indices:
        ax.scatter(X[idx], full_profile[idx], color='red', s=150, zorder=5)

    def onclick(event):
        if len(selected_points) < 2 and event.inaxes == ax:
            idx = np.argmin(np.abs(np.array(X) - event.xdata))
            if idx not in selected_indices:
                x_real = X[idx]
                y_real = full_profile[idx]
                selected_indices.append(idx)
                selected_points.append((x_real, y_real))
                ax.scatter(x_real, y_real, color='orange', s=150, zorder=10)
                fig.canvas.draw()

    def on_submit(event):
        nonlocal selected_indices

        # If no clicks → keep automatic indices
        if len(selected_indices) < 1:
            selected_indices = automatic_indices
        if len(selected_indices) == 1:
            selected_indices = [selected_indices[0]] * 2

        # Final full profile with legend
        fig_final, ax_final = plt.subplots(figsize=(15, 7))

        for idx in indices_20:
            if idx < len(X):
                ax_final.scatter(X[idx], full_profile[idx], color='red', zorder=5)

        for idx in selected_indices:
            ax_final.scatter(X[idx], full_profile[idx], color='orange', zorder=5)

        ax_final.plot(X, full_profile, color='blue', marker='x', label='Topographic profile')
        ax_final.set_xlabel("Distance (m)")
        ax_final.set_ylabel("Altitude")
        ax_final.set_title(f'Topographic profile for angles {i * 10}° to {(i + 18) * 10}°')
        ax_final.grid(True)
        legend_elements = [
            plt.Line2D([0], [0], color='red', marker='o', linestyle='', label='Inner 20', markersize=10),
            plt.Line2D([0], [0], color='orange', marker='o', linestyle='', label='Floor delimitation',
                       markersize=10)
        ]
        ax_final.legend(handles=legend_elements)

        fig_final.savefig(build_save_path(zone, swirl_on_or_off, crater_id, i=i, suffix=""))
        plt.close(fig_final)
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
    button = Button(ax_button, 'Validate Input')
    button.on_clicked(on_submit)
    plt.show()

    selected_indices = sorted(selected_indices)

    # ---------------- AUTOMATIC SAVING ----------------

    # Second derivative
    idx_l = int(np.argmin(np.abs(X_slice - floor_left)))
    idx_r = int(np.argmin(np.abs(X_slice - floor_right)))

    plt.figure(figsize=(15, 7))
    plt.scatter(X_slice[idx_l], second_derivative[idx_l], color='green', zorder=5)
    plt.scatter(X_slice[idx_r], second_derivative[idx_r], color='green', zorder=5)
    plt.plot(X_slice, second_derivative, color='red', label='Second derivative')
    plt.scatter(X_slice, second_derivative, color='red', s=80)
    plt.xlabel("Distance (m)")
    plt.ylabel("Second derivative")
    plt.title(f'Second derivative for angles {i * 10}° to {(i + 18) * 10}°')
    plt.grid(True)
    plt.legend()
    plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, i=i, suffix="_second_derivative"))
    plt.close()

    return limit_profile - selected_indices[0], selected_indices[1] - limit_profile


def extract_cleaned_profile(profile_coords, profile_values, no_data_value):
    '''
    This function creates a cleaned profile with NaN for no-data values and returns a [x, y, z] structured array.

    Parameters:
    -----------
    profile_coords: list
        Contains the coordinates (x, y) of profile points

    profile_values: list
        Contains the elevation values (z) of the profile points

    no_data_value: ???
        Value indicating no data or invalid elevation

    Returns:
    --------
    profile: np.ndarray
        Cleaned profile as an Nx3 array [[x, y, z], ...], with NaNs for no data
    '''

    m = len(profile_coords[0])
    profile = np.array([[profile_coords[0][j], profile_coords[1][j], profile_values[j]] for j in range(m)])
    profile[profile == no_data_value] = np.nan
    return profile


def find_inner_points(half_profile, depth_value, min_val):
    '''
    This function calculates the two inner points near 20% and 80% of the total depth.

    Parameters:
    -----------
    half_profile: list
        Contains the coordinates and elevation of each point on the semi-profile

    depth_value: float
        Depth value of the semi-profile

    min_val: float
        Elevation of the lowest point in the semi-profile

    Returns:
    --------
    point_min: list
        Point near 20% depth (lower inner point)

    point_max: list
        Point near 80% depth (upper inner point)

    idx_min: int
        Index in the semi-profile of the point near 20%

    idx_max: int
        Index in the semi-profile of the point near 80%
    '''
    alt_min = 0.2 * depth_value + min_val
    alt_max = 0.8 * depth_value + min_val

    # Find the index of the point with maximum elevation
    max_idx = -1
    max_z = -np.inf
    for i, (_, _, z) in enumerate(half_profile):
        if not np.isnan(z) and z > max_z:
            max_z = z
            max_idx = i

    point_min = point_max = None
    idx_min = idx_max = -1
    dist_min = dist_max = np.inf

    # Consider only points before the highest point
    for idx in range(max_idx):
        _, _, z = half_profile[idx]
        if np.isnan(z):
            continue
        if abs(z - alt_min) < dist_min:
            dist_min = abs(z - alt_min)
            point_min = half_profile[idx]
            idx_min = idx
        if abs(z - alt_max) < dist_max:
            dist_max = abs(z - alt_max)
            point_max = half_profile[idx]
            idx_max = idx

    # Ensure the order of indices is correct
    if idx_min > idx_max:
        idx_min, idx_max = idx_max, idx_min
        point_min, point_max = point_max, point_min

    # Avoid points that equal the minimum elevation
    point_min = avoid_min_value_point(half_profile, point_min, idx_min, min_val)
    point_max = avoid_min_value_point(half_profile, point_max, idx_max, min_val)

    return point_min, point_max, idx_min, idx_max


def avoid_min_value_point(profile, point, idx, min_val):
    '''
    This function replaces a point if its elevation equals the minimum value of the profile.

    Parameters:
    -----------
    profile: list
        Contains the coordinates and elevation of each profile point

    point: list
        The studied point [x, y, z]

    idx: int
        Index of the studied point in the profile

    min_val: float
        Elevation of the lowest point in the profile

    Returns:
    --------
    point: list
        Either the original point or a neighboring point close to the 20% or 80% depth
    '''

    if point is None or point[2] != min_val:
        return point

    next_idx = idx + 1 if idx + 1 < len(profile) else idx - 1
    if 0 <= next_idx < len(profile) and not np.isnan(profile[next_idx][2]):
        return profile[next_idx]
    return point


def process_all_inner_points(demi_profiles_coords_relative, demi_profiles_values, no_data_value, depth, min_val):
    '''
    This function processes all profiles to extract inner points near 20% and 80% of depth.

    Parameters:
    -----------
    demi_profiles_coords_relative: list
        Relative coordinates of each point in each semi-profile

    demi_profiles_values: list
        Elevation values of each point in each semi-profile

    no_data_value: ???
        Value representing no data in elevation profiles

    depth: list
        Depth values for each profile

    min_val: float
        Lowest elevation value in the crater

    Returns:
    --------
    points_inner_20: list
        Coordinates and elevation of inner points selected for each profile

    index_inner_20: list
        Indices of the inner points within each semi-profile
    '''
    points_inner_20 = []
    index_inner_20 = []

    for i, (coords, values) in enumerate(zip(demi_profiles_coords_relative, demi_profiles_values)):
        half_profile = extract_cleaned_profile(coords, values, no_data_value)
        clean_values = np.array(values)
        clean_values = clean_values[~np.isnan(clean_values)]
        if len(clean_values) == 0:
            print(f"Profile {i} is empty after cleaning.")
            points_inner_20.append([None, None])
            index_inner_20.append([-1, -1])
            continue

        point_min, point_max, idx_min, idx_max = find_inner_points(
            half_profile, depth[i], min_val
        )

        if point_min is None or point_max is None:
            print(f"Profile {i}: inner point min or max is None")
            points_inner_20.append([None, None])
            index_inner_20.append([-1, -1])
        else:
            points_inner_20.append([point_min, point_max])
            index_inner_20.append([idx_min, idx_max])

    return points_inner_20, index_inner_20


def process_profiles_and_plot(demi_profiles_value, demi_profiles_coords_relatives, pixel_size_tb,
                              points_inner_20, zone, crater_id, swirl_on_or_off):
    '''
    This function generates profiles, saves their plots, and extracts the indices delimiting the crater floor
    in each given profile.

    Parameters:
    -----------
    demi_profiles_value: list
        Contains elevation values for each point in each semi-profile.

    demi_profiles_coords_relatives: list
        Contains the relative coordinates of each point in each semi-profile.

    pixel_size_tb: int
        The size of each pixel on the terrain (spatial resolution).
    points_inner_20: list
        Contains all inner points for each semi-profile (2 points per semi-profile).

    zone: str
        Indicates the study zone of the crater (values can be '1' to '7').

    crater_id: int
        The ID of the crater being studied.

    swirl_on_or_off: str
        Indicates whether the crater is on or off a swirl feature.

    Returns:
    --------
    crater_floor: list
        Contains all points selected to delimit the crater floor.

    crater_morph: str
        Morphology classification of the crater based on user selection (e.g., "Bowl-shaped", "Flat-floored").
    '''

    min_X = [0] * 1000
    crater_floor = [0] * 36
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.subplots_adjust(bottom=0.3)

    crater_morph = None

    colors = ['black', 'silver', 'red', 'saddlebrown', 'orange', 'bisque',  'gold', 'darkgoldenrod', 'yellow',
              'greenyellow', 'green', 'lime', 'turquoise', 'blue', 'skyblue', 'purple', 'plum', 'pink']

    for i in range(int(len(demi_profiles_value) / 2)):

        full_profile, X, min_X = process_profile(
            demi_profiles_value, demi_profiles_coords_relatives, i, pixel_size_tb, min_X)

        # Plot and save each individual profile
        plt.figure(figsize=(15, 7))
        plt.plot(X, full_profile, color='b', label='Profile')
        plt.xlabel("Distance (m)")
        plt.ylabel("Elevation")
        plt.title(f'Topographic Profile for Angles {i * 10}° to {(i + 18) * 10}° of Crater {crater_id}')
        plt.grid(True)
        plt.legend()
        plt.savefig(build_save_path(zone, swirl_on_or_off, crater_id, i=i, suffix=""))
        plt.close()

        # Find index of minimum elevation point (crater floor)
        limit_profile = np.where(full_profile == np.nanmin(full_profile))[0]

        if len(limit_profile) > 1:
            limit_profile = limit_profile[0]

        limit_profile = int(limit_profile)

        X_test = [index - limit_profile for index in range(len(X))]

        # Functions to set crater morphology based on user button clicks
        def set_morph_to_Bowl(event):
            nonlocal crater_morph
            crater_morph = "Bowl-shaped"
            plt.close()

        def set_morph_to_Flat(event):
            nonlocal crater_morph
            crater_morph = "Flat-floored"
            plt.close()

        def set_morph_to_Mound(event):
            nonlocal crater_morph
            crater_morph = "With a mound"
            plt.close()

        def set_morph_to_Unknown(event):
            nonlocal crater_morph
            crater_morph = "Unknown"
            plt.close()

        # Plot profiles centered on their minimum elevation point
        ax.plot(X_test, full_profile, color=colors[i], marker='', label='Topographic Profile', linewidth=0.5)

    # Add buttons for user to classify crater morphology
    button_axes = [
        plt.axes([0.05 + i * 0.2, 0.1, 0.15, 0.075])
        for i in range(4)
    ]

    button1 = Button(button_axes[0], 'Bowl-shaped')
    button2 = Button(button_axes[1], 'Flat-floored')
    button3 = Button(button_axes[2], 'With a mound')
    button4 = Button(button_axes[3], 'Unknown')

    button1.on_clicked(set_morph_to_Bowl)
    button2.on_clicked(set_morph_to_Flat)
    button3.on_clicked(set_morph_to_Mound)
    button4.on_clicked(set_morph_to_Unknown)

    ax.set_title(f'Select two points for Crater {crater_id}')
    ax.set_xlabel("Distance in points relative to the lowest point")
    ax.set_ylabel("Elevation")
    ax.grid(True)
    plt.show()

    # If crater is not bowl-shaped, process profiles differently
    if crater_morph != "Bowl-shaped":
        for i in range(int(len(demi_profiles_value) / 2)):
            full_profile, X, min_X = process_profile(
                demi_profiles_value, demi_profiles_coords_relatives, i, pixel_size_tb, min_X)

            alt_20 = [points_inner_20[i][0], points_inner_20[i + 18][0]]

            idx1, idx2 = save_and_plot_profile(full_profile, X, i, zone, crater_id, swirl_on_or_off,
                                               alt_20)

            crater_floor[i] = idx1
            crater_floor[i + 18] = idx2

        # If morphology is unknown, open a Tkinter dialog to get further user input
        if crater_morph == "Unknown":

            def on_yes():
                # Show additional buttons if "Yes" is pressed
                btn_bowl.pack(pady=5)
                btn_flat.pack(pady=5)
                btn_mound.pack(pady=5)

            def on_no():
                root.destroy()  # Close the window

            def on_choice(choice):
                nonlocal crater_morph
                crater_morph = choice

                if choice == "Bowl-shaped":
                    crater_floor = [0] * 36
                root.destroy()  # Close window after choice

            # Create main window
            root = tk.Tk()
            root.title("User Choice")

            # Question label
            label_question = tk.Label(root, text="Have you changed your mind about the crater morphology?",
                                      font=("Helvetica", 12))
            label_question.pack(pady=15)

            # Yes and No buttons
            btn_yes = tk.Button(root, text="Yes", width=20, command=on_yes)
            btn_yes.pack(pady=5)

            btn_no = tk.Button(root, text="No", width=20, command=on_no)
            btn_no.pack(pady=5)

            # Additional morphology choice buttons (initially hidden)
            btn_bowl = tk.Button(root, text="Bowl-shaped", width=20, command=lambda: on_choice("Bowl-shaped"))
            btn_flat = tk.Button(root, text="Flat-floored", width=20, command=lambda: on_choice("Flat-floored"))
            btn_mound = tk.Button(root, text="With a mound", width=20, command=lambda: on_choice("With a mound"))

            root.mainloop()

    return crater_floor, crater_morph



def main(demi_profiles_value, demi_profiles_coords_relatives, pixel_size_tb, swirl_on_or_off, zone, crater_id,
         no_data_value, depth, min_val):
    '''
    This function plots and saves profiles, estimates the crater floor for each profile,
    and computes two inner points near 20% and 80% of the total depth per profile.

    Parameters:
    -----------
    demi_profiles_value: list
        Contains elevation values for each point in each semi-profile.

    demi_profiles_coords_relatives: list
        Contains the relative coordinates of each point in each semi-profile.

    pixel_size_tb: int
        The size of each pixel on the terrain (spatial resolution).

    swirl_on_or_off: str
        Indicates whether the crater is on or off a swirl feature.

    zone: str
        Indicates the study zone of the crater (values can be '1' to '7').

    crater_id: int
        The ID of the crater being studied.

    no_data_value: ???
        Value representing no data in the profiles.

    depth: list
        Contains the depth values for each profile.

    min_val: float
        Elevation of the lowest point of the crater.

    Returns:
    --------
    crater_floor: list
        Contains points selected to delimit the crater floor.

    points_inner_20: list
        Contains coordinates and elevation of the inner points selected.

    index_inner_20: list
        Contains the indices in the semi-profile of all inner points selected.

    crater_morph: str
        Morphology classification of the crater.
    '''

    points_inner_20, index_inner_20 = process_all_inner_points(
        demi_profiles_coords_relatives, demi_profiles_value, no_data_value, depth, min_val
    )

    crater_floor, crater_morph = process_profiles_and_plot(
        demi_profiles_value, demi_profiles_coords_relatives, pixel_size_tb, points_inner_20,
        zone, crater_id, swirl_on_or_off
    )

    return crater_floor, points_inner_20, index_inner_20, crater_morph



def visualisation3d(masked_image, crater_id, zone, swirl_on_or_off):
    '''
    This function plots and saves a 3D model of the studied crater.

    Parameters:
    -----------
    masked_image: array-like
        The input masked image containing elevation data of the crater.
        Expected shape: (1, rows, cols)

    crater_id: int
        ID of the crater being studied.

    zone: str
        Specifies the study zone of the crater (can be one of '1', '2', '3', '4', '5', '6', or '7').

    swirl_on_or_off: str
        Indicates whether the crater is located on a swirl feature or not (e.g., 'on' or 'off').

    Returns:
    --------
    None
    '''
    masked_band = masked_image[0]  # masked_image shape is expected to be (1, rows, cols)

    # Create grid coordinates X and Y based on the shape of the masked band
    rows, cols = masked_band.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # 3D visualization
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, masked_band, cmap='terrain', linewidth=0, antialiased=False)

    ax.set_title(f'3D Visualization of Crater {crater_id} in RG{zone}')

    # Construct the path to save the figure
    path = f'results/RG{zone}/profiles/{swirl_on_or_off}/{crater_id}'
    save_path = os.path.join(path, f'3D_Representation_{crater_id}.png')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure and close plot to free memory
    plt.savefig(save_path)
    # plt.show()  # Uncomment to display plot interactively
    plt.close()
