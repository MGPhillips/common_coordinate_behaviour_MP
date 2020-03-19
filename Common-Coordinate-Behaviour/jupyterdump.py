import nptdms
import pickle
import os
from os import walk
import h5py
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import math
from nptdms import TdmsFile
import numpy as np
from shutil import copyfile
import subprocess
import matplotlib.animation as animation
from pylab import *
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from scipy.stats import kde
import matplotlib.colors as colors


dist = 20 # distance cutoff for definition of 'in nest'
pre_stim_window = 300 # in frames

tdms_base = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\longrange\\data\\dark'
folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\analysis\\big_arena_data\\big_arena_analysis'
#folder = '/Users/phillipsmg/Dropbox (UCL - SWC)/big_Arena/analysis/big_arena_data/big_arena_analysis'
#folder = 'C:\\Users\\phillipsmg\\Dropbox (UCL - SWC)\\big_Arena\\analysis\\big_arena_data\\big_arena_analysis'

to_exclude = ['180402_longrange_nowall_2a', '180402_longrange_nowall_3a']

DLC_network = 'DeepCut_resnet50_DWMNov25shuffle1_700000'

shelter_locations = {
    '180329_longrange_nowall_1a': [230, 450],
    '180329_longrange_nowall_2a': [20, 230],
    '180402_longrange_nowall_1a': [230, 450],
    '190215_longrange_us_445_1a': [140, 75],
    '190215_longrange_us_445_1b': [140, 75],
    '190215_longrange_us_445_2a': [130, 80],
    '190215_longrange_us_445_2b': [420, 390],
    '190215_longrange_us_445_3a': [90 , 370],
    '190220_longrange_us_dark_445_3a': [110, 75],
    '190220_longrange_us_dark_445_3b': [110, 75],
    '190220_longrange_us_dark_445_5a': [400, 70],
    '190220_longrange_us_dark_445_5b': [400, 70],
    '190220_longrange_us_dark_445_5c': [400, 70],
    '190221_dwm_dark_us_445_1a': [410, 400],
    '190221_dwm_dark_us_445_2a': [60, 60],
    '190221_dwm_dark_us_445_2b': [70, 70],
    '190221_dwm_dark_us_445_4a': [360, 70],
    '190221_dwm_dark_us_445_4b': [360, 70],
    '190221_dwm_dark_us_445_4c': [360, 70]}


def get_filetype_paths(filetype, base):
    # Takes a desired filetype and a base directory and returns a list of their locations
    # within it's subdirectories

    # Set up file location list
    fpath = []
    f = []

    # Search through folders
    for (dirpath, dirnames, filenames) in walk(base):

        # if a file is detected, enter detection of its type append to list
        if filenames:
            for file in filenames:

                # If filetype matches the desired, append to list
                if file.endswith(filetype):
                    fpath.append(dirpath + '/' + file)  #
                    f.append('/' + file)

    return fpath, f


def get_tdms_indexes(path):
    indices = []

    tdms_file = TdmsFile(path)

    audio_completed = False
    visual_completed = False
    audio_keys = ['Audio Stimulation', 'Audio Stimulis']
    visual_keys = ['Visual Stimulation', 'Visual Stimulis']

    for ind_type in tdms_file.groups():
        if not audio_completed and ind_type in audio_keys:

            if ind_type == 'Audio Stimulation':
                indices = indices + [
                    x for x in np.where(tdms_file.group_channels(ind_type)[1].data != 0)[0]]

                audio_completed = True

            if ind_type == 'Audio Stimulis':
                for obj in tdms_file.group_channels(ind_type):
                    string = obj.channel.split('-')[0]

                    indices.append(int(string.replace(" ", "")))

                audio_completed = True

        if not visual_completed and ind_type in visual_keys:

            if ind_type == 'Visual Stimulis':
                for obj in tdms_file.group_channels(ind_type):
                    string = obj.channel.split('-')[0]

                    indices.append(int(string.replace(" ", "")))

                visual_completed = True

            if ind_type == 'Visual Stimulation':
                indices = indices + [
                    x for x in np.where(tdms_file.group_channels(ind_type)[1].data != 0)[0]]

                visual_completed = True

    return indices


def detect_jump(current, prev, threshold):
    jump = current - prev
    if abs(jump) > threshold:
        print('THRESHOLD EXCEEDED. CURRENT = {}, PREV = {}'.format(current, prev))
        current = prev

    return current


def zeros_fun(arr):
    # Create an array that is 1 where arr is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    # Runs start and end where absdiff is 1.
    zero_runs = np.where(absdiff == 1)[0].reshape(-1, 2)

    # Get sequences of zero coordinates and set first run equal to 1
    if len(zero_runs) != 0:

        start_zero_index1, start_zero_index2 = zero_runs[0][0], zero_runs[0][1]
        arr[start_zero_index1:start_zero_index2] = 1

        # Delete elements from zero_runs that are no longer zero
        np.delete(zero_runs, 0, 0)

        # Replace zeros
        for i in range(len(zero_runs)):
            if zero_runs[i][1] > (len(arr) - 1):
                return arr

            else:
                # Get indicies of the runs of zeros
                index1, index2 = zero_runs[i][0], zero_runs[i][1]

                # Get values of nearest non-zero values
                val1, val2 = arr[index1], arr[index2]

                # Get the length of the run
                len_run = index2 - index1

                # Smooth between two most recent non-zero points
                arr[index1:index2] = np.linspace(val1, val2, len_run)

    return arr


def fix_tracking(x, y, zeros_fun, distance_jump_limit):
    x = zeros_fun(x)
    y = zeros_fun(y)

    # Create array of distance travelled
    dist = np.zeros([len(x), 1])
    speed = np.zeros([len(x), 1])

    # Iterate through x
    for i in range(len(x)):
        if i < len(x) - 1:
            # Get distance between points
            dist[i] = (((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2) ** .5)

            if dist[i] > distance_jump_limit:
                x[i + 1], y[i + 1] = x[i], y[i]

    return x, y

h5_directories, filenames = get_filetype_paths('.h5', folder)
tdms_directories, tdms_filenames = get_filetype_paths('.tdms', tdms_base)

data_dict = {}

for i, file in enumerate(h5_directories):

    print(file)

    if DLC_network in file:  # and file[49:59] not in to_exclude: #in shelter_locations:

        with h5py.File(file, 'r') as f:

            if file[76:-52] in to_exclude:
                continue

            data_dict[file[76:-52]] = {}

            traj_x = []
            traj_y = []
            head_x = []
            head_y = []
            tail_x = []
            tail_y = []

            for j, data in f['df_with_missing']['table']:
                traj_x.append(data[3])
                traj_y.append(data[4])

                head_x.append(data[0])
                head_y.append(data[1])

                tail_x.append(data[6])
                tail_y.append(data[7])

            traj_x = np.asarray(traj_x)
            traj_y = np.asarray(traj_y)

            head_x, head_y = np.asarray(head_x), np.asarray(head_y)
            tail_x, tail_y = np.asarray(tail_x), np.asarray(tail_y)

            fix_tracking(traj_x, traj_y, zeros_fun, 50)

            data_dict[file[76:-52]]['x'], data_dict[file[76:-52]]['y'] = traj_x, traj_y
            data_dict[file[76:-52]]['head_x'], data_dict[file[76:-52]]['head_y'] = head_x, head_y
            data_dict[file[76:-52]]['tail_x'], data_dict[file[76:-52]]['tail_y'] = tail_x, tail_y

            for tdms_dir in tdms_directories:
                if file[76:-52] in tdms_dir:
                    data_dict[file[76:-52]]['stimulus_indices'] = get_tdms_indexes(tdms_dir)

data_df = pd.DataFrame.from_dict(data_dict, orient='index')


def get_speed(row):
    traj_x, traj_y = row['x'], row['y']
    dx, dy = (traj_x[1:] - traj_x[:-1], traj_y[1:] - traj_y[:-1])
    distance_travelled = np.sqrt(dx ** 2 + dy ** 2)

    return distance_travelled


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def correct_orientation(body_x, body_y, head_x, head_y, tail_x, tail_y):
    current_vector = [body_x - body[i - 1], body_y[i] - body_y[i - 1]]
    body_to_head_vector = [head_x[i] - body_x[i], head_y[i] - body_y[i]]
    body_to_tail_vector = [tail_x[i] - body_x[i], tail_y[i] - body_y[i]]

    head_angle = angle_between(current_vector, body_to_head_vector)
    tail_angle = angle_between(current_vector, tail_to_head_vector)

    # if tail_angle < head_angle:


def get_start_position(row):
    start_positions = []

    for ind in row['stimulus_indices']:

        # print(ind)

        if ind > len(row['x']):
            # print('IND out of range')
            return start_positions

        start_positions.append([row['x'][ind], row['y'][ind]])

    return start_positions


def get_start_distance(row):
    start_distances = []

    for ind in row['stimulus_indices']:

        if ind > len(row['x']):
            print('IND out of range', row.name, ind)
            return start_distances

        start_distances.append(row['shelter_distance'][ind])

    return start_distances


def get_angle(row):
    x, y = row['x'], row['y']
    nest_coord = [0, 0]
    angles = np.arctan2(y - nest_coord[1], x - nest_coord[0])
    # coord_2[1] - coord_1[1], coord_2[0] - coord_1[0]
    return angles


def get_shelter_distance(row):
    dx, dy = (row['x'] - row['shelter_position'][0], row['y'] - row['shelter_position'][1])
    shelter_distances = np.sqrt(dx ** 2 + dy ** 2)

    return shelter_distances


def add_shelter_location(row):
    shelter_position = [0, 0]
    exp = row.name
    if exp in shelter_locations:
        shelter_position = shelter_locations[exp]

    return shelter_position


def convert_coordinate_space(row):
    dist = 50
    conv_xy = []
    # start_distance = shelter
    # print('START ANGLE =')
    print(row.name)
    for start_frame in row['stimulus_indices']:

        if start_frame > len(row['angles']):
            return conv_xy
        print('start frame =', start_frame)
        print(np.where(row['shelter_distance'][start_frame:-1] < dist))
        if len(np.where(row['shelter_distance'][start_frame:-1] < dist)[0]) == 0:
            print('DID NOT RETURN TO SHELTER')
            end_frame = -1
        elif len(np.where(row['shelter_distance'][start_frame:-1] < dist)[0]) != 0:
            end_frame = start_frame + np.where(row['shelter_distance'][start_frame:-1] < dist)[0][0]

        print('end_frame =', end_frame)
        start_angle = row['angles'][start_frame]
        # print('START ANGLE =', start_angle)
        converted_x = row['shelter_distance'][start_frame:end_frame] * np.cos(
            np.pi / 2 + row['angles'][start_frame:end_frame] - start_angle)
        converted_y = row['shelter_distance'][start_frame:end_frame] * np.sin(
            np.pi / 2 + row['angles'][start_frame:end_frame] - start_angle)

        conv_xy.append([converted_x, converted_y])
    return conv_xy


def get_distance_travelled_before_flight(row):
    distances = []
    for ind in row['stimulus_indices']:

        print(ind)
        if len(np.where(row['shelter_distance'][0:ind] < 50)[0]) == 0:
            continue
        prev_nest_frame = np.where(row['shelter_distance'][0:ind] < 50)[0][-1]

        distance_travelled_since_nest = sum(row['shelter_distance'][prev_nest_frame:ind])

        distances.append(distance_travelled_since_nest)

    return distances


def get_flight_data(row, flight_dict, datatype, start_ind, end_ind):
    flight_dict[datatype] = row[datatype][start_ind:end_ind]
    return flight_dict


def populate_flight_dict(row, flight_dict):
    dist = 20  # distance cutoff for definition of 'in nest'
    pre_flight_window = 300  # in frames

    row_name = row.name
    print(row_name)

    experiment_name = row_name[:-1]
    experiment_subvid = row_name[-1]

    for i, ind in enumerate(row['stimulus_indices']):
        global_index = str(len(flight_dict))
        flight_dict[global_index] = {}
        flight_dict[global_index]['experiment_name'] = experiment_name
        flight_dict[global_index]['subvid'] = experiment_subvid
        flight_dict[global_index]['stimulus_index'] = ind

        trial_num_count = 0
        for g_ind in flight_dict:
            if flight_dict[g_ind]['experiment_name'] == experiment_name:
                trial_num_count += 1

        flight_dict[global_index]['trial_num'] = trial_num_count
        flight_start_index = ind - pre_flight_window

        if len(np.where(row['shelter_distance'][ind:-1] < dist)[0]) == 0:
            flight_end_index = -1
        elif len(np.where(row['shelter_distance'][ind:-1] < dist)[0]) != 0:
            flight_end_index = ind + np.where(row['shelter_distance'][ind:-1] < dist)[0][0]

        data_to_move = ['x', 'y', 'head_x', 'head_y', 'tail_x', 'tail_y',
                        'angles', 'distance_per_frame', 'shelter_distance']

        for dtype in data_to_move:
            flight_dict[global_index] = get_flight_data(
                row, flight_dict[global_index], dtype, flight_start_index, flight_end_index)

        if len(row['conv_xy']) - 1 >= i:
            print('Entering CONV_XY AT ', i)
            if len(row['conv_xy']) > 0:
                print(i)
                flight_dict[global_index]['conv_xy'] = row['conv_xy'][i]

            elif len(row['conv_xy']) == 0:
                flight_dict[global_index]['conv_xy'] = []

        try:
            flight_dict[global_index]['distance_travelled_before'] = row['distance_travelled_before'][i]
        except:
            flight_dict[global_index]['distance_travelled_before'] = NaN

    return flight_dict

data_df['distance_per_frame'] = data_df.apply(get_speed, axis=1)
data_df['start_position'] = data_df.apply(get_start_position, axis=1)
data_df['angles'] = data_df.apply(get_angle, axis=1)
data_df['shelter_position'] = data_df.apply(add_shelter_location, axis=1)
data_df['shelter_distance'] = data_df.apply(get_shelter_distance, axis=1)
data_df['start_distance'] = data_df.apply(get_start_distance, axis=1)
data_df['conv_xy'] = data_df.apply(convert_coordinate_space, axis=1)
data_df['distance_travelled_before'] = data_df.apply(get_distance_travelled_before_flight, axis=1)
#flight_dict = data_df.apply(populate_flight_dict, args=(flight_dict,))

flight_dict = {}

flight_dict = data_df.apply(populate_flight_dict, args=(flight_dict,), axis=1)

flight_df = pd.DataFrame.from_dict(flight_dict[0], orient='index')

def smooth_array(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

for exp in data_df.index:
    print(exp)
#['stimulus_indices']:
    stim_ind = []
    for i in data_df['stimulus_indices'][exp]:
        if type(i) == int:
            stim_ind.append(i)
    data_df['stimulus_indices'][exp] = stim_ind
    #print(exp)


def heatmap_for_nest_location(data_df_row, nest_dict):
    fig, ax = plt.subplots()

    session_x, session_y = np.array([]), np.array([])

    for i, d in enumerate(data_df_rows):
        session_x = np.concatenate((session_x, data_df_rows[i]['x'][0]))
        session_y = np.concatenate((session_y, data_df_rows[i]['y'][0]))

    data = np.vstack([list(session_x), list(session_y)]).T

    x, y = data.T

    nbins = 100

    k = kde.gaussian_kde(data.T)

    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    zi_shape = zi.reshape(xi.shape)
    nest_loc = np.where(zi_shape == zi_shape.max())
    nest_loc_scaling_x, nest_loc_scaling_y = x.max() / 100, y.max() / 100
    nest_loc = [nest_loc[0][0] * nest_loc_scaling_x, nest_loc[1][0] * nest_loc_scaling_y]
    print('Nest location: ', nest_loc)

    ax.set_title('Heatmap of exploration for nest location ID')
    pcm = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), norm=colors.LogNorm(vmin=zi.min(), vmax=zi.max()),
                        shading='gouraud', cmap=plt.cm.gist_gray)  # cmap=plt.cm.PuBu_r

    fig.colorbar(pcm, ax=ax, extend='max')

    ax.scatter(nest_loc[0], nest_loc[1], color='magenta', s=50)

    exp_name = data_df_rows[0].index[0][0:-1]
    print('Appending...', )
    nest_dict[exp_name] = nest_loc

    return nest_dict


completed_indices = []

zis = []

nest_dict = {}
http: // localhost: 8888 / notebooks / Desktop / BigArenaAnalysis % 20.
ipynb  #
for index in data_df.index:

    if index[0:-1] in completed_indices:
        continue

    print(index[0:-1])

    completed_indices.append(index[0:-1])

    indices = [x for x in data_df.index if index[0:-1] in x]
    data_df_rows = []

    for i in indices:
        data_df_rows.append(data_df[data_df.index == i])

    nest_dict = heatmap_for_nest_location(data_df_rows, nest_dict)


def plot_heatmap(data_df_rows, flight_df_rows):
    session_x, session_y = np.array([]), np.array([])

    for i, d in enumerate(data_df_rows):
        session_x = np.concatenate((session_x, data_df_rows[i]['x'][0]))
        session_y = np.concatenate((session_y, data_df_rows[i]['y'][0]))

    data = np.vstack([list(session_x), list(session_y)]).T

    x, y = data.T

    ##################################################################
    # HEATMAP + FLIGHTS  # NORMALISED FLIGHTS # FLIGHT SPEED HEATMAP #
    ##################################################################

    nbins = 100
    # fig, axes = plt.subplots(figsize=(26, 20))

    fig = plt.figure(figsize=(30, 30))
    ax0 = plt.subplot2grid((30, 30), (0, 0), colspan=25, rowspan=20)
    ax1 = plt.subplot2grid((30, 30), (0, 25), colspan=9, rowspan=20)
    ax2 = plt.subplot2grid((30, 30), (20, 0), colspan=19, rowspan=10)
    ax3 = plt.subplot2grid((30, 30), (20, 19), colspan=1, rowspan=10)
    ax4 = plt.subplot2grid((30, 30), (20, 20), colspan=10, rowspan=10)

    k = kde.gaussian_kde(data.T)

    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]

    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    ax0.set_title('Heatmap of exploration with trajectories overlaid')
    pcm = ax0.pcolormesh(xi, yi, zi.reshape(xi.shape), norm=colors.LogNorm(vmin=zi.min(), vmax=zi.max()),
                         shading='gouraud', cmap=plt.cm.gist_gray)  # cmap=plt.cm.PuBu_r

    fig.colorbar(pcm, ax=ax0, extend='max')

    for i, d in enumerate(flight_df_rows.index):

        ax0.scatter(flight_df_rows['x'][i], flight_df_rows['y'][i], s=20, alpha=0.5)  # color='magenta',

        try:
            ax1.scatter(flight_df_rows['conv_xy'][i][0], flight_df_rows['conv_xy'][i][1], s=10, alpha=0.5)

        except:
            print('conv_xy = nan')

    ax1.set_xlim([-500, 500])
    ax1.set_ylim([0, 600])

    cmap = plt.cm.hot

    rankedData = pd.Series([])
    max_flight_len = 30 * 12
    longest_flight_len = 0
    max_speed = 0

    for i in flight_df_rows.index:

        if len(flight_df_rows['distance_per_frame'][i]) == 0:
            continue

        if len(flight_df_rows['distance_per_frame'][i]) > longest_flight_len:
            longest_flight_len = len(flight_df_rows['distance_per_frame'][i][300 - 90:])

        if flight_df_rows['distance_per_frame'][i].max() > max_speed:
            max_speed = flight_df_rows['distance_per_frame'][i].max()

    if longest_flight_len > max_flight_len:
        longest_flight_len = max_flight_len

    for i in flight_df_rows.index:

        speed = flight_df_rows['distance_per_frame'][i][300 - 90:]

        pad_width = longest_flight_len - len(speed)

        if pad_width < 0:
            pad_width = 0

        # print('Speed:', speed)
        speed = np.pad(speed, (0, pad_width), 'constant', constant_values=(0, 0))

        if len(speed) == 0:
            print('Speed empty')
            continue

        speed = smooth_array(speed, 5)

        if len(speed) > longest_flight_len:
            speed = speed[0:longest_flight_len]

        if len(rankedData) == 0:
            rankedData = speed

        else:

            rankedData = np.vstack([rankedData, speed])

    vmax, vmin = max_speed, -0
    ax2.matshow(rankedData, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none', aspect='auto')
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    ax2ymin, ax2ymax = ax2.get_ylim()
    ax2.vlines([90], ax2ymin, ax2ymax, linestyles='dotted', color='w', linewidth=5)

    # ax2.axis('off')
    ax2.linewidth = 50.0
    cbar = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, spacing='proportional')
    cbar.set_label('cm/s', rotation=0, labelpad=10, y=1, color='w')
    # cbar.set_ticklabels(['Low', 'Medium'])# horizontal colorbar
    mpl.rcParams.update({'xtick.color': 'black', 'font.size': 15})  # 'font.size': 5,

    for i in flight_df_rows.index:
        try:
            start_distance = flight_df_rows['shelter_distance'][i][300]
            distance_in_flight = sum(flight_df_rows['distance_per_frame'][i][300:])

            ax4.scatter(start_distance, distance_in_flight, c='navy')
        except:
            print('Ind out of range')

    ax4.set_ylim(ymin=0)
    ax4.set_xlim(xmin=0)
    ax4.set_title('Flight Accuracy Overview')
    ax4.set_ylabel('Distance travelled in flight')
    ax4.set_xlabel('Start distance to nest')

    plt.tight_layout()
    figname = 'E:\\big_arena_analysis\\sess_summary_{}'.format(flight_df_rows['experiment_name'][0])
    fig.savefig(figname, dpi=fig.dpi)


completed_indices = []

for index in data_df.index:

    if index[0:-1] in completed_indices:
        continue

    print(index[0:-1])

    completed_indices.append(index[0:-1])

    indices = [x for x in data_df.index if index[0:-1] in x]
    data_df_rows = []

    for i in indices:
        data_df_rows.append(data_df[data_df.index == i])

    plot_heatmap(data_df_rows, flight_df[flight_df['experiment_name'] == index[0:-1]])

fig, ax = plt.subplots(flight_df['trial_num'].max(), 1, figsize=(10, 10 * flight_df['trial_num'].max()))

for i in range(flight_df['trial_num'].max()):
    if i == 0:
        continue
    exps = flight_df[flight_df['trial_num'] == i].index

    # fig, ax = plt.subplots()

    for exp in exps:
        try:
            if len(flight_df['conv_xy'][exp][0]) > 450:
                continue
            ax[i].scatter(flight_df['conv_xy'][exp][0], flight_df['conv_xy'][exp][1], s=1.0, alpha=0.7)
            ax[i].set_xlim([-500, 500])
            ax[i].set_ylim([0, 600])

        except:
            print(exp, 'could not be plotted')

fig_name = 'E:\\big_arena_analysis\\sorted_by_trial_number_{}'.format(i)
# fig1_name = 'E:\\big_arena_analysis\\traj_normalised_plot_joint'
fig.savefig(fig_name, dpi=fig.dpi)

fig, ax = plt.subplots()

for exp in flight_df.index:
    # print(exp)
    flight_dist_trav = sum(flight_df['distance_per_frame'][exp])

    if 'dark' in flight_df['experiment_name'][exp]:
        print('DARK')
        try:
            shelter_dist_trav_ratio = flight_dist_trav / flight_df['shelter_distance'][exp][300]

            ax.scatter(flight_df['trial_num'][exp], shelter_dist_trav_ratio, c='blue', alpha=0.5)
        except:
            print('EXCEPTION')
            continue

    if not 'dark' in flight_df['experiment_name'][exp]:
        print('NOT DARK')
        try:
            shelter_dist_trav_ratio = flight_dist_trav / flight_df['shelter_distance'][exp][300]

            ax.scatter(flight_df['trial_num'][exp], shelter_dist_trav_ratio, c='magenta', alpha=0.5)
        except:
            print('EXCEPTION')
            continue

ax.set_ylim([0, 6])
ax.set_xlim([0, 10.5])

plt.show()

f = plt.figure(frameon=False, figsize=(4, 5), dpi=100)
canvas_width, canvas_height = f.canvas.get_width_height()
ax = f.add_axes([0, 0, 1, 1])
ax.axis('off')


# def produce_tracking_video(data, filepath, ):

def update(frame):
    # ax.clear()
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 500])
    i, key = h5_f['df_with_missing']['table'][frame]  # ['title']:

    ax.scatter(key[6], key[7], c='magenta', alpha=0.5, s=1.0)
    ax.scatter(key[0], key[1], c='blue', alpha=0.5, s=1.0)
    ax.scatter(key[3], key[4], c='black', alpha=0.5, s=1.0)

    # your matplotlib code goes here


# Open an ffmpeg process
outf = 'C:\\Users\\matthewp.W221N\\Desktop\\ffmpeg.mp4'
cmdstring = ('ffmpeg',
             '-y', '-r', '30',  # overwrite, 30fps
             '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
             '-pix_fmt', 'argb',  # format
             '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
             '-vcodec', 'mpeg4', outf)  # output encoding
p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

# Draw 1000 frames and write to the pipe
for frame in range(100):
    if frame % 1000 == 0:
        print(frame)

    frame = frame + 3000
    # draw the frame
    update(frame)
    plt.draw()

    # extract the image as an ARGB string
    string = f.canvas.tostring_argb()

    # write to pipe
    p.stdin.write(string)

# Finish up
p.communicate()

f = plt.figure(frameon=False, figsize=(4, 5), dpi=100)
canvas_width, canvas_height = f.canvas.get_width_height()
ax = f.add_axes([0, 0, 1, 1])
ax.axis('off')


# def produce_tracking_video(data, filepath, ):
def update(frame):
    for i, d in enumerate(zipped):
        i_ind, j_ind = zipped[i][0]

        print('Frame:', frame)
        ax.set_xlim([0, 500])
        ax.set_ylim([0, 500])
        print(data_df['conv_xy'][i_ind][j_ind][0])
        print(data_df['conv_xy'][i_ind][j_ind][1])
        ax.scatter(data_df['conv_xy'][i_ind][j_ind][0][frame],
                   data_df['conv_xy'][i_ind][j_ind][1][frame],
                   alpha=0.3, s=1.0)


# def update(frame):
# ax.clear()

#    i, key = h5_f['df_with_missing']['table'][frame] #['title']:

#    for i, exp in enumerate(data_df['conv_xy'].index):
#        for j in data_df['conv_xy'][exp]:
#            for k in j:
#                print(frame)
#                print(k)
#                ax.scatter(k[0][frame], k[1][frame], alpha=0.3,s=5.0)


# Open an ffmpeg process
outf = 'C:\\Users\\matthewp.W221N\\Desktop\\ffmpeg.mp4'
cmdstring = ('ffmpeg',
             '-y', '-r', '30',  # overwrite, 30fps
             '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
             '-pix_fmt', 'argb',  # format
             '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
             '-vcodec', 'mpeg4', outf)  # output encoding
p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

# Draw 1000 frames and write to the pipe
for frame in range(30000):
    if frame % 1000 == 0:
        print(frame)

    # frame = frame + 3000
    # draw the frame
    update(frame)
    plt.draw()

    # extract the image as an ARGB string
    string = f.canvas.tostring_argb()

    # write to pipe
    p.stdin.write(string)

# Finish up
p.communicate()


for exp in data_df.index:
    #if exp[0:2] in shelter_locations:
    fig,ax = plt.subplots(figsize=(6,6))
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    plt.plot(data_df['x'][exp], data_df['y'][exp])
    #plt.scatter(shelter_locations[exp[0:2]][0], shelter_locations[exp[0:2]][1], c='red')
    fig_name = 'E:\\big_arena_analysis\\'+ exp+'-traj_plot'
    fig.savefig(fig_name, dpi=fig.dpi)
        #plt.show()
    #plt.close(fig)

for exp in data_df.index:
    fig, ax = plt.subplots(1,2, figsize=(60, 30))
    if len(data_df['stimulus_indices'][exp]) > 0:
        for ind in data_df['stimulus_indices'][exp]:
            ax[0].plot(data_df['x'][exp][ind-300:ind+300], data_df['y'][exp][ind-300:ind+300])
            ax[0].set_xlim([0,500])
            ax[0].set_ylim([0,500])
            ax[1].plot(data_df['x'][exp], data_df['y'][exp], color='blue',zorder=2)
            ax[1].scatter(data_df['shelter_position'][exp][0],
                          data_df['shelter_position'][exp][1], s=500, color='red', zorder=1)
            figname = 'E:\\big_arena_analysis\\nest_pos\\' + exp
            fig.savefig(figname, dpi=fig.dpi)

ind_list = []
distance_list = []
for i, dist_list in enumerate(data_df['start_distance']):
    for j, dist in enumerate(dist_list):
        ind_list.append([i, j])
        distance_list.append(dist)

zipped = zip(ind_list, distance_list)
zipped = sorted(zipped, key=lambda x: x[1])

n_exps = len(zipped)  # len(data_df.index)
n_rows, n_cols = int(math.ceil(n_exps / 5)), 5
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 10, n_rows * 10))
data_df.sort_values(by=['start_distance'])

x_max = 0

for i, d in enumerate(zipped):
    i_ind, j_ind = zipped[i][0]

    ax.scatter(
        data_df['conv_xy'][i_ind][j_ind][0],  # [1439:1400+shelter_entered[0][0]],
        data_df['conv_xy'][i_ind][j_ind][1],  # [1439:1400+shelter_entered[0][0]],
        alpha=1.0, s=5.0)

    ax.set_xlim([-400, 400])
    ax.set_ylim([0, 600])
    title = (str(i_ind) + '/' + str(j_ind))
    ax.set_title(title)

fig_name = 'E:\\big_arena_analysis\\traj_normalised_plot'
# fig1_name = 'E:\\big_arena_analysis\\traj_normalised_plot_joint'
fig.savefig(fig_name, dpi=fig.dpi)
# fig1.savefig(fig1_name, dpi=fig1.dpi)


n_exps = len(data_df.index)
n_rows, n_cols = int(math.ceil(n_exps / 5)), 5
fig, ax = plt.subplots()
# data_df.sort_values(by=['start_distance'])

for i, d in enumerate(zipped):
    i_ind, j_ind = zipped[i][0]

    # for i, exp in enumerate(data_df.sort_values(by='start_distance').index):

    # if count ==3:
    # plt.plot(data_df['angles'][exp])
    # count+=1
    # count+=1
    # shelter_entered = np.where(data_df['shelter_distance'][exp][1400:-1] < 20)
    # colors = [cm.hot(x) for x in cm.hot(data_df['distance_per_frame'][exp][1439:1400+shelter_entered[0][0]])]
    # print(exp, i)
    # for j in data_df['conv_xy'][exp]:

    ax.scatter(data_df['conv_xy'][i_ind][j_ind][0],
               data_df['conv_xy'][i_ind][j_ind][1],
               alpha=0.3, s=1.0)

    # data_df['conv_xy'][exp][0][1439:1400+shelter_entered[0][0]],
    # data_df['conv_xy'][exp][1][1439:1400+shelter_entered[0][0]],

ax.set_xlim([-500, 500])
ax.set_ylim([0, 600])

fig_name = 'E:\\big_arena_analysis\\traj_normalised_plot'
# fig1_name = 'E:\\big_arena_analysis\\traj_normalised_plot_joint'
fig.savefig(fig_name, dpi=fig.dpi)
# fig1.savefig(fig1_name, dpi=fig1.dpi)

n_exps = len(data_df.index)
n_rows, n_cols = int(math.ceil(n_exps / 5)), 5
fig, ax = plt.subplots(2, 1, figsize=(10, 20))
# data_df.sort_values(by=['start_distance'])


for i, exp in enumerate(data_df['conv_xy'].index):

    if 'dark' in exp:
        for j in data_df['conv_xy'][exp]:
            ax[0].scatter(j[0],
                          j[1],
                          alpha=0.3, s=5.0)

    if 'dark' not in exp:
        for j in data_df['conv_xy'][exp]:
            ax[1].scatter(j[0],
                          j[1],
                          alpha=0.3, s=5.0)

ax[0].set_xlim([-500, 500])
ax[0].set_ylim([0, 600])
ax[1].set_xlim([-500, 500])
ax[1].set_ylim([0, 600])

ax[0].set_title('Dark Trials')
ax[1].set_title('Light Trials')

fig_name = 'E:\\big_arena_analysis\\traj_normalised_plot_split_DarkLight'
# fig1_name = 'E:\\big_arena_analysis\\traj_normalised_plot_joint'
fig.savefig(fig_name, dpi=fig.dpi)
# fig1.savefig(fig1_name, dpi=fig1.dpi)


dist = 20
dist_trav_before = []
dist_trav_before_dark = []
dist_ratio_dark = []

dist_trav_before_light = []
dist_ratio_light = []

start_dist_normalised = []
half_dist = []
dist_ratio_first_half = []
dist_ratio_second_half = []

### PLOT DISTANCE TRAVELLED IN FLIGHT / START DISTANCE VERSUS DISTANCE TRAVELLED BEFORE FLIGHT ###

for exp in data_df.index:
    if 'dark' in exp:
        for i, start_frame in enumerate(data_df['stimulus_indices'][exp]):

            if start_frame > len(data_df['x'][exp]):
                continue

            if len(np.where(data_df['shelter_distance'][exp][0:start_frame] < dist)[0]) == 0:
                print('ERROR: HAD NOT BEEN IN SHELTER?')

                prev_shelter_frame = 0

            elif len(np.where(data_df['shelter_distance'][exp][0:start_frame] < dist)[0]) != 0:
                prev_shelter_frame = np.where(data_df['shelter_distance'][exp][0:start_frame] < dist)[0][-1]

            if len(np.where(data_df['shelter_distance'][exp][start_frame:-1] < dist)[0]) == 0:
                print('ERROR: DID NOT RETURN TO SHELTER?')

                dist_trav_in_flight = 0

            elif len(np.where(data_df['shelter_distance'][exp][start_frame:-1] < dist)[0]) != 0:
                end_frame = start_frame + np.where(data_df['shelter_distance'][exp][start_frame:-1] < dist)[0][0]

                dist_trav_in_flight = sum(data_df['distance_per_frame'][exp][start_frame:end_frame])

                half_frame = int(start_frame + np.where(
                    data_df['shelter_distance'][exp][start_frame:-1] < dist)[0][0] / 2)

                dist_trav_first_half = sum(data_df['distance_per_frame'][exp][start_frame:half_frame])
                dist_trav_second_half = sum(data_df['distance_per_frame'][exp][half_frame:end_frame])

            dist_before_flight = sum(data_df['distance_per_frame'][exp][prev_shelter_frame:start_frame])
            dist_trav_before_dark.append(dist_before_flight)
            dist_trav_before.append(dist_before_flight)

            start_dist_normalised.append(data_df['shelter_distance'][exp][start_frame] / 2)
            half_dist.append(data_df['shelter_distance'][exp][half_frame])

            dist_ratio_first_half.append(dist_trav_first_half / data_df['shelter_distance'][exp][start_frame])
            dist_ratio_second_half.append(dist_trav_second_half / data_df['shelter_distance'][exp][start_frame])

            dist_ratio_dark.append(dist_trav_in_flight / data_df['shelter_distance'][exp][start_frame])

    if 'dark' not in exp:
        for i, start_frame in enumerate(data_df['stimulus_indices'][exp]):

            if start_frame > len(data_df['x'][exp]):
                continue

            if len(np.where(data_df['shelter_distance'][exp][0:start_frame] < dist)[0]) == 0:
                print('ERROR: HAD NOT BEEN IN SHELTER?')

                prev_shelter_frame = 0

            elif len(np.where(data_df['shelter_distance'][exp][0:start_frame] < dist)[0]) != 0:
                prev_shelter_frame = np.where(data_df['shelter_distance'][exp][0:start_frame] < dist)[0][-1]

            if len(np.where(data_df['shelter_distance'][exp][start_frame:-1] < dist)[0]) == 0:
                print('ERROR: DID NOT RETURN TO SHELTER?')

                dist_trav_in_flight = 0

            elif len(np.where(data_df['shelter_distance'][exp][start_frame:-1] < dist)[0]) != 0:
                end_frame = start_frame + np.where(data_df['shelter_distance'][exp][start_frame:-1] < dist)[0][0]

                dist_trav_in_flight = sum(data_df['distance_per_frame'][exp][start_frame:end_frame])

                half_frame = int(start_frame + np.where(
                    data_df['shelter_distance'][exp][start_frame:-1] < dist)[0][0] / 2)

                dist_trav_first_half = sum(data_df['distance_per_frame'][exp][start_frame:half_frame])
                dist_trav_second_half = sum(data_df['distance_per_frame'][exp][half_frame:end_frame])

            dist_before_flight = sum(data_df['distance_per_frame'][exp][prev_shelter_frame:start_frame])
            dist_trav_before_light.append(dist_before_flight)

            dist_ratio_light.append(dist_trav_in_flight / data_df['shelter_distance'][exp][start_frame])

            dist_trav_before.append(dist_before_flight)
            start_dist_normalised.append(data_df['shelter_distance'][exp][start_frame] / 2)
            half_dist.append(data_df['shelter_distance'][exp][half_frame])

            dist_ratio_first_half.append(dist_trav_first_half / data_df['shelter_distance'][exp][start_frame])
            dist_ratio_second_half.append(dist_trav_second_half / data_df['shelter_distance'][exp][start_frame])


fig, ax = plt.subplots()

#data_df.sort_values(by=['start_distance'])


ax.scatter(start_dist, dist_ratio_first_half, c='blue',alpha=0.5)
ax.scatter(half_dist, dist_ratio_second_half, c='magenta', alpha=0.5)
ax.set_ylim([0,5])
#ax.set_xlim([0,2000])
ax.set_ylabel('Dist travelled in flight/distance to shelter')
ax.set_xlabel('Start distance (normalised)')
plt.show()

fig, ax = plt.subplots()

dark_list = [x for x in flight_df.index if 'dark' in flight_df['experiment_name'][x]]
light_list = [x for x in flight_df.index if x not in dark_list]

dark_means = []
light_means = []

for i, distances in enumerate(flight_df['distance_per_frame']):
    for j, dist in enumerate(flight_df['distance_per_frame'][i]):
        if len(means) < j:
            means.append(dist)
            n.append(1)
        elif len(means) > j:
            means[j] += dist
            n[j] += 1

for i, data in enumerate(means):
    means[i] = means[i] / n[i]

for trial in flight_df.index:

    if trial in dark_list:

        for j, dist in enumerate(flight_df['distance_per_frame'])

        x = np.arange(len(flight_df['distance_per_frame'][trial]))
        ax.plot(x, flight_df['distance_per_frame'][trial], alpha=0.5, c='blue')

    if trial in light_list:
        x = np.arange(len(flight_df['distance_per_frame'][trial]))
        ax.plot(x, flight_df['distance_per_frame'][trial], alpha=0.5, c='magenta')

ax.set_ylim([0, 8])
ax.set_ylim([0, 8])


max_dist_len = 0

for i in flight_df['distance_per_frame']:
    if len(i) > max_dist_len:
        max_dist_len = len(i)


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def get_collected_dists(distances_per_frame):
    collected_dists = [[]]

    for i, distances in enumerate(distances_per_frame):
        for j, dist in enumerate(distances_per_frame[i]):
            if len(collected_dists) <= j:
                # print(j)
                collected_dists.append([])
                collected_dists[j].append(dist)
            elif len(collected_dists) > j:
                # print(j)
                collected_dists[j].append(dist)

    np_dists = np.array([np.array(xi) for xi in collected_dists])
    np_dists_nan = boolean_indexing(np_dists)

    return np_dists_nan


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


collected_dists = get_collected_dists(flight_df['distance_per_frame'])

means = np.nanmean(collected_dists, axis=1)
stds = np.nanstd(collected_dists, axis=1)
means, stds

# collected_dists = np.asarray(collected_dists)

dark_dists = get_collected_dists(flight_df['distance_per_frame'][dark_list])
light_dists = get_collected_dists(flight_df['distance_per_frame'][light_list])


# SMOOTHED

dark_means, light_means = np.nanmean(dark_dists, axis=1), np.nanmean(light_dists, axis=1)


dark_std = np.nanstd(dark_dists, axis=1)
light_std = np.nanstd(light_dists, axis=1)

dark_x = np.arange(len(dark_dists))
light_x = np.arange(len(light_dists))

smoothed_means_dark, smoothed_means_light =
smoothed_std_dark, smoothed_std_dark =


fig, ax = plt.subplots(2,1, figsize=(60,30))

ax[0].plot(dark_x, dark_means, color='magenta')
ax[0].fill_between(dark_x, dark_means+dark_std, dark_means-dark_std, alpha=0.5, color='magenta')

ax[1].plot(light_x, light_means, color='blue')
ax[1].fill_between(light_x, light_means+light_std, light_means-light_std, alpha=0.5, color='blue')


#ax[0].set_xlim([0,5000])
ax[0].set_ylim([0,5])

#ax[1].set_xlim([0,5000])
ax[1].set_ylim([0,5])



# SMOOTHED

dark_means, light_means = np.nanmean(dark_dists, axis=1), np.nanmean(light_dists, axis=1)


dark_std = np.nanstd(dark_dists, axis=1)
light_std = np.nanstd(light_dists, axis=1)

dark_x = np.arange(len(dark_dists))
light_x = np.arange(len(light_dists))

smoothed_means_dark, smoothed_means_light =
smoothed_std_dark, smoothed_std_dark =


fig, ax = plt.subplots(2,1, figsize=(60,30))

ax[0].plot(dark_x, dark_means, color='magenta')
ax[0].fill_between(dark_x, dark_means+dark_std, dark_means-dark_std, alpha=0.5, color='magenta')

ax[1].plot(light_x, light_means, color='blue')
ax[1].fill_between(light_x, light_means+light_std, light_means-light_std, alpha=0.5, color='blue')


#ax[0].set_xlim([0,5000])
ax[0].set_ylim([0,5])

#ax[1].set_xlim([0,5000])
ax[1].set_ylim([0,5])
