import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import re
import seaborn as sns
import random
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def load_pickle_to_pandas(path):

    return pd.read_pickle(path)

data_df = load_pickle_to_pandas(r'E:\Dropbox (UCL - SWC)\big_Arena\analysis\data_df')
flight_df = load_pickle_to_pandas(r'E:\Dropbox (UCL - SWC)\big_Arena\analysis\flight_df')

colors = {'dark': 'mediumvioletred',
          'light': 'royalblue'}


def tracking_fix_check(x, y, tracking_jump, fix_tracking, zeros_fun, save_path, fig_name):
    plt.close('all')

    fig, axes = plt.subplots(2, 1, figsize=(15, 30))

    axes[0].plot(x, y)

    fix_x, fix_y = fix_tracking(x, y, zeros_fun, tracking_jump)

    axes[1].plot(fix_x, fix_y)

    for ax in axes:
        ax.set_ylim(0, 500)
        ax.set_xlim(0, 500)

    fig.savefig(save_path + '\\' + fig_name)


tracking_jump = 30
for ind in flight_df.index:
    # tracking_fix_check(flight_df[flight_df.index==ind]['x'][0],
    #                   flight_df[flight_df.index==ind]['y'][0],
    #                   tracking_jump, fix_tracking, zeros_fun,
    #                  r'E:\Dropbox (UCL - SWC)\big_Arena\analysis\figtest\body',
    #                  ind + '_tracking_check')

    # tracking_fix_check(flight_df[flight_df.index==ind]['tail_x'][0],
    ##                   flight_df[flight_df.index==ind]['tail_y'][0],
    #                   tracking_jump, fix_tracking, zeros_fun,
    #                  r'E:\Dropbox (UCL - SWC)\big_Arena\analysis\figtest\tail',
    #                  ind + '_tracking_check')

    tracking_fix_check(flight_df[flight_df.index == ind]['x'][0],
                       flight_df[flight_df.index == ind]['y'][0],
                       tracking_jump, fix_tracking, zeros_fun,
                       r'E:\Dropbox (UCL - SWC)\big_Arena\analysis\figtest\head\flicker',
                       ind + '_tracking_check')

    tracking_fix_check(flight_df[flight_df.index == ind]['x'][0],
                       flight_df[flight_df.index == ind]['y'][0],
                       tracking_jump, fix_tracking, zeros_fun,
                       r'E:\Dropbox (UCL - SWC)\big_Arena\analysis\figtest\head\flicker',
                       ind + '_tracking_check')

    for ind in flight_df.index:
        fig, ax = plt.subplots()

        x, y = fix_tracking(flight_df[flight_df.index == ind]['x'][0],
                            flight_df[flight_df.index == ind]['y'][0],
                            zeros_fun,
                            30)

        ax.plot(x, y)

        ax.set_ylim(0, 500)
        ax.set_xlim(0, 500)

        fig.savefig(r'E:\Dropbox (UCL - SWC)\big_Arena\analysis\figtest' + '\\' + ind + '_fixed')

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


def fix_tracking(in_x, in_y, zeros_fun, distance_jump_limit):
    x = zeros_fun(in_x)
    y = zeros_fun(in_y)

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


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def head_tail_flicker_correction(body_x, body_y, head_x, head_y, tail_x, tail_y):
    head_x_list, head_y_list, tail_x_list, tail_y_list = [], [], [], []
    count_a, count_d, count_f = 0, 0, 0
    to_flip = False

    for i, bx in enumerate(body_x):

        head_x_i, head_y_i, tail_x_i, tail_y_i = head_x[i], head_y[i], tail_x[i], tail_y[i]

        if to_flip:
            head_x_i, head_y_i, tail_x_i, tail_y_i = tail_x_i, tail_y_i, head_x_i, head_y_i

        to_flip = False

        if i + 2 < len(head_x):

            head_x_i2, head_y_i2, tail_x_i2, tail_y_i2 = (
                head_x[i + 1], head_y[i + 1], tail_x[i + 1], tail_y[i + 1])

            mouse_vector = [head_x_i - tail_x_i, head_y_i - tail_y_i]
            mouse2_vector = [head_x_i2 - tail_x_i2, head_y_i2 - tail_y_i2]
            mouse2_vector_flip = [tail_x_i2 - head_x_i2, tail_y_i2 - head_y_i2]

            angle = angle_between(mouse_vector, mouse2_vector)
            flip_angle = angle_between(mouse_vector, mouse2_vector_flip)

            if angle > flip_angle:
                if count_a == 0:
                    print('Flipping first coords based on angle')

                to_flip = True
                count_a += 1

        head_x_list.append(head_x_i), head_y_list.append(head_y_i)
        tail_x_list.append(tail_x_i), tail_y_list.append(tail_y_i)

    return head_x_list, head_y_list, tail_x_list, tail_y_list


ind = '185'

hx, hy, tx, ty = head_tail_flicker_correction(

    flight_df[flight_df.index == ind]['x'][0],
    flight_df[flight_df.index == ind]['y'][0],

    flight_df[flight_df.index == ind]['head_x'][0],
    flight_df[flight_df.index == ind]['head_y'][0],

    flight_df[flight_df.index == ind]['tail_x'][0],
    flight_df[flight_df.index == ind]['tail_y'][0])

fig, ax = plt.subplots()

ax.plot(flight_df[flight_df.index == ind]['head_x'][0],
        flight_df[flight_df.index == ind]['head_y'][0],
        c='blue')

ax.plot(hx, hy, c='red')

ax.set_ylim(0, 500)
ax.set_xlim(0, 500)

fig.show()


def plot_conv_xy(ax, df, color, n_traj):
    inds = random.sample(range(0, len(df)), n_traj)
    count = 0

    for i, ind in enumerate(inds):

        # df[df.index==ind]['conv_xy'][0][0], df[df.index==ind]['conv_xy'][0][1] / conversion_factor

        index = df.index[ind]

        conversion_factor = df[df.index == index]['start_distance'][0]
        # print('Conversion factor = ', conversion_factor)

        x, y = fix_tracking(df[df.index == index]['conv_xy'][0][0], df[df.index == index]['conv_xy'][0][1],
                            zeros_fun, 20)
        y = y / conversion_factor

        if any(j > 1.1 for j in y):
            continue

        ax.plot(x, y, c=color,
                alpha=0.7)

    ax.set_xlim(-250, 250)

    return ax

colors = {'dark': 'mediumvioletred',
              'light': 'royalblue'}


fig, ax = plt.subplots(1,2, figsize=(6,6))

ax[0] = plot_conv_xy(ax[0], flight_df[(flight_df['expt_type']=='dark') &
                                      (flight_df['flight_success']=='successful')],
                     colors['dark'], 20)
ax[1] = plot_conv_xy(ax[1], flight_df[(flight_df['expt_type']=='light') &
                                      (flight_df['flight_success']=='successful')],
                     colors['light'], 20)

for axis in ax:
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['left'].set_bounds(0, 1)
    axis.set_ylim([-0.1, 1.1])

ax[1].spines['left'].set_visible(False)
ax[1].tick_params(labelleft=False)
ax[1].tick_params(left=False)

ax[0].set_ylabel('Normalised distance to shelter')

ax[0].set_title('Dark')
ax[1].set_title('Light')

fig.suptitle('Normalised trajectories', fontsize=16)

######################

from sklearn.preprocessing import StandardScaler

pca_variables = ['distance_travelled_in_flight', 'cumulative_vec_angles',
                 'max_speed', 'flight_dist_ratio_float', 'abs_speed_change_sum']

pca_df = flight_df[pca_variables]

x = pca_df.values
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
                           , columns = ['principal component 1', 'principal component 2'])

principalDf.reset_index(drop=True, inplace=True)
flight_df.reset_index(drop=True, inplace=True)

finalDf = pd.concat([principalDf, flight_df['expt_type']], axis=1)

x = pca_df.values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1', 'principal component 2'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['dark', 'light']
colors = ['mediumvioletred', 'royalblue']

for target, color in zip(targets,colors):
    indicesToKeep = flight_df[flight_df['expt_type'] == target].index.astype(int)
    print(indicesToKeep)
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 30,
              alpha=0.75)
ax.legend(targets)
ax.grid()

x = pca_df.values

pca3 = PCA(n_components=3)
X = pca3.fit_transform(x)
principalDf3 = pd.DataFrame(data = X,
                            columns = ['principal component 1', 'principal component 2', 'principal component 3'])

#centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(4, 3))


plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
#pca = decomposition.PCA(n_components=3)
#pca.fit(x)
#X = pca.transform(x)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

fig = plt.figure(1, figsize=(4, 3))

plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=20)

plt.cla()

ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

pca.explained_variance_ratio_

###################

s = set(flight_df[flight_df['expt_type'] == 'light']['experiment_name'])

sum_s = 0
for ind in s:
    print(ind)
    print(len(flight_df[flight_df['experiment_name'] == ind]))
    sum_s += len(flight_df[flight_df['experiment_name'] == ind])

print(sum_s)

print(flight_df[flight_df['experiment_type']])

flight_df['t_to_nest_float'] = flight_df['t_to_nest'].astype(float)
flight_df['flight_dist_ratio_float'] = flight_df['flight_dist_ratio'].astype(float)


def get_max_speed(row):
    max_speed = row['distance_per_frame'].max()
    if max_speed > 15:
        max_speed = 15
    return max_speed


def get_cumulative_vec_angles(row):
    cumulative_vec_angles = sum(row['vec_angles'][300:])

    return cumulative_vec_angles


def get_distance_travelled_in_flight(row):
    distance_travelled_in_flight = sum(row['distance_per_frame'][300:]) + 50

    return distance_travelled_in_flight

flight_df['max_speed'] = flight_df.apply(get_max_speed, axis=1)
flight_df['cumulative_vec_angles'] = flight_df.apply(get_cumulative_vec_angles, axis=1)
flight_df['distance_travelled_in_flight'] = flight_df.apply(get_distance_travelled_in_flight, axis=1)


plot_df = flight_df[flight_df['flight_success']=='successful'][[
    'expt_type','start_distance',
    'distance_travelled_before',
    'abs_speed_change_sum',
    'flight_dist_ratio_float',
    't_to_nest_float',
    'max_speed',
    'cumulative_vec_angles',
    'distance_travelled_in_flight']]

plot_df = plot_df.drop(index='28')

navy_magenta_palette = sns.diverging_palette(334, 255, l=50, s=80,n=2)

colors = {'dark': 'mediumvioletred',
              'light': 'royalblue'}

ax = sns.kdeplot(flight_df[flight_df['flight_success']=='successful'][
    flight_df['expt_type']=='dark']['max_speed'], shade=True, color=colors['dark'], label='dark')
ax = sns.kdeplot(flight_df[flight_df['flight_success']=='successful'][
    flight_df['expt_type']=='light']['max_speed'], shade=True, color=colors['light'], label='light')


ax.set(xlabel='Max speed (m/s)', ylabel='Density')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


colors = {'dark': 'mediumvioletred',
              'light': 'royalblue'}

ax = sns.kdeplot(flight_df[flight_df['flight_success']=='successful'][
    flight_df['expt_type']=='dark']['mean_speed'], shade=True, color=colors['dark'], label='dark')
ax = sns.kdeplot(flight_df[flight_df['flight_success']=='successful'][
    flight_df['expt_type']=='light']['mean_speed'], shade=True, color=colors['light'], label='light')


ax.set(xlabel='Mean speed (m/s)', ylabel='Density')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

colors = {'dark': 'mediumvioletred',
          'light': 'royalblue'}

dark_speeds = []
light_speeds = []

for ind in flight_df[flight_df['flight_success'] == 'successful'][flight_df['expt_type'] == 'dark'].index:
    dark_speeds.extend(flight_df[flight_df.index == ind]['distance_per_frame'][0])

for ind in flight_df[flight_df['flight_success'] == 'successful'][flight_df['expt_type'] == 'light'].index:
    light_speeds.extend(flight_df[flight_df.index == ind]['distance_per_frame'][0])

ax = sns.kdeplot(dark_speeds, shade=True, color=colors['dark'], label='dark')
ax = sns.kdeplot(light_speeds, shade=True, color=colors['light'], label='light')

ax.set_xlim(0, 10)
ax.set(xlabel='Speed (m/s)', ylabel='Density')
ax.set_title('Distribution of speeds through flights')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

to_plot = ['expt_type','mean_speed', 'max_speed']

sns.pairplot(data=flight_df[flight_df['flight_success']=='successful'][to_plot],hue="expt_type",
             palette=navy_magenta_palette,
             kind = 'reg',
             diag_kind='kde',
             diag_kws=dict(shade=True))



###################################

ax = sns.regplot(
    flight_df[flight_df['flight_success'] == 'successful'][flight_df['expt_type'] == 'light']['start_distance'],
    flight_df[flight_df['flight_success'] == 'successful'][flight_df['expt_type'] == 'light'][
        'distance_travelled_in_flight'],
    fit_reg=False, color='royalblue')

ax.plot(np.linspace(0, 700, 700), np.linspace(0, 700, 700), linestyle='--', color='gray')

tick_locations = [int(i) for i in np.arange(0, 800, 100)]
tick_values = [int(i) for i in np.arange(0, 400, 250 / 5)]

# ax.set_xticks()
# ax.set_xticklabels()
ax.figure.set_size_inches(5, 5)
sns.despine()

plt.xlim(0, 750)
plt.ylim(0, 750)
plt.xticks(tick_locations, tick_values)
plt.yticks(tick_locations, tick_values)
plt.xlabel('Start distance (cm)')
plt.ylabel('Distance travelled in flight (cm)')

fig, ax = plt.subplots()

ax.plot(np.linspace(0, 700, 700), np.linspace(0, 700, 700), linestyle='--', color='gray')
# = sns
sns.regplot(flight_df[flight_df['flight_success'] == 'successful'][flight_df['expt_type'] == 'light']['start_distance'],
            flight_df[flight_df['flight_success'] == 'successful'][flight_df['expt_type'] == 'light'][
                'distance_travelled_in_flight'],
            color='royalblue', truncate=False, ax=ax, scatter_kws={'alpha': 0.5}, fit_reg=False, label='Light')  #

sns.regplot(flight_df[flight_df['flight_success'] == 'successful'][flight_df['expt_type'] == 'dark']['start_distance'],
            flight_df[flight_df['flight_success'] == 'successful'][flight_df['expt_type'] == 'dark'][
                'distance_travelled_in_flight'],
            color='mediumvioletred', truncate=False, ax=ax, scatter_kws={'alpha': 0.5}, label='Dark', fit_reg=False, )

tick_locations = [int(i) for i in np.arange(0, 800, 100)]
tick_values = [int(i) for i in np.arange(0, 400, 250 / 5)]

# ax.set_xticks()
# ax.set_xticklabels()
ax.figure.set_size_inches(5, 5)
sns.despine()
ax.legend()
plt.xlim(0, 750)
plt.ylim(0, 750)
plt.xticks(tick_locations, tick_values)
plt.yticks(tick_locations, tick_values)
plt.xlabel('Start distance (cm)')
plt.ylabel('Distance travelled in flight (cm)')

####################################

def multivariateGrid(col_x, col_y, col_k, df, color_by_group=True, scatter_alpha=.5,
                     use_kde=True, do_global_hist=False):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['color'] = c
            # kwargs['alpha'] = scatter_alpha
            sns.regplot(*args, **kwargs)  # plt.scatter

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )

    colors = {'dark': 'mediumvioletred',
              'light': 'royalblue'}
    legends = []

    for name, df_group in df.groupby(col_k):
        legends.append(name)

        if color_by_group:
            color = colors[name]

        g.plot_joint(
            colored_scatter(df_group[col_x], df_group[col_y], color),
        )
        sns.kdeplot(  # distplot
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
            # kde = use_kde,
            shade=True
        )
        sns.kdeplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,
            # kde = use_kde,
            vertical=True,
            shade=True
        )
    if do_global_hist:
        # Do also global Hist:
        sns.distplot(
            df[col_x].values,
            ax=g.ax_marg_x,
            color='grey'
        )
        sns.distplot(
            df[col_y].values.ravel(),
            ax=g.ax_marg_y,
            color='grey',
            vertical=True
        )

    plt.legend(legends)

    return g

multivariateGrid("start_distance", "flight_dist_ratio_float", 'expt_type', df=plot_df)
multivariateGrid("distance_travelled_before", "flight_dist_ratio_float", 'expt_type', df=plot_df)

ax = multivariateGrid("start_distance", 'distance_travelled_in_flight', 'expt_type', df=plot_df)

ax.plot(np.linspace(0, 700, 700), np.linspace(0, 700, 700))  # linestyle = '--', ,color='gray'

tick_locations = [int(i) for i in np.arange(0, 800, 100)]
tick_values = [int(i) for i in np.arange(0, 400, 250 / 5)]

# ax.set_xticks()
# ax.set_xticklabels()
ax.figure.set_size_inches(5, 5)
sns.despine()

plt.xlim(0, 750)
plt.ylim(0, 750)
plt.xticks(tick_locations, tick_values)
plt.yticks(tick_locations, tick_values)
plt.xlabel('Start distance (cm)')
plt.ylabel('Distance travelled in flight (cm)')

ax = multivariateGrid("start_distance", 'abs_speed_change_sum', 'expt_type', df=plot_df)

# ax[1].tick_params(labelleft=False)
# ax[1].tick_params(left=False)
ax.set_axis_labels('Start distance (m)', 'Absolute cumulative speed change')

xtick_locations = [int(i) for i in np.arange(0, 700, 100)]
xtick_values = [i for i in np.arange(0, 3.0, 2.5 / 5)]

# ax.set_xticks()
# ax.set_xticklabels()

plt.xticks(xtick_locations, xtick_values)

multivariateGrid(
    "start_distance", 'cumulative_head_shelter_angle', 'expt_type',
    df=flight_df[flight_df['flight_success'] == 'successful'])

ax.set_axis_labels('Start distance (m)', 'Cumulative head angle change (A.U)')

xtick_locations = [int(i) for i in np.arange(0, 700, 100)]
xtick_values = [i for i in np.arange(0, 3.0, 2.5 / 5)]

# ax.set_xticks()
# ax.set_xticklabels()

plt.xticks(xtick_locations, xtick_values)

multivariateGrid(
    "start_distance", 'cumulative_head_shelter_angle_change', 'expt_type',
    df=flight_df[flight_df['flight_success'] == 'successful'])

xtick_locations = [int(i) for i in np.arange(0, 700, 100)]
xtick_values = [i for i in np.arange(0, 3.0, 2.5 / 5)]

# ax.set_xticks()
# ax.set_xticklabels()

plt.xticks(xtick_locations, xtick_values)
plt.xlabel('Start distance (m)')
plt.ylabel('Cumulative head angle change (A.U)')


def convert_t_to_nest_to_accuracy_plot(row):
    t = row['t_to_nest_float']

    if row['t_to_nest_float'] > 3500:
        t = float('nan')

    return t


flight_df['accuracy_plot_t'] = flight_df.apply(convert_t_to_nest_to_accuracy_plot, axis=1)

###############################

maxes = {'light': 0,
         'dark': 0}

for ind in flight_df.index:

    experiment_type = flight_df[flight_df.index == ind]['expt_type'][0]
    # print(experiment_type)

    # print(flight_df[flight_df.index == ind]['t_to_nest_float'][0])

    if flight_df[flight_df.index == ind]['t_to_nest_float'][0] > maxes[experiment_type]:
        maxes[experiment_type] = flight_df[flight_df.index == ind]['t_to_nest_float'][0]


#####################

#n_light, n_dark = len(fight_df[flight_df['expt_type']=='light'], flight_df[flight_df['expt_type']=='dark'])

colors = {'dark': 'mediumvioletred',
        'light': 'royalblue'}

bin_width = 5
n_bins_light = int(maxes['light'] / bin_width)
n_bins_dark = int(maxes['dark'] / bin_width)

#axes = sns.kdeplot(flight_df[flight_df['expt_type']=='light']['t_to_nest_float'],cumulative=True, shade=True,color=colors['light'])
#axes = sns.kdeplot(flight_df[flight_df['expt_type']=='dark']['t_to_nest_float'],cumulative=True, shade=True, color=colors['dark'])

axes = sns.distplot(flight_df[flight_df['expt_type']=='light']['t_to_nest_float']/30, #'accuracy_plot_t'
                    #kde=False,
                    bins = n_bins_light,
                    hist_kws=dict(cumulative=True, label='Light'),
                    kde_kws=dict(cumulative=True, lw=0),
                    color=colors['light'],
                   label='Light') # shade=True,

axes = sns.distplot(flight_df[flight_df['expt_type']=='dark']['t_to_nest_float']/30, #'t_to_nest_float'
                    #kde=False,
                    bins = n_bins_dark,
                    hist_kws=dict(cumulative=True, label='Dark'),
                    kde_kws=dict(cumulative=True, lw=0),
                    color=colors['dark'],
                   label='Dark')

axes.set_xlim(0,30) #3500/
axes.set(xlabel='Time to shelter (s)', ylabel='Cumulative fraction')
sns.despine(offset=10, trim=True)

axes.legend()


############################

dark_means = np.mean(dark_distances_to_shelter, axis=0)
light_means = np.mean(light_distances_to_shelter, axis=0)

############################

from scipy import stats

dark_means = np.mean(dark_distances_to_shelter, axis=0) * (500 / (200 * 1000))
dark_sem = stats.sem(dark_distances_to_shelter, axis=0) * (500 / (200 * 1000))

light_means = np.mean(light_distances_to_shelter, axis=0) * (500 / (200 * 1000))
light_sem = stats.sem(light_distances_to_shelter, axis=0) * (500 / (200 * 1000))

x = (np.linspace(0, max_len, max_len) / 30) - 5

fig, ax = plt.subplots()

ax.plot(x, dark_means, color='mediumvioletred', label='Dark')
ax.fill_between(x, dark_means + dark_sem, dark_means - dark_sem,
                color='mediumvioletred', alpha=0.3)

ax.plot(x, light_means, color='royalblue', label='Light')
ax.fill_between(x, light_means + light_sem, light_means - light_sem,
                color='royalblue', alpha=0.3)
# sns.despine(offset=10, trim=True)
# ax.set_xticklabels(np.linspace(0, max_len, max_len/100))
ax.set_xlim([0, 900 / 30])

#############

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_bounds(0, 1)
# ax.set_ylim([-0.1, 1.1])

# ax[1].spines['left'].set_visible(False)
# ax[1].tick_params(labelleft=False)
# ax[1].tick_params(left=False)
axymin, axymax = 0, 500 * (500 / (200 * 1000))

ax.vlines([5], axymin, axymax, linestyles='dotted', color='gray', linewidth=3)

xtick_locations = [int(i) for i in np.arange(0, 30, 5)]
xtick_values = [int(i) for i in np.arange(-5, 30, 5)]

ax.set_xticks(xtick_locations)
ax.set_xticklabels(xtick_values)

ax.set_ylabel('Distance to shelter (m)')
ax.set_xlabel('Time (s)')
ax.set_ylim(0, 450 * (500 / (200 * 1000)))
ax.legend()


##############################


def get_body_to_shelter_angle(row):
    # vectors = (row['y'] - row['shelter_position'][0], row['y'] - row['shelter_position'][1])
    angles = np.degrees(np.arctan2(row['y'] - row['shelter_position'][1], row['x'] - row['shelter_position'][0]))

    return angles


def get_body_to_head_angle(row):
    angles = np.degrees(np.arctan2(row['y'] - row['head_y'], row['x'] - row['head_x']))  # np.degrees(

    return angles


def smooth_array(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


################################

flight_df['body_shelter_angle'] = flight_df.apply(get_body_to_shelter_angle, axis=1)
flight_df['body_head_angle'] = flight_df.apply(get_body_to_head_angle, axis=1)

################################

fig, axes = plt.subplots()

dark_angles = []
light_angles = []
max_len = 0

for ind in flight_df.index:

    if len(flight_df[flight_df.index == ind]['body_shelter_angle'][0]) > max_len:
        max_len = len(flight_df[flight_df.index == ind]['body_shelter_angle'][0])
# print(max_len)

x = np.linspace(0, max_len, max_len)

for ind in flight_df[flight_df['flight_success'] == 'successful'].index:

    arr = flight_df[flight_df.index == ind]['body_shelter_angle'][0] - \
          flight_df[flight_df.index == ind]['body_head_angle'][0]

    to_append = np.array((max_len - len(arr)) * [0])

    out = np.concatenate((arr, to_append), axis=None)
    out = np.degrees(out)
    out = smooth_array(out, 5)
    if flight_df[flight_df.index == ind]['expt_type'][0] == 'dark':
        dark_angles.append(out)

    if flight_df[flight_df.index == ind]['expt_type'][0] == 'light':
        light_angles.append(out)

    # color = colors[flight_df[flight_df.index==ind]['expt_type'][0]]

dark_angle_means_abs = np.mean(np.absolute(dark_angles), axis=0)
dark_angle_means = np.mean(dark_angles, axis=0)
dark_angle_sem = stats.sem(dark_angles, axis=0)

light_angle_means_abs = np.mean(np.absolute(light_angles), axis=0)
light_angle_means = np.mean(light_angles, axis=0)
light_angle_sem = stats.sem(light_angles, axis=0)

# axes[0].plot(x, dark_angle_means, color='mediumvioletred', label='Dark')
# axes[0].fill_between(x, dark_angle_means+dark_angle_sem, dark_angle_means-dark_angle_sem,
#                  color='mediumvioletred', alpha=0.3)

# axes[0].plot(x, light_angle_means, color='royalblue', label='Light')
# axes[0].fill_between(x, light_angle_means+light_angle_sem, light_angle_means-light_angle_sem,
#                  color='royalblue', alpha=0.3)

axes.plot(x, dark_angle_means_abs, color='mediumvioletred', label='Dark')
axes.fill_between(x, dark_angle_means_abs + dark_angle_sem, dark_angle_means_abs - dark_angle_sem,
                  color='mediumvioletred', alpha=0.3)

axes.plot(x, light_angle_means_abs, color='royalblue', label='Light')
axes.fill_between(x, light_angle_means_abs + light_angle_sem, light_angle_means_abs - light_angle_sem,
                  color='royalblue', alpha=0.3)

y_axis_titles = ['Angle to shelter (degrees)', 'Absolute angle to shelter (degrees)']
x_axis_titles = ['t to shelter (frames)', 't to shelter (frames)']

axes.legend()
axes.set_xlim([0, 900])
axes.set(xlabel='n frames (30 fps)', ylabel='Absolute head angle to shelter')
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)


#######################################

def get_mean_speed(row):
    mean_speed = np.mean(row['distance_per_frame'][300:])  #

    return mean_speed

flight_df['mean_speed'] = flight_df.apply(get_mean_speed, axis=1)

#######################################

fig, ax = plt.subplots()

dark_angles = []
light_angles = []
max_len = 0

for ind in flight_df.index:

    if len(flight_df[flight_df.index == ind]['body_shelter_angle'][0]) > max_len:
        max_len = len(flight_df[flight_df.index == ind]['body_shelter_angle'][0])
# print(max_len)

x = np.linspace(0, max_len, max_len) / 30

for ind in flight_df.index:

    arr = flight_df[flight_df.index == ind]['body_shelter_angle'][0] - \
          flight_df[flight_df.index == ind]['body_head_angle'][0]

    to_append = np.array((max_len - len(arr)) * [0])

    out = np.concatenate((arr, to_append), axis=None)
    out = np.degrees(out)
    out = smooth_array(out, 5)
    if flight_df[flight_df.index == ind]['expt_type'][0] == 'dark':
        dark_angles.append(out)

    if flight_df[flight_df.index == ind]['expt_type'][0] == 'light':
        light_angles.append(out)

    # color = colors[flight_df[flight_df.index==ind]['expt_type'][0]]

dark_angle_means_abs = np.mean(np.absolute(dark_angles), axis=0)
dark_angle_means = np.mean(dark_angles, axis=0)
dark_angle_sem = stats.sem(dark_angles, axis=0)

light_angle_means_abs = np.mean(np.absolute(light_angles), axis=0)
light_angle_means = np.mean(light_angles, axis=0)
light_angle_sem = stats.sem(light_angles, axis=0)

# axes[0].plot(x, dark_angle_means, color='mediumvioletred', label='Dark')
# axes[0].fill_between(x, dark_angle_means+dark_angle_sem, dark_angle_means-dark_angle_sem,
#                  color='mediumvioletred', alpha=0.3)

# axes[0].plot(x, light_angle_means, color='royalblue', label='Light')
# axes[0].fill_between(x, light_angle_means+light_angle_sem, light_angle_means-light_angle_sem,
#                  color='royalblue', alpha=0.3)

ax.plot(x, dark_angle_means_abs, color='mediumvioletred', label='Dark')
ax.fill_between(x, dark_angle_means_abs + dark_angle_sem, dark_angle_means_abs - dark_angle_sem,
                color='mediumvioletred', alpha=0.3)

ax.plot(x, light_angle_means_abs, color='royalblue', label='Light')
ax.fill_between(x, light_angle_means_abs + light_angle_sem, light_angle_means_abs - light_angle_sem,
                color='royalblue', alpha=0.3)

y_axis_titles = ['Angle to shelter (degrees)', 'Absolute angle to shelter (degrees)']
x_axis_titles = ['t to shelter (frames)', 't to shelter (frames)']

ax.legend()
ax.set_xlim([150 / 30, 1050 / 30])
ax.set(xlabel='Time (s)', ylabel='Absolute head angle to shelter')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

axymin, axymax = 0, 200

ax.vlines([10], axymin, axymax, linestyles='dotted', color='gray', linewidth=3)

ax.set_ylim(0, 180)

xtick_locations = [int(i) for i in np.arange(5, 35, 5)]
xtick_values = [int(i) for i in np.arange(-5, 30, 5)]

ax.set_xticks(xtick_locations)
ax.set_xticklabels(xtick_values)

ax.legend()


########################################


cut = pd.cut(flight_df['distance_travelled_before'], 600 / 50)

vals = cut.value_counts()
# for ind in flight_df.groupby(cut).groups.keys():
#    print(ind)
d_proportions = []
l_proportions = []
for i, key in enumerate(flight_df.groupby(cut).groups.keys()):
    inds = flight_df.groupby(cut).groups[key]
    # n_success = len(flight_df[flight_df.index == ]])
    np.seterr(divide='ignore', invalid='ignore')
    try:
        d_proportions.append(len(
            flight_df.loc[list(inds)][flight_df['flight_success'] == 'successful'][
                flight_df['expt_type'] == 'dark']) / len(
            flight_df.loc[list(inds)][flight_df['expt_type'] == 'dark']))
    except ZeroDivisionError:
        d_proportions.append(0)

    try:
        l_proportions.append(len(
            flight_df.loc[list(inds)][flight_df['flight_success'] == 'successful'][
                flight_df['expt_type'] == 'light']) / len(
            flight_df.loc[list(inds)][flight_df['expt_type'] == 'light']))
    except ZeroDivisionError:
        l_proportions.append(0)

fig, ax = plt.subplots()

ax.scatter(range(len(proportions)), d_proportions, c=colors['dark'], label='Dark', alpha=0.6)
ax.scatter(range(len(proportions)), l_proportions, c=colors['light'], label='Light', alpha=0.6)

ax.legend()


#################################

from matplotlib import colors
from matplotlib import colorbar


def plot_speed_heatmap(df_rows):
    # fig, ax = plt.subplots(2,1)

    longest_flight_len = 0
    cmap = plt.cm.magma  # inferno#gist_heat  #seismic  # coolwarm   #hot
    rankedData = pd.Series([])
    max_flight_len = 30 * 15
    max_speed = 0
    speed_limit = 10

    fig = plt.figure(figsize=(30, 30))
    ax0 = plt.subplot2grid((30, 30), (0, 0), colspan=19, rowspan=10)
    ax1 = plt.subplot2grid((30, 30), (0, 19), colspan=1, rowspan=10)

    for i in df_rows.index:

        if len(df_rows['distance_per_frame'][i]) == 0:
            continue

        if len(df_rows['distance_per_frame'][i]) > longest_flight_len:
            longest_flight_len = len(df_rows['distance_per_frame'][i][300 - 90:])

        if df_rows['distance_per_frame'][i].max() > max_speed:
            max_speed = df_rows['distance_per_frame'][i].max()

    # if longest_flight_len > max_flight_len:

    longest_flight_len = max_flight_len

    for i in df_rows.index:

        speed = df_rows['distance_per_frame'][i][300 - 90:]

        speed[speed > speed_limit] = speed_limit

        pad_width = longest_flight_len - len(speed)

        if pad_width < 0:
            pad_width = 0

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

    ind = []
    max_pos = []

    for i, d in enumerate(rankedData):
        ind.append(i)
        max_pos.append(np.where(rankedData[i] == rankedData[i].max())[0][0])

    zipped = zip(ind, max_pos)
    zipped = sorted(zipped, key=lambda t: t[1])
    index = [i[0] for i in zipped]
    rankedData = rankedData[index]

    rankedData = [x for x in rankedData if np.any(x)]

    vmax, vmin = speed_limit, -0
    # try:

    # print(rankedData)

    ax0.matshow(rankedData, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none', aspect='auto')
    norm = colors.Normalize(vmin=0, vmax=vmax)
    axymin, axymax = ax0.get_ylim()
    ax0.vlines([90], axymin, axymax, linestyles='dotted', color='w', linewidth=5)

    # ax2.axis('off')
    ax0.linewidth = 50.0
    # ax0.set_xlim(0,900)
    cbar = colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, spacing='proportional')
    cbar.set_label('m/s', rotation=0, labelpad=15, y=1, color='black')
    # cbar.set_ticklabels(['Low', 'Medium'])# horizontal colorbar
    plt.rcParams.update({'xtick.color': 'black', 'font.size': 15})  # 'font.size': 5,

    xtick_locations = [int(i) for i in np.arange(0, longest_flight_len, longest_flight_len / 15)]
    xtick_values = [int(i) for i in np.arange(0, longest_flight_len - 1, longest_flight_len / 15) / 30]

    ax0.set_xticks(xtick_locations)

    ax0.set_xticklabels(xtick_values)

    plt.show()

    return rankedData


light_flights = flight_df.loc[(flight_df['expt_type'] == 'light') &
                              (flight_df['flight_success'] == 'successful')]
dark_flights = flight_df.loc[(flight_df['expt_type'] == 'dark') &
                             (flight_df['flight_success'] == 'successful')]

light_df = plot_speed_heatmap(light_flights)
dark_df = plot_speed_heatmap(dark_flights)

##########################

fig, ax = plt.subplots()
colors = {'light': 'navy',
          'dark': 'magenta'}

for ind in flight_df[flight_df['flight_success'] == 'successful'].index:

    if ind == '28':
        continue

    color = colors[flight_df[flight_df.index == ind]['expt_type'][0]]

    shelter_x, shelter_y = flight_df[flight_df.index == ind]['shelter_position'][0]

    x, y = flight_df[flight_df.index == ind]['x'][0][300], flight_df[flight_df.index == ind]['y'][0][300]

    ax.scatter(x - shelter_x,
               y - shelter_y,
               c=color,
               alpha=0.5)


################

from scipy import interpolate

colors = {'dark': 'mediumvioletred',
          'light': 'royalblue'}
# flight_df['flight_success'] = flight_df.apply(get_flight_success, args=(40 * 30,), axis=1)

recent_mice_df = flight_df

interpolated_cums_dark = []
interpolated_cums_light = []

normalised_interpolated_dark = []
normalised_interpolated_light = []

for ind in recent_mice_df.index:

    if recent_mice_df['flight_success'][ind] != 'successful':
        continue

    flight_start = 300
    # try:
    #   flight_start += np.where(recent_mice_df[recent_mice_df['experiment_name'] == '190401_dwm_light_us_619_3'][
    #                                 'distance_per_frame']['276'][300:] > 1.5)[0][0]
    # except:
    #    print('could not add flight speed')

    cumulative = np.cumsum(recent_mice_df['vec_angles'][ind][flight_start:])

    x_interpolate = np.linspace(0, len(cumulative) - 1, 1000)
    x_cum = np.arange(len(cumulative))

    try:
        f_int = interpolate.interp1d(x_cum, cumulative)

        # if np.isnan(f_int(x_interpolate)).any():
        #    print('Nans in:', recent_mice_df['experiment_name'][ind])
        # print(cumulative)

        # fig, ax = plt.subplots(2,1)
        interpolated = f_int(x_interpolate)

        normalised_interpolated = interpolated / interpolated[-1]

        # max_len = 500
        if recent_mice_df['expt_type'][ind] == 'dark':
            interpolated_cums_dark.append(interpolated)
            normalised_interpolated_dark.append(normalised_interpolated)

        if recent_mice_df['expt_type'][ind] == 'light':
            interpolated_cums_light.append(interpolated)
            normalised_interpolated_light.append(normalised_interpolated)

    except:
        print(recent_mice_df['experiment_name'][ind])

    # x_interpolate_plot = x_interpolate * (max_len/len(cumulative))

    # ax[0].plot(x_cum, cumulative)
    # ax[1].plot(x_interpolate_plot, f_int(x_interpolate))

## WITH SEM
from scipy import stats

dark_cum_means = np.nanmean(np.array(interpolated_cums_dark), axis=0)
dark_cum_sem = stats.sem(np.array(interpolated_cums_dark), nan_policy='omit')

light_cum_means = np.nanmean(np.array(interpolated_cums_light), axis=0)
light_cum_sem = stats.sem(np.array(interpolated_cums_light), nan_policy='omit')

normalised_dark_cum_means = np.nanmean(np.array(normalised_interpolated_dark), axis=0)
normalised_dark_cum_sem = stats.sem(np.array(normalised_interpolated_dark), nan_policy='omit')

normalised_light_cum_means = np.nanmean(np.array(normalised_interpolated_light), axis=0)
normalised_light_cum_sem = stats.sem(np.array(normalised_interpolated_light), nan_policy='omit')

x_plot = np.linspace(0, 1, 1000)

fig, ax = plt.subplots(2, 1, figsize=(10, 20))

ax[0].plot(x_plot, dark_cum_means,
           color=colors['dark'], linewidth=3.0, label='Dark')
ax[0].fill_between(x_plot, dark_cum_means + dark_cum_sem, dark_cum_means - dark_cum_sem,
                   color=colors['dark'], alpha=0.3)

ax[0].plot(x_plot, light_cum_means,
           color=colors['light'], linewidth=3.0, label='Light')
ax[0].fill_between(x_plot, light_cum_means + light_cum_sem, light_cum_means - light_cum_sem,
                   color=colors['light'], alpha=0.3)

ax[1].plot(x_plot, normalised_dark_cum_means,
           color=colors['dark'], linewidth=3.0, label='Dark')
ax[1].fill_between(x_plot, normalised_dark_cum_means + normalised_dark_cum_sem,
                   normalised_dark_cum_means - normalised_dark_cum_sem,
                   color=colors['dark'], alpha=0.3)

ax[1].plot(x_plot, normalised_light_cum_means,
           color=colors['light'], linewidth=3.0, label='Light')
ax[1].fill_between(x_plot, normalised_light_cum_means + normalised_light_cum_sem,
                   normalised_light_cum_means - normalised_light_cum_sem,
                   color=colors['light'], alpha=0.3)

ax[0].set_title('Normalised cumulative trajectory angle change across normalised flight time')
ax[1].set_title('Normalised cumulative trajectory angle change across normalised flight time')

# x_line = np.linspace(0,1000, 1000)
# y_line = np.linspace(0,1,1000)

# ax[1].plot(x_line, y_line, color='gray', label='linear', linestyle=':')
ax[0].legend(loc=2)
ax[1].legend(loc=2)

for axis in ax:
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)


#########################################

def get_head_shelter_angle(row):
    head_shelter_angle = row['body_shelter_angle'] - row['body_head_angle']

    return head_shelter_angle


def get_head_shelter_angle_change(row):
    return np.ediff1d(row['head_shelter_angle'], to_begin=0)


def get_cumulative_head_shelter_angle(row):
    cumulative_head_shelter_angle = sum(np.absolute(row['head_shelter_angle']))

    return cumulative_head_shelter_angle


def get_cumulative_head_shelter_angle_change(row):
    cumulative_head_shelter_angle_change = sum(np.absolute(row['head_shelter_angle_change']))

    return cumulative_head_shelter_angle_change


flight_df['head_shelter_angle'] = flight_df.apply(get_head_shelter_angle, axis=1)
flight_df['head_shelter_angle_change'] = flight_df.apply(get_head_shelter_angle_change, axis=1)
flight_df['cumulative_head_shelter_angle'] = flight_df.apply(get_cumulative_head_shelter_angle, axis=1)
flight_df['cumulative_head_shelter_angle_change'] = flight_df.apply(get_cumulative_head_shelter_angle_change, axis=1)


##############################


for ind in flight_df.index:
    plt.close('all')
    fig, ax = plt.subplots()
    print(flight_df[flight_df.index == ind]['experiment_name'])
    xs, ys = fix_tracking(flight_df[flight_df.index == ind]['x'][0], flight_df[flight_df.index == ind]['y'][0],
                          zeros_fun, 20)

    colors = cm.winter(np.linspace(0, 1, len(ys)))

    shelter = flight_df[flight_df.index == ind]['shelter_position'][0]

    for i, c in enumerate(colors):
        ax.scatter(xs[i], ys[i], s=5, alpha=0.6, color=c)

    ax.scatter(shelter[0], shelter[1], s=50, color='blue')
    ax.set_ylim([0, 500])
    ax.set_xlim([0, 500])

    figname = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\analysis\\big_arena_analysis\\traj_plots\\traj_plot_winter_' + ind + '_' + \
              flight_df[flight_df.index == ind]['experiment_name'][0]
    fig.savefig(figname)

################################


def load_nest_info(nest_dict, nest_info_path):
    #'E:\\big_arena_analysis\\nest_locations.csv'
    with open(nest_info_path, 'r') as f:
        read = csv.reader(f, delimiter=',')
        for row in read:
            if len(row) == 0:
                continue

            nest_dict[row[0]] = [list(x) for x in list(group([int(x) for x in re.findall(r"[\w']+", row[1])], 2))]

    return nest_dict

def get_mean_shelter_location(shelter_locations, nest_dict):

    for key in nest_dict:
        x_mean, y_mean = 0, 0
        for i, coord in enumerate(nest_dict[key]):
            if i == 0:
                continue
            x_mean += coord[0]
            y_mean += coord[1]

        x_mean, y_mean = x_mean / 5, y_mean / 5

        shelter_locations[key] = [x_mean, y_mean]

    return shelter_locations



###########################


to_tsne = ['shelter_distance', 'x', 'y', 'head_x', 'head_y', 'tail_x', 'tail_y',
           'distance_per_frame', 'speed_change', 'angles']  # vec_angles

flight_df[to_tsne]

concat = pd.concat(
    [flight_df[flight_df.index == '0']['shelter_distance'], flight_df[flight_df.index == '1']['shelter_distance']])

coord_lens = []
speed_lens = []
dist_lens = []

tsne_df = pd.DataFrame(columns=to_tsne)
tsne_dict = {}
for key in to_tsne:

    data_list = []

    for ind in flight_df.index:

        if key == 'speed_change' or key == 'distance_per_frame':

            coord_len = len(flight_df[flight_df.index == ind]['x'][0])
            key_len = len(flight_df[flight_df.index == ind][key][0])

            if coord_len != key_len:
                data_list.extend([0] * (coord_len - key_len))

        data_list.extend(flight_df[flight_df.index == ind][key][0])

    tsne_dict[key] = data_list


##############################


tsne_df = pd.DataFrame.from_dict(tsne_dict)

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, perplexity = 200.).fit_transform(tsne_df)

x_coords = []
y_coords = []
for n,i in enumerate(X_embedded):
    if n % 50000 == 0:
        print('n = ' , n)
    x_coords.append(i[0])
    y_coords.append(i[1])

#%%
plt.scatter(x_coords, y_coords, s = 1)
