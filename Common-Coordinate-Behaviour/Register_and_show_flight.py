# Import packages
import os

from video_funcs import (
    peri_stimulus_video_clip, whole_video_clip, register_arena, get_background, get_filetype_paths,
    check_both_videotypes, convert_video, get_tdms_indexes, extract_trial_clips) #  make_labelled_video, get_cmap
from dlc_tracking import analyse, dlc_setupTF
from termcolor import colored
from os import walk
from os import listdir
from os.path import isfile, join
from loadsave_funcs import load_paths, load_yaml
import ffmpy

import yaml



# ========================================================
#           SET PARAMETERS
# ========================================================

# file path of behaviour video
#base_folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\longrange\\data\\HC_lesion\\default'
#base_folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\dwm\\data\\dark'
#base_folder = r'E:\Dropbox (UCL - SWC)\big_Arena\experiments\dwm\data\rotate_cues'
#base_folder = r'E:\Dropbox (UCL - SWC)\big_Arena\experiments\corridor\light'

#base_folder = r'E:\Dropbox (UCL - SWC)\big_Arena\experiments\dwm\data\basic' #dark #basic #\190401_dwm_light_us_550_1'

#base_folder = r'E:\Dropbox (UCL - SWC)\big_Arena\experiments\dwm\data\trap'
#base_folder = r'E:\Dropbox (UCL - SWC)\big_Arena\experiments\hc_lesion'

#base_folder = r'E:\Dropbox (UCL - SWC)\big_Arena\experiments\mouserotation2.0'
base_folder = r'E:\Dropbox (UCL - SWC)\big_Arena\experiments\subiculum'
#base_folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\corridor\\light'
#base_folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\dwm\\data\\basic' #\\191017_dwm_nocurtains_800_1\\191017_dwm_nocurtains_800_1b'

#base_folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\mouserotation'
#base_folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\longrange\\data\\HC_lesion\\dark'
#base_folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\twonest\\data'
#base_folder = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\longrange\\data\\HC_lesion\\default'
#base_folder = 'E:\Dropbox (UCL - SWC)\\big_Arena\\experiments\\dwm\\data\\HC_lesion' #181116_dwm_loom_337_5\\181116_dwm_loom_337_5b'

file_path = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\longrange\\data\\HC_lesion\\default\\'
file_name = 'cam1.mp4'


# DLC config

#dlc_config_settings = load_yaml(r'./Configs/cfg_dlc_mouserotation_190917.yml') # './Configs/cfg_dlc_DWM.yml' #'./Configs/cfg_dlc_mouserotation_190917.yml' #'./Configs/cfg_dlc_DWM.yml' #'./Configs/cfg_dlc_mouserotation_190917.yml' #'./Configs/cfg_dlc_DWM.yml' #'./Configs/cfg_dlc.yml' cfg_dlc_190530.yml
dlc_config_settings = load_yaml(r'./Configs/cfg_dlc_DWM.yml') #'./Configs/cfg_dlc_mouserotation_190917.yml' #'./Configs/cfg_dlc_DWM.yml' #'./Configs/cfg_dlc_mouserotation_190917.yml' #'./Configs/cfg_dlc_DWM.yml' #'./Configs/cfg_dlc.yml' cfg_dlc_190530.yml


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

subdicts = get_immediate_subdirectories(file_path)

video_file_path = 'E:\\Dropbox (UCL - SWC)\\big_Arena\\experiments\\longrange\\data\\HC_lesion\\default\\190118_hc_413_1\\190118_hc_413_1b\\cam1.mp4'

# file path of behaviour video
save_file_path = 'E:\\fisheye_correction_test'

# file path of fisheye correction -- set to an invalid location such as '' to skip fisheye correction
# A corrective mapping for the branco lab's typical camera is included in the repo!
fisheye_map_location = 'C:\\Users\\matthewp.W221N\\Documents\\GitHub\\Common-Coordinate-Behaviour\\fisheye_maps.npy'

# If True, will not put on the stimuli around the frame or apply stimulus specific changes
correct_whole_video = False
convert_videos = False
preprocess = False
show_video = False
produce_trial_clips = False
overwrite_previous = False
dlc_track = True
make_labelled_video = False

# frame of stimulus onset
stim_frame = 90

# seconds before stimulus to start video
window_pre = 3

# seconds before stimulus to start video
window_post = 7

# frames per second of video
fps = 30

# name of experiment
experiment = '190118_hc_413_1a_TESTVID' #'180402_longrange_nowall_1_new'

# name of mouse
mouse_id = 'CA413_1'

# stimulus type
stim_type = 'loom'

# x and y offset as set in the behaviour software
x_offset = 170
y_offset = 80

# for flight image analysis: darkness relative to background threshold
# (relative darkness, number of dark pixels)

dark_threshold = [.55,950]

# ========================================================
#           GET BACKGROUND
# ========================================================

desired_videotypes = ['.avi', '.mp4']

if convert_videos:
    n_converted=0
    video_directories = get_filetype_paths('.avi', base_folder)

    for f in video_directories:
        check_output = check_both_videotypes(f, 'cam1', desired_videotypes)
        if check_output != True:
            if check_output == False:
                print(f)
                print('SOMETHING BROKE')
                break


            for ele in enumerate(desired_videotypes):
                if ele != check_output:
                    save = ele

            video_name_path = f + '\\cam1'

            input_video = '{}{}'.format(video_name_path, '.avi')
            output_video = '{}{}'.format(video_name_path, save[1])
            #print('CONVERTING VIDEO:', input_video, 'to OUTPUT VIDEO:', output_video)
            print('converted', n_converted, '/', len(video_directories))
            convert_video(input_video, output_video)
            n_converted +=1

    print('COMPLETED VIDEO CONVERSION. {} VIDEOS CONVERTED'.format(n_converted))
    print('Now moving to fisheye correction...')

if correct_whole_video:

    video_directories = get_filetype_paths('.avi', base_folder)
    registration_dict = {}
    offset_dict = {}
    roi_dict = {}
    roi = False

    for f in video_directories:
        files = [file for file in listdir(f) if isfile(join(f, file))]

        if any(file.endswith('FEC.avi') or file.endswith('FEC.mp4') for file in files):

            print('Skipped {} -- already corrected.'.format(f))
            continue
        load_path = f + '\\cam1.avi'
        videoname = f + '\\cam1_FEC'
        save_path = f
        print('Registering video:', f)

        print(colored('Fetching background', 'green'))
        background_image = get_background(
            load_path, start_frame=1000, avg_over=10)

        print(colored('Registering arena', 'green'))

        if f[0:-1] not in registration_dict or f[0:-1] not in roi_dict:
            print('Previous x-offset: {}. Previous y-offset: {}'.format(x_offset, y_offset))

            registration, x_offset, y_offset, roi = register_arena(background_image, fisheye_map_location, x_offset, y_offset, roi)

            registration_dict[f[0:-1]] = registration

            x_offset, y_offset = int(x_offset), int(y_offset)

            offset_dict[f[0:-1]] = (x_offset, y_offset)

            roi_dict[f[0:-1]] = roi

        elif f[0:-1] in registration_dict and f[0:-1] in roi_dict:
            print('REGISTRATION AND ROI DETECTED. USING PREVIOUS ROI')
            registration = registration_dict[f[0:-1]]
            roi = roi_dict[f[0:-1]]

        print(colored('Creating flight clip', 'green'))

    print('COMPLETED ALL REGISTRATIONS IN FOLDER')
    print('PROCEEDING TO CORRECT VIDEOS')



    for f in video_directories:

        files = [file for file in listdir(f) if isfile(join(f, file))]

        if any(file.endswith('FEC.avi') or file.endswith('FEC.mp4') for file in files):
            print('Skipped {} -- already corrected.'.format(f))
            continue

        load_path = f + '\\cam1.avi'
        videoname = f + '\\cam1_FEC'
        save_path = f
        print('Correcting video:', f)

        registration =  registration_dict[f[0:-1]]
        x_offset, y_offset = offset_dict[f[0:-1]]
        roi = roi_dict[f[0:-1]]
        print('reg:', registration)
        print('x_offset, y_offset:', x_offset, y_offset)
        print('roi:', roi)
        print('load_path, videoname, save_path', load_path, videoname, save_path)
        print('converted', n_converted, '/', len(video_directories))
        whole_video_clip(load_path, videoname, save_path, 0, -1,
                         registration, x_offset, y_offset, dark_threshold, show_video, roi,
                         save_clip=True, display_clip=True, counter=True)

if produce_trial_clips:
    video_directories = get_filetype_paths('.avi', base_folder)

    # Remove duplicates
    video_directories = list(dict.fromkeys(video_directories))


    print('ENTERING TRIAL CLIP PRODUCTION')

    for f in video_directories:
        files = [file for file in listdir(f) if isfile(join(f, file))]


        if any(file.endswith('FEC.avi') for file in files):
            print('CUTTING VIDEOS FROM:', f)
            load_path = f + '\\cam1_FEC.avi'
            videoname = f + '\\cam1_FEC'
            save_path = f

            stim_indices = get_tdms_indexes(f)
            extract_trial_clips(vidpath=load_path, videoname=videoname, savepath=save_path, stim_indices=stim_indices,
                            pre_window=30, post_window=30,produce_txt=True, fps=False, save_clip=True,
                            show_video=False, overwrite_previous=False, counter=True)

        else:
            print('No fec.avi files in .{}'.format(f))

if dlc_track:

    ### returns {'scorer': scorer, 'sess': sess, 'inputs': inputs, 'outputs': outputs, 'cfg': cfg}
    TF_settings = dlc_setupTF(dlc_config_settings)


    # TODO: Change this!! it's for Fede's code not mine

    video_directories = get_filetype_paths('FEC.avi', base_folder)
    #video_directories = [x for x in video_directories if x.endswith('FEC.avi')]
    for f in video_directories:

        analyse(TF_settings, f, 'cam1_FEC.avi')

if make_labelled_video:
    video_directories = get_filetype_paths('FEC.avi', base_folder)

    for video in video_directories:
        vname = video.split('.')[0]
        tmpfolder = 'temp' + vname
        auxiliaryfunctions.attempttomakefolder(tmpfolder)

        if os.path.isfile(os.path.join(tmpfolder, vname + '_DeepLabCutlabeled.mp4')):
            print("Labeled video already created.")
        else:
            print("Loading ", video, "and data.")
            dataname = video.split('.')[0] + scorer + '.h5'
            try:  # to load data for this video + scorer
                Dataframe = pd.read_hdf(dataname)
                clip = VideoFileClip(video)
                CreateVideo(clip, Dataframe)
            except FileNotFoundError:
                datanames = [fn for fn in os.listdir(os.curdir) if (vname in fn) and (".h5" in fn) and "resnet" in fn]
                if len(datanames) == 0:
                    print("The video was not analyzed with this scorer:", scorer)
                    print("No other scorers were found, please run AnalysisVideos.py first.")
                elif len(datanames) > 0:
                    print("The video was not analyzed with this scorer:", scorer)
                    print("Other scorers were found, however:", datanames)
                    print("Creating labeled video for:", datanames[0], " instead.")

                    Dataframe = pd.read_hdf(datanames[0])
                    clip = VideoFileClip(video)
                    CreateVideo(clip, Dataframe)






