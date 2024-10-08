from turtledemo.chaos import f

import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
import os
from os import walk
import os.path
import ffmpy
from nptdms import TdmsFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import sys

imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import subprocess
import pandas as pd
#import auxiliaryfunctions

def get_background(vidpath, start_frame = 1000, avg_over = 100):
    """ extract background: average over frames of video """

    vid = cv2.VideoCapture(vidpath)

    # initialize the video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = np.zeros((height, width))
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # initialize the counters
    every_other = int(num_frames / avg_over)
    j = 0

    for i in tqdm(range(num_frames)):

        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()  # get the frame

            if ret:
                # store the current frame in as a numpy array
                background += frame[:, :, 0]
                j+=1


    background = (background / (j)).astype(np.uint8)
    cv2.imshow('background', background)
    cv2.waitKey(10)
    vid.release()

    return background

def get_filetype_paths(filetype, base):
    # Takes a desired filetype and a base directory and returns a list of their locations
    # within it's subdirectories

    # Set up file location list
    f = []

    # Search through folders
    for (dirpath, dirnames, filenames) in walk(base):

        # if a file is detected, enter detection of its type append to list
        if filenames:
            for file in filenames:

                # If filetype matches the desired, append to list
                if file.endswith(filetype):
                    f.append(dirpath)  # + '\\' + file

    return f

def check_both_videotypes(directory, file_name, file_types):
    files = []
    present = False

    for (dirpath, dirnames, filenames) in walk(directory):
        files.extend(filenames)
        break

    if file_name + file_types[0] in files and file_name + file_types[1] in files:
        #print('Both types present in:', directory)
        present = True

    elif present==False:
        for filename in files:
            if filename[0:len(file_name)] == file_name:
                present = filename[-4:]

    return present

def convert_video(input_vid, output_vid):
    ff = ffmpy.FFmpeg(inputs={input_vid: None}, outputs={output_vid: None})
    ff.run()

def get_tdms_indexes(path):
    indices = []
    tdms_path = []
    #print('TDMS PATH: ', tdms_path)

    for (dirpath, dirnames, filenames) in walk(path):

        # if a file is detected, enter detection of its type append to list
        if filenames:
            for file in filenames:

                # If filetype matches the desired, append to list
                if file.endswith('.tdms'):
                    tdms_path.append(dirpath + '\\' + file)

    if len(tdms_path) > 1:
        print('TOO MANY TMDS FILES DETECTED IN', path)

    tdms_file = TdmsFile(tdms_path[0])

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

# =================================================================================
#              CREATE MODEL ARENA FOR COMMON COORDINATE BEHAVIOUR
# =================================================================================
def model_arena(size):
    ''' NOTE: this is the model arena for the Barnes maze with wall
    this function must be customized for any other arena'''
    # initialize model arena
    model_arena = np.zeros((1000,1000)).astype(np.uint8)
    cv2.rectangle(model_arena, (250,250),(750, 750), 50, thickness=-1)

    # add wall - up
    # cv2.rectangle(model_arena, (int(500 - 554 / 2), int(500 - 6 / 2)), (int(500 + 554 / 2), int(500 + 6 / 2)), 60, thickness=-1)
    # add wall - down
    # cv2.rectangle(model_arena, (int(500 - 504 / 2), int(500 - 8 / 2)), (int(500 + 504 / 2), int(500 + 8 / 2)), 0, thickness=-1)

    # add shelter
    model_arena_shelter = model_arena.copy()
    cv2.rectangle(model_arena_shelter, (int(500 - 50), int(500 + 385 + 25 - 50)), (int(500 + 50), int(500 + 385 + 25 + 50)), (0, 0, 255),thickness=-1)
    alpha = .5
    cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

    model_arena = cv2.resize(model_arena,size)

    # --------------------------------------------------------------------------------------------
    # THESE ARE THE FOUR POINTS USED TO INITIATE REGISTRATION -- CUSTOMIZE FOR YOUR OWN PURPOSES
    # --------------------------------------------------------------------------------------------
    points = np.array(([250,250],[250, 750],[750,750],[750,250]))* [size[0]/1000,size[1]/1000]

    # cv2.imshow('model_arena',model_arena)

    return model_arena, points
# =================================================================================
#              IMAGE REGISTRATION GUI
# =================================================================================
def register_arena(background, fisheye_map_location, x_offset, y_offset, roi):
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """

    # create model arena and background
    arena, arena_points = model_arena(background.shape)

    # load the fisheye correction

    while True:

        maps = np.load(fisheye_map_location)
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2]*0

        background_copy = cv2.copyMakeBorder(background, y_offset, int((map1.shape[0] - background.shape[0]) - y_offset),
                                             x_offset, int((map1.shape[1] - background.shape[1]) - x_offset),
                                             cv2.BORDER_CONSTANT, value=0)

        background_copy = cv2.remap(background_copy, map1, map2,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        background_copy = background_copy[y_offset:-int((map1.shape[0] - background.shape[0])- y_offset),
                                        x_offset:-int((map1.shape[1] - background.shape[1]) - x_offset)]

        #except:
            #background_copy = background.copy()
            #fisheye_map_location = ''
            #print('fisheye correction not available')
        cv2.namedWindow('bg_offset')
        cv2.imshow('bg_offset', background_copy)

        cv2.waitKey(10)
        done = input('enter y for done, n for not done')

        if done != 'y':
            print('Prev x offset:', x_offset)
            print('Prev y offset:', y_offset)
            x_offset = int(input('Enter new X Offset:'))
            y_offset = int(input('Enter new Y Offset:'))
            cv2.destroyWindow('bg_offset')
            cv2.waitKey(10)
            cv2.waitKey(10)
        if done == 'y':
            break


    # initialize clicked points
    blank_arena = arena.copy()
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]
    cv2.namedWindow('registered background')
    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)
    use_loaded_transform = False
    make_new_transform_immediately = False
    use_loaded_points = False

    # LOOP OVER TRANSFORM FILES
    file_num = -1;
    transform_files = glob.glob('*transform.npy')
    for file_num, transform_file in enumerate(transform_files[::-1]):

        # USE LOADED TRANSFORM AND SEE IF IT'S GOOD
        loaded_transform = np.load(transform_file)
        M = loaded_transform[0]
        background_data[1] = loaded_transform[1]
        arena_data[1] = loaded_transform[2]

        # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
        registered_background = cv2.warpAffine(background_copy, M, background.shape)
        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB))
                                       #* np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB))
                      # * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)


        print('Does transform ' + str(file_num+1) + ' / ' + str(len(transform_files)) + ' match this session?')
        print('\'y\' - yes! \'n\' - no. \'q\' - skip examining loaded transforms. \'p\' - update current transform')


        while True:
            cv2.imshow('registered background', overlaid_arenas)
            k = cv2.waitKey(10)
            if  k == ord('n'):
                print('N PRESSED')
                break
            elif k == ord('y'):
                print('Y PRESSED')
                use_loaded_transform = True
                break
            elif k == ord('q'):

                make_new_transform_immediately = True
                break
            elif k == ord('p'):
                use_loaded_points = True
                break
        if use_loaded_transform:
            update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M]
            break
        elif make_new_transform_immediately or use_loaded_points:
            file_num = len(glob.glob('*transform.npy'))-1
            break

    if not use_loaded_transform:
        if not use_loaded_points:
            print('\nSelect reference points on the experimental background image in the indicated order')

            # initialize clicked point arrays
            background_data = [background_copy, np.array(([], [])).T]
            arena_data = [[], np.array(([], [])).T]

            # add 1-2-3-4-5 markers to model arena
            for i, point in enumerate(arena_points.astype(np.uint32)):
                arena = cv2.circle(arena, (point[0], point[1]), 3, 255, -1)
                arena = cv2.circle(arena, (point[0], point[1]), 4, 0, 1)
                cv2.putText(arena, str(i+1), tuple(point), 0, .55, 150, thickness=2)

                point = np.reshape(point, (1, 2))
                arena_data[1] = np.concatenate((arena_data[1], point))

            # initialize GUI
            cv2.startWindowThread()
            cv2.namedWindow('background')
            cv2.imshow('background', background_copy)
            cv2.namedWindow('model arena')
            cv2.imshow('model arena', arena)


            # create functions to react to clicked points
            cv2.setMouseCallback('background', select_transform_points, background_data)  # Mouse callback

            while True: # take in clicked points until four points are clicked
                cv2.imshow('background',background_copy)

                number_clicked_points = background_data[1].shape[0]
                if number_clicked_points == len(arena_data[1]):
                    break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # perform projective transform
        # M = cv2.findHomography(background_data[1], arena_data[1])


        M = cv2.estimateRigidTransform(background_data[1], arena_data[1], False)


        # REGISTER BACKGROUND, BE IT WITH LOADED OR CREATED TRANSFORM
        #background_copy = cv2.cvtColor(background_copy, cv2.COLOR_BGR2GRAY)
        #background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        # registered_background = cv2.warpPerspective(background_copy,M[0],background.shape)
        registered_background = cv2.warpAffine(background_copy, M, background.shape)

        # --------------------------------------------------
        # overlay images
        # --------------------------------------------------
        alpha = .7
        colors = [[150, 0, 150], [0, 255, 0]]
        color_array = make_color_array(colors, background.shape)

        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB))
                                # * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB))
                      # * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        cv2.imshow('registered background', overlaid_arenas)

        # --------------------------------------------------
        # initialize GUI for correcting transform
        # --------------------------------------------------
        print('\nLeft click model arena // Right click model background')
        print('Purple within arena and green along the boundary represent the model arena')
        print('Advanced users: use arrow keys and \'wasd\' to fine-tune translation and scale as a final step')
        print('Crème de la crème: use \'tfgh\' to fine-tune shear\n')
        print('y: save and use transform')
        print('r: reset transform (left and right click four points to recommence)')
        update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M, background_copy]

        # create functions to react to additional clicked points
        cv2.setMouseCallback('registered background', additional_transform_points, update_transform_data)

        # take in clicked points until 'q' is pressed
        initial_number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
        M_initial = M
        M_indices = [(0,2),(1,2),(0,0),(1,1),(0,1),(1,0),(2,0),(2,2)]
        # M_indices_meanings = ['x-translate','y-translate','x-scale','y-scale','x->y shear','y->x shear','x perspective','y perspective']
        M_mod_keys = [2424832, 2555904, 2490368, 2621440, ord('w'), ord('a'), ord('s'), ord('d'), ord('f'), ord('t'),
                      ord('g'), ord('h'), ord('j'), ord('i'), ord('k'), ord('l')]
        while True:
            cv2.imshow('registered background',overlaid_arenas)
            cv2.imshow('background', registered_background)
            number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
            update_transform = False
            k = cv2.waitKey(10)
            # If a left and right point are clicked:
            if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
                initial_number_clicked_points = number_clicked_points
                # update transform and overlay images
                try:
                    # M = cv2.findHomography(update_transform_data[1], update_transform_data[2])
                    M = cv2.estimateRigidTransform(update_transform_data[1], update_transform_data[2],False) #True ~ full transform
                    update_transform = True
                except:
                    continue
            elif k in M_mod_keys: # if an arrow key if pressed
                if k == 2424832: # left arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] - abs(M_initial[M_indices[0]]) * .005
                elif k == 2555904: # right arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] + abs(M_initial[M_indices[0]]) * .005
                elif k == 2490368: # up arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] - abs(M_initial[M_indices[1]]) * .005
                elif k == 2621440: # down arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] + abs(M_initial[M_indices[1]]) * .005
                elif k == ord('a'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] + abs(M_initial[M_indices[2]]) * .005
                elif k == ord('d'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] - abs(M_initial[M_indices[2]]) * .005
                elif k == ord('s'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] + abs(M_initial[M_indices[3]]) * .005
                elif k == ord('w'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] - abs(M_initial[M_indices[3]]) * .005
                elif k == ord('f'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] - abs(M_initial[M_indices[4]]) * .005
                elif k == ord('h'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] + abs(M_initial[M_indices[4]]) * .005
                elif k == ord('t'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] - abs(M_initial[M_indices[5]]) * .005
                elif k == ord('g'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] + abs(M_initial[M_indices[5]]) * .005

                update_transform = True

            elif  k == ord('r'):
                print('Transformation erased')
                update_transform_data[1] = np.array(([],[])).T
                update_transform_data[2] = np.array(([],[])).T
                initial_number_clicked_points = [3,3]
            elif k == ord('q') or k == ord('y'):
                print('Registration completed')
                break

            if update_transform:
                update_transform_data[3] = M
                # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
                registered_background = cv2.warpAffine(background_copy, M, background.shape)
                registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                               * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
                overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
                update_transform_data[0] = overlaid_arenas

        overlaid_arenas, roi = crop_to_roi(overlaid_arenas)

        np.save(str(file_num+1)+'_transform',[M, update_transform_data[1], update_transform_data[2], fisheye_map_location])

    #selection = False
    # Empty Region of Interest Python List
    # roi = [x1, y1, x2, y2]

    #roi = []
    #if crop_video:


    cv2.destroyAllWindows()
    return [M, update_transform_data[1], update_transform_data[2], fisheye_map_location], x_offset, y_offset, roi

# mouse callback function I
def select_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
        data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[1] = np.concatenate((data[1], clicks))

# mouse callback function II
def additional_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_RBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (200,0,0), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        M_inverse = cv2.invertAffineTransform(data[3])
        transformed_clicks = np.matmul(np.append(M_inverse,np.zeros((1,3)),0), [x, y, 1])
        # M_inverse = np.linalg.inv(data[3])
        # M_inverse = cv2.findHomography(data[2][:len(data[1])], data[1])
        # transformed_clicks = np.matmul(M_inverse[0], [x, y, 1])

        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)

        clicks = np.reshape(transformed_clicks[0:2],(1,2))
        data[1] = np.concatenate((data[1], clicks))
    elif event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (0,200,200), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[2] = np.concatenate((data[2], clicks))

def make_color_array(colors, image_size):
    color_array = np.zeros((image_size[0],image_size[1], 3, len(colors)))  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = np.ones((image_size[0],image_size[1])) * colors[c][i] / sum(
                colors[c])
    return color_array

def crop_to_roi(frame):
    f = frame[:, :, 0]

    r = cv2.selectROI(f)

    # Crop video
    imCrop = f[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    #imCrop = imCrop[:, :]
    # Display cropped image
    #cv2.imshow('Cropped Image', imCrop)

    return imCrop, r

# =================================================================================
#              GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE
# =================================================================================
def peri_stimulus_video_clip(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
                             registration = 0, x_offset = 0, y_offset = 0, dark_threshold = [.55, 950],
                             fps=False, save_clip = False, display_clip = False, counter = True, make_flight_image = True):
    # GET BEAHVIOUR VIDEO - ######################################
    vid = cv2.VideoCapture(vidpath)
    roi = []
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SETUP VIDEO CLIP SAVING - ######################################
    # file_already_exists = os.path.isfile(os.path.join(savepath,videoname+'.avi'))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    border_size = 20
    if save_clip:
        video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size*counter, height+2*border_size*counter), counter)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pre_stim_color = [255, 120, 120]
    post_stim_color = [120, 120, 255]

    if registration[3]: # setup fisheye correction (registration[3] is the file location of the fisheye_maps.npy)
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0
    else:
        print(colored('Fisheye correction unavailable', 'green'))

    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    count = 0
    while True: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame

        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
            if [registration]:
                # load the fisheye correction
                frame_register = frame[:, :, 0]
                if registration[3]:
                    frame_register = cv2.copyMakeBorder(frame_register, x_offset, int((map1.shape[0] - frame.shape[0]) - x_offset),
                                                         y_offset, int((map1.shape[1] - frame.shape[1]) - y_offset), cv2.BORDER_CONSTANT, value=0)
                    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    frame_register = frame_register[x_offset:-int((map1.shape[0] - frame.shape[0]) - x_offset),
                                      y_offset:-int((map1.shape[1] - frame.shape[1]) - y_offset)]
                frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)


            if count == 0:
                print('FIRST FRAME -- ENTERING CROP')
                frame, roi = crop_to_roi(frame)
                count += 1
                print('1 added to count. Count =', count)

            frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]


            # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
            if make_flight_image:
                # at stimulus onset, take this frame to lay all the superimposed mice on top of
                if frame_num == stim_frame:
                    flight_image_by_distance = frame[:,:,0].copy()

                # in subsequent frames, see if frame is different enough from previous image to merit joining the image
                elif frame_num > stim_frame and (frame_num - stim_frame) < 30*10:
                    # get the number of pixels that are darker than the flight image
                    difference_from_previous_image = ((frame[:,:,0]+.001) / (flight_image_by_distance+.001))<dark_threshold[0] #.5 original parameter
                    number_of_darker_pixels = np.sum(difference_from_previous_image)

                    # if that number is high enough, add mouse to image
                    if number_of_darker_pixels > dark_threshold[1]: # 850 original parameter
                        # add mouse where pixels are darker
                        flight_image_by_distance[difference_from_previous_image] = frame[difference_from_previous_image,0]

            # SHOW BOUNDARY AND TIME COUNTER - #######################################
            if counter and (display_clip or save_clip):
                # cv2.rectangle(frame, (0, height), (150, height - 60), (150,150,150), -1)
                if frame_num < stim_frame:
                    cur_color = tuple([x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
                    sign = ''
                else:
                    cur_color = tuple([x * (1 - (frame_num - stim_frame) / (end_frame-stim_frame))  for x in post_stim_color])
                    sign = '+'

                # border and colored rectangle around frame
                frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size,cv2.BORDER_CONSTANT, value=cur_color)

                # report video details
                cv2.putText(frame, videoname, (20, 40), 0, .55, (180, 180, 180), thickness=1)

                # report time relative to stimulus onset
                frame_time = (frame_num - stim_frame) / fps
                frame_time = str(round(.2*round(frame_time/.2), 1))+ '0'*(abs(frame_time)<10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width-110, height+10), 0, 1,(180,180,180), thickness=2)

            else:
                frame = frame[:,:,0] # or use 2D grayscale image instead

            # SHOW AND SAVE FRAME - #######################################
            if display_clip:
                cv2.imshow('Trial Clip', frame)
            if save_clip:
                video_clip.write(frame)
            if display_clip:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if frame_num >= end_frame:
                break
        else:

            print('Problem with movie playback')
            cv2.waitKey(1000)
            break

    # wrap up
    vid.release()
    if make_flight_image:
        flight_image_by_distance = cv2.copyMakeBorder(flight_image_by_distance, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(flight_image_by_distance, videoname, (border_size, border_size-5), 0, .55, (180, 180, 180), thickness=1)
        cv2.imshow('Flight image', flight_image_by_distance)
        cv2.waitKey(10)
        #
        # scipy.misc.imsave(os.path.join(savepath, videoname + '.tif'), flight_image_by_distance)
    if save_clip:
        video_clip.release()
    # cv2.destroyAllWindows()

def whole_video_clip(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=-1.,
                             registration = 0, x_offset = 0, y_offset = 0, dark_threshold = [.55, 950], show_video = False,
                             roi=False,
                             fps=False, save_clip = False, display_clip = False, counter = False):

    # GET BEAHVIOUR VIDEO - ######################################
    vid = cv2.VideoCapture(vidpath)

    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)

    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SETUP VIDEO CLIP SAVING - ######################################
    # file_already_exists = os.path.isfile(os.path.join(savepath,videoname+'.avi'))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if registration[3]: # setup fisheye correction (registration[3] is the file location of the fisheye_maps.npy)
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0
    else:
        print(colored('Fisheye correction unavailable', 'green'))

    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    count = 0
    while True: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame
        n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = n_frames
        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
            if [registration]:
                # load the fisheye correction
                frame_register = frame[:, :, 0]
                if registration[3]:

                    if int((map1.shape[0] - frame.shape[0]) - y_offset) < 0:
                        print('Y OFFSET', y_offset, 'TOO BIG. CHANGING IT BY APPROPRIATE AMOUNT')

                        #print(
                        #    'Y OFFSET {} TOO BIG BY {}. CHANGING IT BY APPROPRIATE AMOUNT').format(y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset))
                        y_offset = y_offset + (int((map1.shape[0] - frame.shape[0]) - y_offset))

                    if int((map1.shape[1] - frame.shape[1]) - x_offset) < 0:
                        print('X OFFSET', x_offset, 'TOO BIG. CHANGING IT BY APPROPRIATE AMOUNT')

                        #    'X OFFSET {} TOO BIG BY {}. CHANGING IT BY APPROPRIATE AMOUNT').format(x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset))
                        x_offset = x_offset + (int((map1.shape[1] - frame.shape[1]) - x_offset))

                    frame_register = cv2.copyMakeBorder(frame_register, y_offset,
                                                        int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                        x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset),
                                                        cv2.BORDER_CONSTANT, value=0)
                    
                    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                                      x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]

                frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB) #frame.shape[0:2]

            frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

            video_writer_shape = frame.shape[1], frame.shape[0]
            if count == 0:
                video_clip = cv2.VideoWriter(os.path.join(savepath, videoname + '.avi'), fourcc, fps,
                                         (frame.shape[1], frame.shape[0]), counter)
                print('FRAME.SHAPE:', frame.shape)
                print('Video writer shape: ', video_writer_shape)
                count+=1

            # SHOW AND SAVE FRAME - #######################################
            if show_video:
                cv2.imshow('Final Video {}'.format(savepath), frame)

            video_clip.write(frame)

            if show_video:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if frame_num >= end_frame:
                break
        else:
            print('Problem with movie playback')
            print('PROBLEM FOLDER:', vidpath)
            cv2.waitKey(1000)
            break

    # wrap up
    vid.release()
    video_clip.release()

    # cv2.destroyAllWindows()

def extract_trial_clips(vidpath = '', videoname = '', savepath = '', stim_indices=[], pre_window=30, post_window=30,
                        produce_txt = True, fps=False, save_clip = True, show_video = False, overwrite_previous = True,
                        counter = False):

    ### Takes a video, a list of stimulus indices, a pre and post window (in seconds) and exports video clips,
    ### with a .txt file with settings used to produce them

    # Get video
    vid = cv2.VideoCapture(vidpath)

    # Setup the video writing settings
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # If frames per second not defined, use the same FPS as the video
    if not fps: fps = int(vid.get(cv2.CAP_PROP_FPS))
    print('FPS:', fps)

    end_clip_frame = None
    write_clip = False
    # Now cycle through the video
    while True:
        ret, frame = vid.read()

        # Get n_frames of video, and remove any indices beyond the length of the video
        n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        stim_indices = [x for x in stim_indices if x < n_frames]

        # Produce lists of start and end indices for our clips
        start_indices = [x - fps * pre_window for x in stim_indices]
        end_indices = [x + fps * post_window for x in stim_indices]

        for ind in end_indices:
            if ind > n_frames:
                end_ind = end_indices.index(ind)
                end_indices[end_ind] = n_frames
                print('End of clip cut short as it went beyond vid length')

        # Will need to produce a dictionary in case we have overlapping video clips

        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)


            if frame_num in start_indices:
                index = start_indices.index(frame_num)
                end_clip_frame = end_indices[index]
                clip_name = videoname + '_' + str(start_indices[index])
                if os.path.isfile(vidpath+'\\'+clip_name):
                    if overwrite_previous:
                        trial_clip = cv2.VideoWriter(os.path.join(savepath, clip_name + '.avi'), fourcc, fps,
                                             (frame.shape[1], frame.shape[0]), counter)
                        write_clip = True

                elif not os.path.isfile(vidpath+'\\'+clip_name):
                    trial_clip = cv2.VideoWriter(os.path.join(savepath, clip_name + '.avi'), fourcc, fps,
                                                 (frame.shape[1], frame.shape[0]), counter)
                    write_clip = True


            if frame_num in end_indices:
                write_clip = False

            if show_video:
                cv2.imshow('Trial Clip {}'.format(savepath+clip_name), frame)

            if write_clip:
                trial_clip.write(frame)

            if not write_clip:
                end_clip_frame = n_frames

            if show_video:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if frame_num > end_clip_frame:
                print('Number of frames:', n_frames)
                print('End clip frame:', end_clip_frame)
                #Drop trial clip
                trial_clip.release()

                break
        else:

            print('Problem with movie playback')
            print('PROBLEM FOLDER:', vidpath)
            cv2.waitKey(1000)
            break
    
    if produce_txt:
        # Write a .txt file with key settings and save it in same folder as the clips
        txt_file_name = savepath + '\\_clip_settings.txt'
        txt_file = open(txt_file_name, 'w+')
        txt_file.write('Original file:{}'.format(vidpath + videoname))
        txt_file.write('FPS:{}'.format(fps))
        txt_file.write('Prewindow (s):{}'.format(pre_window))
        txt_file.write('Postwindow (s):{}'.format(post_window))
        txt_file.write('Stimulis used:{}'.format(str(stim_indices[:])))
        txt_file.close()
    
    vid.release()

#def get_cmap(n, name=colormap):
#    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
#    RGB color; the keyword argument name must be a standard mpl colormap name.'''
#    return plt.cm.get_cmap(name, n)

def make_labelled_video(clip, Dataframe):
    ''' Creating individual frames with labeled body parts and making a video'''
    scorer = np.unique(Dataframe.columns.get_level_values(0))[0]
    bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))
    colors = get_cmap(len(bodyparts2plot))

    ny, nx = clip.size  # dimensions of frame (height, width)
    fps = clip.fps
    nframes = len(Dataframe.index)
    if cropping:
        # one might want to adjust
        clip = clip.crop(y1=y1, y2=y2, x1=x1, x2=x2)
    clip.reader.initialize()
    print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
          "fps!")
    print("Overall # of frames: ", nframes, "with cropped frame dimensions: ",
          clip.size)
    print("Generating frames")
    for index in tqdm(range(nframes)):

        imagename = tmpfolder + "/file%04d.png" % index
        if os.path.isfile(tmpfolder + "/file%04d.png" % index):
            pass
        else:
            plt.axis('off')
            image = img_as_ubyte(clip.reader.read_frame())
            # image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))

            if np.ndim(image) > 2:
                h, w, nc = np.shape(image)
            else:
                h, w = np.shape(image)

            plt.figure(frameon=False, figsize=(w * 1. / 100, h * 1. / 100))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.imshow(image)

            for bpindex, bp in enumerate(bodyparts2plot):
                if Dataframe[scorer][bp]['likelihood'].values[index] > pcutoff:
                    plt.scatter(
                        Dataframe[scorer][bp]['x'].values[index],
                        Dataframe[scorer][bp]['y'].values[index],
                        s=dotsize ** 2,
                        color=colors(bpindex),
                        alpha=alphavalue)

            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()
            plt.savefig(imagename)

            plt.close("all")

    os.chdir(tmpfolder)

    print("Generating video")
    subprocess.call([
        'ffmpeg', '-framerate',
        str(clip.fps), '-i', 'file%04d.png', '-r', '30', '../' + vname + '_DeepLabCutlabeled.mp4'])
    if deleteindividualframes:
        for file_name in glob.glob("*.png"):
            os.remove(file_name)

    os.chdir("../")


########################################################################################################################
if __name__ == "__main__":
    if correct_whole_video:
        whole_video_clip(video_file_path, videoname, save_file_path, start_frame, end_frame, registration, x_offset,
                         y_offset, dark_threshold, save_clip=True, display_clip=True, counter=True)
    
    elif not correct_whole_video:
        peri_stimulus_video_clip(vidpath='', videoname='', savepath='', start_frame=0., end_frame=100., stim_frame=0,
                                registration=0, fps=False, save_clip=False, display_clip=False, counter=True,
                                make_flight_image=True)