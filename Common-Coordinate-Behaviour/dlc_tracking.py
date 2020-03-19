import os.path
import sys
from loadsave_funcs import load_paths, load_yaml
paths = load_paths()

dlc_folder = paths['DLC folder']

# add parent directory: (where nnet & config are!)
sys.path.append(os.path.join(dlc_folder, "pose-tensorflow"))
sys.path.append(os.path.join(dlc_folder, "Generating_a_Training_Set"))

import default_config

cfg = default_config.cfg

from nnet import predict
from dataset.pose_dataset import data_to_input
import pickle
import imageio
imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import skimage
import yaml
import skimage.color
import time
from easydict import EasyDict as edict
import pandas as pd
import numpy as np
import os
import logging
import pprint
from tqdm import tqdm

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        #if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config from file filename and merge it into the default options.
    """
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, cfg)
    logging.info("Config:\n"+pprint.pformat(cfg))
    return cfg

def load_config(path, filename = "pose_cfg.yaml"):
    filename = os.path.join(path, filename)
    return cfg_from_file(filename)

def dlc_setupTF(dlc_config_settings):

    cfg = load_config(dlc_config_settings['dlc_network_posecfg'])
    cfg['init_weights'] = dlc_config_settings['dlc_network_snapshot']
    scorer = dlc_config_settings['scorer']
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    return {'scorer': scorer, 'sess': sess, 'inputs': inputs, 'outputs': outputs, 'cfg': cfg}

# TODO: Review getpose
def getpose(sess, inputs, image, cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose


def analyse(tf_setting, videofolder:str, video_name):
    """  analyse the video passed"""
    # Load TENSORFLOW settings
    cfg = tf_setting['cfg']
    scorer = tf_setting['scorer']
    sess = tf_setting['sess']
    inputs = tf_setting['inputs']
    outputs = tf_setting['outputs']

    pdindex = pd.MultiIndex.from_product(
        [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])
    frame_buffer = 10


    # TODO: Check if changing the directory like this is necessary
    os.chdir(videofolder)

    # TODO: Set a dataname better than this
    # TODO: Add a .txt that gives the scorer etc, rather than having it in the name of the file

    # OLD VERSION: dataname = video.split('.')[0] + scorer + '.h5'

    dataname = 'cam1_FEC' + scorer + '.h5'

    video = videofolder + '\\' + video_name
    try:
        # Attempt to load data...
        pd.read_hdf(dataname)
        print("            ... video already analyzed!", dataname)

    except FileNotFoundError:
        print("                 ... loading ", video)

        # Load clip and extract info

        clip = VideoFileClip(video)
        ny, nx = clip.size  # dimensions of frame (height, width)
        fps = clip.fps
        nframes_approx = int(np.ceil(clip.duration * clip.fps) + frame_buffer)

        start = time.time()

        PredicteData = np.zeros((nframes_approx, 3 * len(cfg['all_joints_names'])))

        temp_image = img_as_ubyte(clip.get_frame(0))

        scmap, locref, pose = getpose(sess, inputs, temp_image, cfg, outputs, outall=True)

        PredictedScmap = np.zeros((nframes_approx, scmap.shape[0], scmap.shape[1], len(cfg['all_joints_names'])))

        for index in tqdm(range(nframes_approx)):

            image = img_as_ubyte(clip.reader.read_frame())

            if index == int(nframes_approx - frame_buffer * 2):
                last_image = image

            elif index > int(nframes_approx - frame_buffer * 2):
                if (image == last_image).all():
                    nframes = index
                    print("Detected frames: ", nframes)
                    break
                else:
                    last_image = image
            try:
                pose = getpose(sess, inputs,image, cfg, outputs, outall=True)
                PredicteData[index, :] = pose.flatten()

            except:
                scmap, locref, pose = getpose(sess, inputs, image, cfg, outputs, outall=True)
                PredicteData[index, :] = pose.flatten()
                PredictedScmap[index, :, :, :] = scmap

        stop = time.time()

        txt_name = videofolder + '\\' + 'DLC_tracking_settings.txt'
        txt_file=open(txt_name, 'w+')
        txt_file.write("start:")
        txt_file.write(str(start))
        txt_file.write("stop:")
        txt_file.write(str(stop))
        txt_file.write("run_duration:")
        txt_file.write(str(stop - start))
        txt_file.write("Scorer:")
        txt_file.write(str(scorer))
        txt_file.write("config file:")
        txt_file.write(str(cfg))
        txt_file.write("fps:")
        txt_file.write(str(fps))
        txt_file.write("frame_dimensions:")
        txt_file.write(str(ny))
        txt_file.write(str(nx))
        txt_file.write("nframes:")
        txt_file.write(str(nframes))

        txt_file.close()

        print("Saving results...")
        DataMachine = pd.DataFrame(PredicteData[:nframes, :], columns=pdindex,
                                   index=range(nframes))  # slice pose data to have same # as # of frames.
        DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')

        #with open(dataname.split('.')[0] + 'includingmetadata.pickle',
        #          'wb') as f:
        #    pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

        #except:
        #    from warnings import warn
        #    warn('Could not do DLC tracking on video {}'.format(video))
