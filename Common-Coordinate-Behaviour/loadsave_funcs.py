import pandas as pd
import yaml
import os
from warnings import warn
from termcolor import colored

def load_yaml(fpath):
    """ load settings from a yaml file and return them as a dictionary """
    with open(fpath, 'r') as f:
        settings = yaml.load(f)
    return settings

def load_paths():
    """ load PATHS.yml to set all the user-specific paths correctly """
    filename = './PATHS.yml'
    return load_yaml(filename)

