import numpy as np
import scipy.special as sps
from matplotlib import pyplot as plt
from func import *


par_dict = {"Aquatic": 0.8,  "Aerial": 0.8, "Game": 0.8, "Medical": 0.8, "Surgical": 0.8}

# "Aquatic", "Aerial", "Game", "Medical", "Surgical"  
datasets = ["Aquatic", "Aerial", "Game", "Medical", "Surgical"]
experiment_root = '/home/xx/FOMO/experiments/full_repeat/owlvit-large-patch14/t1'

wind_max(datasets, experiment_root, par_dict)