from Acrobot import Acrobot
from Model import Model

import sys
import numpy as np
import tensorflow as tf
from time import sleep

env = Acrobot(
    gui=True,
    num_frames=1,
    length2_min=1.,
    length2_max=1.,
    length1_min=1.,
    length1_max=1.,
)
obv_space = env.obv_space()

model = Model(
    path=(sys.argv[2] if len(sys.argv) > 2 else 'models/policy'),
    obv_space=obv_space)

obv = env.reset()
while True:
    act = model.predict(obv)
    obv, rew, done, _ = env.step(act)
    print(act, rew)
    if done: obv = env.reset()
