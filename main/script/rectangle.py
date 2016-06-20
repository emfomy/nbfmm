#!/usr/bin/python

import os
import sys

nbfmm  = sys.argv[1]
output = sys.argv[2]

width      = 1024
height     = 768
fps        = 60
num_second = 10

fmm_level     = 4
num_star      = 10000
pos_width     = 16
pos_height    = 12
display_scale = 0.8
grav_const    = 1
size_scale    = 1

model        = 'rectangle'
model_width  = pos_width * display_scale
model_height = pos_height * display_scale
max_weight   = 16

arg = [nbfmm, output, width, height, fps, num_second, \
       fmm_level, num_star, pos_width, pos_height, \
       display_scale, grav_const, size_scale, \
       model, model_width, model_height, max_weight]
cmd = ' '.join(str(a) for a in arg)
os.system(cmd)
