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
num_star      = 1000
pos_width     = 16
pos_height    = 12
display_scale = 0.5
grav_const    = 1
size_scale    = 1

model    = 'circle'
radius   = min(pos_width, pos_height) * display_scale / 8
weight   = 1

arg = [nbfmm, output, width, height, fps, num_second, \
       fmm_level, num_star, pos_width, pos_height, \
       display_scale, grav_const, size_scale, \
       model, radius, weight]
cmd = ' '.join(str(a) for a in arg)
os.system(cmd)
