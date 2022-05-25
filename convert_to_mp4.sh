#!/bin/bash
echo Please enter the name of the .mov file with the .mov included e.g. test.mov:
read varname
echo Please enter the name of the output .mp4 file with the .mp4 included e.g. test.mp4:
read varname1
pip install ffmpeg
ffmpeg -i $varname -qscale 0 $varname1
