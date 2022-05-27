#!/bin/bash
echo Please enter the name of the .mov file without the .mov included:
read varname
ffmpeg -i $varname.mov -qscale 0 $varname.mp4
