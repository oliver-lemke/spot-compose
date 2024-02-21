#!/bin/bash

scp spot:/home/cvg-nuc-1/oliver_ws/pcd_body.ply /Users/oliverlemke/Documents/University/2023-24/ext-projects/spot-mask-3d/data/tmp
scp spot:/home/cvg-nuc-1/oliver_ws/pcd_seed.ply /Users/oliverlemke/Documents/University/2023-24/ext-projects/spot-mask-3d/data/tmp
scp spot:/home/cvg-nuc-1/oliver_ws/pcd.ply /Users/oliverlemke/Documents/University/2023-24/ext-projects/spot-mask-3d/data/tmp

cd ../..
python -m scripts.temp_scripts.vis_spot_pcd