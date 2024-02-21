#!/bin/bash
cd /Users/oliverlemke/Documents/University/2023-24/ext-projects/spot-mask-3d
source venv/bin/activate
cd source
p() {
  python -m "scripts.my_robot_scripts.$1"
}
clear
