#!/bin/bash
cd /Users/oliverlemke/Documents/University/2024/ext-projects/spot-drawers
source venv/bin/activate
cd source
p() {
  python -m "scripts.my_robot_scripts.$1"
}
ret() {
  python -m "scripts.my_robot_scripts.return_to_start"
}
clear
