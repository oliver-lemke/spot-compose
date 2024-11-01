#!/bin/bash
DIR="$(dirname "$(realpath "$0")")"
SPOT_COMPOSE_DIR="$(dirname "$(realpath "$DIR")")" 

cd ${SPOT_COMPOSE_DIR}
source venv/bin/activate
cd source
p() {
  python -m "scripts.my_robot_scripts.$1"
}
ret() {
  python -m "scripts.my_robot_scripts.return_to_start"
}
