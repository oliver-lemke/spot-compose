#!/bin/bash
SHELLS_DIR="$(dirname "$(realpath "$0")")"

cd ${SHELLS_DIR}
source start.sh
p estop_nogui
