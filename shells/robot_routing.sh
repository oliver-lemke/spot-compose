#!/bin/bash
export ROS_MASTER_URI=http://192.168.1.24:11311/
export ROS_IP=192.168.1.243
export ROS_HOSTNAME=192.168.1.243
eval $(ssh-agent)
ssh-add ~/.ssh/id_ed25519
sudo nmcli c up id orin ifname enp88s0
sudo nmcli c up id spot_payload  ifname enxa0cec8e56d7c
sudo iptables -P FORWARD ACCEPT
sudo iptables -t nat -A POSTROUTING -o enp88s0 -j MASQUERADE
sudo iptables -t nat -A POSTROUTING -o wlo1 -j MASQUERADE
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o enxa0cec8e56d7c -j MASQUERADE