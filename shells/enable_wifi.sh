export ifname=wlp4s0
export ROBOT_SSID=spot-BD-40750010
export ROBOT_PASSWORD=eyuto043vrnc
export ROBOT_NETWORK=192.168.80.0
export ROBOT_IP_ADDRESS=192.168.80.3

sudo ip link set ${ifname} up
sudo nmcli device wifi connect ${ROBOT_SSID} password ${ROBOT_PASSWORD} ifname ${ifname}
ip addr show wlp4s0
sudo ip route add ${ROBOT_NETWORK}/24 dev ${ifname}
ping -I wlp4s0 ${ROBOT_IP_ADDRESS}
