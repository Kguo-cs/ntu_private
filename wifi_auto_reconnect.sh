#!/bin/bash

# Name of your Wi-Fi interface (usually wlan0 or similar)
WIFI_INTERFACE="wlo1"

# Your Wi-Fi SSID
SSID="NTUSECURE"

# Check if Wi-Fi is connected
if ! nmcli -t -f DEVICE,STATE dev | grep -q "${WIFI_INTERFACE}:connected"; then
    echo "$(date): Wi-Fi is disconnected. Attempting to reconnect..."
    nmcli dev wifi connect "$SSID" ifname "$WIFI_INTERFACE"
else
    echo "$(date): Wi-Fi is connected."
fi
