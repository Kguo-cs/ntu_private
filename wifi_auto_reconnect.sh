#!/bin/bash

# Name of your Wi-Fi interface (usually wlan0 or similar)
WIFI_INTERFACE="wlp100s0"

# Your Wi-Fi SSID
SSID="NTUSECURE"

# Check if Wi-Fi is connected
if ! curl -s --max-time 5 https://www.google.com >/dev/null 2>&1; then
    echo "$(date): Cannot reach Google. Attempting to reconnect Wi-Fi..."
    sudo nmcli radio wifi on
    sudo systemctl restart NetworkManager
    sleep 5
    sudo nmcli dev wifi connect "$SSID" ifname "$WIFI_INTERFACE"
else
    echo "$(date): Google is reachable."
fi
