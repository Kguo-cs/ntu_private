#!/bin/bash

# Name of your Wi-Fi interface (usually wlan0 or similar)
WIFI_INTERFACE="wlp100s0"

# Your Wi-Fi SSID
SSID="NTUSECURE"

# 1. Check connectivity
if ! curl -s --max-time 5 https://www.google.com >/dev/null 2>&1; then
    echo "$(date): Cannot reach Google. Attempting to recover Wi-Fi..."

    # 2. Ensure Wi-Fi is enabled
    sudo nmcli radio wifi on

    # 3. If device is unmanaged, fix it
    if nmcli -t -f DEVICE,STATE dev | grep -q "^${WIFI_INTERFACE}:unmanaged"; then
        echo "$(date): Wi-Fi device is unmanaged. Re-managing it..."
        sudo nmcli device set "$WIFI_INTERFACE" managed yes
    fi

    # 4. Reset interface (important for driver glitches)
    sudo ip link set "$WIFI_INTERFACE" down
    sleep 2
    sudo ip link set "$WIFI_INTERFACE" up

    # 5. Restart NetworkManager
    sudo systemctl restart NetworkManager
    sleep 5

    # 6. Reconnect
    sudo nmcli dev wifi connect "$SSID" ifname "$WIFI_INTERFACE"

else
    echo "$(date): Google is reachable."
fi

