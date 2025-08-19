import json
import matplotlib.pyplot as plt

# Load the JSON file
with open("./waymo/sensor_json/4D_label_ori.json", "r") as f:
    data = json.load(f)

lane_lines = data["annotation"]["lines"]

plt.figure(figsize=(12, 12))
for line in lane_lines:
    x, y, z = line["xyz"]

    attrs = line.get("attrs", {})

    # safely fetch attributes with defaults
    color = attrs.get("laneline_color", "blue")  # fallback if missing
    linetype = attrs.get("laneline_type", "solid")
    linestyle = "-" if linetype == "solid" else "--"

    plt.plot(x, y, color=color, linestyle=linestyle, label=f"ID {line['global_id']}")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Lane Lines from 4D Label JSON")
plt.axis("equal")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

