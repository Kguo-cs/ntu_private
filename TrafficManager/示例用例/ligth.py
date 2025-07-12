import dearpygui.dearpygui as dpg
import time

# --- Config ---
traffic_light_config = {
    "id": "tl_001",
    "position": [34.5, 120.8, 5.0],
    "orientation": "vertical",  # or "horizontal"
    "type": [
        {"color": "red", "shape": "circle"},
        {"color": "yellow", "shape": "circle"},
        {"color": "green", "shape": "circle"},
    ],
    "mode": "periodic",  # "fixed", "periodic", "custom"
    "fixed_state": "red",
    "periodic_schedule": {
        "cycle_time": 20.0,
        "phases": [
            {"duration": 10.0, "lights_on": [{"color": "green", "shape": "circle"}]},
            {"duration": 2.0, "lights_on": [{"color": "yellow", "shape": "circle"}]},
            {"duration": 8.0, "lights_on": [{"color": "red", "shape": "circle"}]},
        ]
    },
    "custom_logic": {
        "enabled": False,
        "script_path": ""
    }
}

# --- Color map ---
color_map = {
    "red": (255, 0, 0, 255),
    "yellow": (255, 255, 0, 255),
    "green": (0, 255, 0, 255),
    "off": (60, 60, 60, 255)
}

# --- Get currently active light(s) ---
def get_current_active_lights():
    if traffic_light_config["mode"] == "fixed":
        return [{"color": traffic_light_config["fixed_state"], "shape": "circle"}]

    elif traffic_light_config["mode"] == "periodic":
        t = time.time() % traffic_light_config["periodic_schedule"]["cycle_time"]
        total = 0
        for phase in traffic_light_config["periodic_schedule"]["phases"]:
            total += phase["duration"]
            if t <= total:
                return phase["lights_on"]
    return [{"color": "red", "shape": "circle"}]  # fallback

# --- Draw the traffic light on the canvas ---
def render_traffic_light():
    dpg.delete_item("drawlist", children_only=True)

    orientation = traffic_light_config["orientation"]
    lights = traffic_light_config["type"]
    active = get_current_active_lights()

    spacing = 100
    radius = 30
    padding = 20
    start_x, start_y = 150, 100

    for i, light in enumerate(lights):
        is_active = any(l["color"] == light["color"] and l["shape"] == light["shape"] for l in active)
        color = color_map[light["color"]] if is_active else color_map["off"]

        if orientation == "vertical":
            x, y = start_x, start_y + i * (2 * radius + padding)
        else:
            x, y = start_x + i * (2 * radius + padding), start_y

        if light["shape"] == "circle":
            dpg.draw_circle(center=[x, y], radius=radius, color=(0, 0, 0, 255),
                            fill=color, thickness=2, parent="drawlist")
        # Optional: handle arrows here

# --- Frame loop callback ---
def on_frame():
    render_traffic_light()
    dpg.set_frame_callback(1, on_frame)  # loop again next frame

# --- GUI Setup ---
dpg.create_context()
dpg.create_viewport(title="Traffic Light Viewer", width=400, height=500)

with dpg.window(label="Traffic Light", width=400, height=500):
    dpg.add_text("Simulated Traffic Light (Live)")
    with dpg.drawlist(width=400, height=400, tag="drawlist"):
        pass

# Start animation
dpg.set_frame_callback(1, on_frame)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
