import numpy as np
import matplotlib.pyplot as plt

# ---- helpers ----
def _to_xyz(arr):
    a = np.asarray(arr, float)
    if a.ndim != 2: a = np.asarray(a).reshape(-1, 3)
    if a.shape[1] == 2: a = np.column_stack([a, np.zeros(len(a))])
    return a

def _plot_poly(ax, P, **kw):
    P = _to_xyz(P)
    ax.plot(P[:,0], P[:,1], **kw)

def _arrow_from_heading(ax, xy, heading_rad, length=5.0, **kw):
    x,y = float(xy[0]), float(xy[1])
    dx, dy = length*np.cos(heading_rad), length*np.sin(heading_rad)
    ax.arrow(x, y, dx, dy, head_width=length*0.25, head_length=length*0.3, length_includes_head=True, **kw)

# ---- main ----
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_agents_on_map(agents, map_infos, show_headings=True):
    for a in agents:
        fig, ax = plt.subplots(figsize=(10,8))

        cls = a.get("cls","agent")
        Ps = np.asarray(a["start_xyz"], float)
        Pr = np.asarray(a["route_xyz"], float)
        Pg = np.asarray(a["goal_xyz"], float)

        ax.plot(Pr[:,0], Pr[:,1], lw=2, color="tab:blue", label=f"{cls} route")
        ax.plot(Ps[0], Ps[1], "go", ms=8, label="start")
        ax.plot(Pg[0], Pg[1], "rx", ms=8, label="goal")

        if show_headings and "start_heading_rad" in a:
            hd = float(a["start_heading_rad"])
            ax.arrow(Ps[0], Ps[1], 4*np.cos(hd), 4*np.sin(hd),
                     head_width=1.0, head_length=1.5, color="green")

        ax.set_aspect("equal","box")
        ax.set_title(f"Agent {a['agent_id']} ({cls})")
        ax.legend(loc="upper right")
        plt.show()  # blocks until you close the window
