import plotly.graph_objects as go
import numpy as np


def  bell_curve(x, y,
                sigma=1,
                mu = 3,
                ):
    z = (1/np.sqrt(2*np.pi*sigma))
    exp = (x - mu)**2 + (y-mu)**2
    exp = exp/(2*(sigma)**2)
    exp = -exp
    z = 2.8*z* np.exp(exp)
    return z

x = np.linspace(5, 8, 30)
y = np.linspace(5, 8, 30)

z = []
for _x in x:
    _z = []
    for _y in x:
        _z.append(bell_curve(_x, _y))
    z.append(_z)


fig = go.Figure(
        go.Surface(
            x = x,
            y = y,
            z = z,
        ))

fig.update_layout(scene = {
            'xaxis_title': 'Opportunity',
            'yaxis_title': 'Flexibility',
            'zaxis_title': 'QoL',
            "xaxis": {"nticks": 1},
            "zaxis": {"nticks": 1},
            'camera_eye': {"x": 0, "y": 0.5, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 1/3},
            'xaxis': dict(showbackground=False, visible=False, showgrid=False, showticklabels=False),
            'yaxis': dict(showbackground=False, visible=False, showgrid=False, showticklabels=False),
            'zaxis': dict(showbackground=False, visible=False, showgrid=False, showticklabels=False),
        })
fig.show()