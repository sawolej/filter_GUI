import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

data = None

data_with_mask_inside = None
data_with_mask_outside = None

def onselect(verts):
    global data, data_with_mask_inside, data_with_mask_outside
    path = Path(verts)
    mask_inside = np.zeros_like(data, dtype=bool)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if path.contains_point((j, i)):
                mask_inside[i, j] = True

    mask_outside = ~mask_inside
    data_with_mask_inside = np.where(mask_inside, data, np.nan)
    data_with_mask_outside = np.where(mask_outside, data, np.nan)


def select_area(come):
    global data, data_with_mask_inside, data_with_mask_outside
    data = come
    fig, ax = plt.subplots()
    ax.imshow(data, cmap='gray')

    polygon_selector = PolygonSelector(ax, onselect)

    plt.show()

    return data_with_mask_inside, data_with_mask_outside
