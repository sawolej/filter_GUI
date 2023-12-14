import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse

plt.rcParams['figure.dpi'] = 142

data = []

def on_select(eclick, erelease, ax, additional_images):
    global data
    if eclick and erelease:
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        width, height = abs(x2 - x1), abs(y2 - y1)
        center = (((x1 + x2) / 2) +0.6, ((y1 + y2) / 2) +0.6)
        radius = min(width, height) / 2

        ellipse = Ellipse(center, width, height, fill=False, color='red', linewidth=1)
        ax.add_patch(ellipse)
        plt.draw()

        hole_data = [len(data) + 1]  # Hole No.

        for img in additional_images:

            average_value = compute_average_in_circle(center, radius, img)
            hole_data.append(average_value)
            print(average_value)
        hole_data.append(0)
        hole_data.append(0)
        data.append(hole_data)

def compute_average_in_circle(center, radius, image):
    y, x = np.ogrid[-center[1]:image.shape[0]-center[1], -center[0]:image.shape[1]-center[0]]
    mask = x*x + y*y <= radius*radius
    selected_pixels = image[mask]
    return np.mean(selected_pixels)

def display_image_with_hover_info(main_image,file_path, *additional_images):
    fig, ax = plt.subplots()
    ax.imshow(main_image, cmap='gray',aspect='equal',  interpolation='none')

    ellipse_selector = EllipseSelector(ax, lambda eclick, erelease: on_select(eclick, erelease, ax, additional_images),
                                       useblit=True, button=[1], minspanx=5, minspany=5,
                                       spancoords='pixels', interactive=True)

    plt.show()

    input_folder = os.path.dirname(file_path)
    df = pd.DataFrame(data, columns=['Hole No.', 'Real', 'Imag', 'Phase', 'Magnitude', 'mm depth', 'scan high'])
    df.to_excel(os.path.join(input_folder, 'phase_data.xlsx'), index=False, engine='openpyxl')

