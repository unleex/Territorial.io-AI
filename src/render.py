import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np


class GameRenderer:
    def __init__(self, colors):
        self.colors = colors
        self.fig, self.ax = plt.subplots()
        self.im = None
        plt.ion()  # Enable interactive mode
        plt.show()

    def update(self, field: np.ndarray, n_players: int):
        # 1. Create the RGB buffer
        img = np.zeros((*field.shape, 3))

        # Map Others
        for i in range(n_players - 1):
            img[field == i] = to_rgb(self.colors[i])

        # 2. Update the existing window
        if self.im is None:
            # First frame: create the image object
            self.im = self.ax.imshow(img, interpolation="nearest")
            self.ax.axis("off")
        else:
            # Subsequent frames: just update the data
            self.im.set_data(img)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
