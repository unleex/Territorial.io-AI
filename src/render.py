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
        self.info_text = self.ax.text(
            1.05,
            0.5,
            "",
            transform=self.ax.transAxes,
            verticalalignment="center",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    def update(
        self,
        field: np.ndarray,
        n_players: int,
        target_player: int,
        attack_amount: float,
    ):
        # 1. Create the RGB buffer
        img = np.zeros((*field.shape, 3))

        # Map Neutral (assuming neutral is -1 or a specific ID, update as needed)
        # Map Players
        for i in range(n_players):
            img[field == i] = to_rgb(self.colors[i])

        # 2. Update the existing window
        if self.im is None:
            self.im = self.ax.imshow(img, interpolation="nearest")
            self.ax.axis("off")
        else:
            self.im.set_data(img)

        # 3. Update Attack Information
        target_color = self.colors[target_player] if target_player != -1 else "gray"

        info_str = (
            f"CURRENT ATTACK\n"
            f"----------------\n"
            f"Target:  {target_player}\n"
            f"Commited:  {attack_amount:.2f}"
        )

        self.info_text.set_text(info_str)
        self.info_text.set_color(target_color)

        # 4. Refresh Canvas
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def reset(self):
        """Clears attack info and prepares for a new episode."""
        if self.info_text:
            self.info_text.set_text("WAITING FOR START...")
            self.info_text.set_color("black")

        self.im = None
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
