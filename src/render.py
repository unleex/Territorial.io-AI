import matplotlib.pyplot as plt
from typing import Literal
from matplotlib.colors import to_rgb
import numpy as np
import os
import subprocess
import datetime

class GameRenderer:
    def __init__(self, colors,mode: Literal["show","save"] = "show", log_folder = "logs"):
        self.colors = colors
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.mode = mode
        plt.ion()  # Enable interactive mode
        plt.show()
        self.log_folder = log_folder
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
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
        if self.mode == "show":
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        else:   
                
                plt.savefig(self.log_folder + "/file%02d.png" % )

                os.chdir("your_folder")
                subprocess.call([
                    'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                    f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
                ])
                for file_name in :
                    os.remove(file_name)

    def reset(self):
        """Clears attack info and prepares for a new episode."""
        if self.info_text:
            self.info_text.set_text("WAITING FOR START...")
            self.info_text.set_color("black")

        self.im = None
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
