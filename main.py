from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageColor
import random
import numpy as np
from scipy.ndimage import gaussian_filter
import time

#TODO: Runs very slowly, analyze and optimize, maybe precalculate frames
class Rewards:
    def __init__(self, size, init_value="zeros") -> None:
        defined_settings = ["zeros", "random", "diag"]
        assert init_value in defined_settings
        if init_value == "zeros":
            self.values = np.zeros(size)
        elif init_value == "random":
            self.values = np.random.uniform(size=size)
        elif init_value == "diag":
            self.values = np.eye(size[0])

    def diffuse(self, diffusion, evaporation):
        gaussian_filter(self.values, sigma = diffusion)
        self.values *= 1 - evaporation

    def renew_at_position(self, pos, value):
        if all(pos > np.array([0, 0])) and all(pos < np.shape(self.values)):
            self.values[pos[1], pos[0]] = value

    def get_val(self, pos):
        ret = -np.inf
        if all(pos > np.array([0, 0])) and all(pos < np.shape(self.values)):
            ret = self.values[pos[1], pos[0]]
        return ret

    def create_rew_bg(self, size) -> Image:
        r = g = b = 255 * np.copy(self.values)
        channels = np.zeros(np.array([size[0],size[1],3]))
        channels[:,:,0] = r
        channels[:,:,1] = g
        channels[:,:,2] = b
        im = Image.fromarray(channels.astype(np.uint8), mode='RGB')
        return im
class Agent:
    def __init__(self, pos: np.array) -> None:
        self.moves = np.array(
            [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
        )
        self.pos = pos
        self.heading = random.choice(self.moves)

    def get_adjacent_moves(self):
        angle_to_heading = lambda m: np.arccos(
            np.dot(self.heading, m) / (np.linalg.norm(self.heading) * np.linalg.norm(m))
        )
        valid_moves_for_heading = [
            m for m in self.moves if abs(angle_to_heading(m)) < np.pi / 2 - 1e-5
        ]
        assert len(valid_moves_for_heading) == 3
        return valid_moves_for_heading
    def next_move(self, rewards: Rewards) -> None:
        # choose just from moves that are on or adjacent to heading
        if rewards.get_val(self.pos+self.heading) == -np.inf:
            self.heading = -random.choice(self.get_adjacent_moves())

        vals = {
            tuple(m): rewards.get_val(self.pos + m) for m in self.get_adjacent_moves()
        }
        rel = max(vals, key=vals.get)
        rew.renew_at_position(self.pos, 1)
        self.pos += np.array(rel)

    def draw(
        self,
        image: Image,
        clr: str | tuple[int, int, int] = "red",
        size=np.array([8, 8]),
    ) -> None:
        d = ImageDraw.Draw(image)
        clipped_pos = np.clip(self.pos, size, image.size - size)
        bound_box = list(np.concatenate([clipped_pos - size, clipped_pos + size]))
        # print(f"drawing agent at bound box {bound_box}")
        d.ellipse(bound_box, clr)


def update():
    image = rew.create_rew_bg([w,h])#Image.new("RGB", (w, h))
    rew.diffuse(30, 0.0005)
    for ag in agents:
        ag.next_move(rew)
        ag.draw(image)
    view = ImageTk.PhotoImage(image)
    panel.configure(image=view)
    panel.image = view
    root.after(1, update)


if __name__ == "__main__":
    root = Tk()
    w, h = 250, 250
    image = Image.new("RGB", (w, h))
    view = ImageTk.PhotoImage(image)
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    panel = ttk.Label(frm, image=view)
    panel.grid(column=0, row=0)

    rew = Rewards(np.array([w, h]), "random")
    agents = []
    for _ in range(1):
        pos = np.array(np.random.uniform(low=0,high=800,size=2), dtype=np.int16)
        agents.append(Agent(pos))

    root.after(0, update)
    root.mainloop()
