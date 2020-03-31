from vispipe import block
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
matplotlib.use('Agg')


@block(tag='vis', max_queue=1, data_type='image')
class vis_image:
    def __init__(self, max_size: int = 128):
        """
        Visualize an image.

        Parameters
        ----------
        max_size : int
            The maximum size accepted for the image.
            If the image has an higher resolution (on major dimension) it will be scaled
            to `max_size` while keeping the aspect ratio.
            Using an higher resolution than default can slow down visualization.
        """
        self.max_size = max_size

    def run(self, x):
        x = np.array(x, dtype=np.uint8)  # Cast to int
        if x.ndim in [0, 1, 4]:
            raise Exception('The format image you passed is not visualizable')

        if x.ndim == 2:  # Convert from gray to rgb
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        if x.shape[-1] == 3:  # Add alpha channel
            x = np.concatenate([x, 255 * np.ones((x.shape[0], x.shape[1], 1))], axis=-1)
        if max(x.shape) > self.max_size:  # Force max resolution
            arg = np.argmax(x.shape)
            ratio = x.shape[1 - arg] / x.shape[arg]
            new_size = self.max_size * np.eye[arg] + self.max_size * np.eye[1 - arg] * ratio
            x = cv2.resize(x, tuple(new_size.astype(np.int)))
        return x


@block(tag='vis', max_queue=1, data_type='raw')
def vis_text(input1):
    """
    Visualize raw text.
    """
    return input1


@block(tag='vis', max_queue=1, data_type='raw')
def vis_shape(input1):
    """
    Visualize shape of object (if supported).
    This is equivalent to calling `input.shape` and passing it to a raw visualization buffer.
    """
    return input1.shape


class plot:
    def __init__(self, show_axis: bool = True):
        self.show_axis = show_axis
        self.fig = plt.Figure(figsize=[1.2, 1.2])
        self.ax = self.fig.gca()

        if self.show_axis:
            self.ax.tick_params(axis='x', pad=2, labelsize=6, direction='in', length=2)
            self.ax.tick_params(axis='y', pad=1, labelsize=6, direction='in', length=2, rotation=90)
            self.fig.tight_layout(pad=0.1)
            self.ax.margins(0)
            self.pl = self.draw([], [])
            self.fig.canvas.draw()
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.get_figure().bbox)
        else:
            self.ax.axis('off')
            self.fig.tight_layout(pad=0)
            self.ax.margins(0)

    def draw(self, x, y):
        raise NotImplementedError

    def update_draw(self, x, y):
        raise NotImplementedError

    def run(self, x, y):
        if self.show_axis:
            self.update_draw(x, y)
            if self.ax.get_xlim() != (min(x), max(x)) or self.ax.get_ylim() != (min(y), max(y)):
                self.ax.set_xlim((min(x), max(x)))
                self.ax.set_ylim((min(y), max(y)))
                self.fig.canvas.draw()
                self.bg = self.fig.canvas.copy_from_bbox(self.ax.get_figure().bbox)
            else:
                self.fig.canvas.restore_region(self.bg)
            self.ax.draw_artist(self.pl)
            self.fig.canvas.blit(self.ax.clipbox)
        else:
            self.ax.cla()
            self.draw(x, y)
            self.ax.axis('off')
            self.ax.margins(0)
            self.fig.canvas.draw()

        self.fig.canvas.flush_events()
        return self.fig


@block(tag='vis', max_queue=1, data_type='plot')
class plt_scatter(plot):
    """
    Visualize a scatter plot, equivalent to `matplotlib.pyplot.scatter` on the `x` and `y` inputs.

    Parameters
    ----------
    show_axis : bool
        Whether to show the axis of the plot.
    """
    def draw(self, x, y):
        return self.ax.scatter(x, y)

    def update_draw(self, x, y):
        return self.pl.set_offsets(list(zip(x, y)))


@block(tag='vis', max_queue=1, data_type='plot')
class plt_plot(plot):
    """
    Visualize a plot, equivalent to `matplotlib.pyplot.plot` on the `x` and `y` inputs.

    Parameters
    ----------
    show_axis : bool
        Whether to show the axis of the plot.
    """
    def draw(self, x, y):
        return self.ax.plot(x, y)[0]

    def update_draw(self, x, y):
        self.pl.set_data(x, y)
