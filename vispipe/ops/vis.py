from vispipe import block
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


@block(tag='vis', max_queue=1, data_type='image')
def vis_image(input1):
    yield input1


@block(tag='vis', max_queue=1, data_type='raw')
def vis_text(input1):
    yield input1


@block(tag='vis', max_queue=1, data_type='raw')
def vis_shape(input1):
    yield str(input1.shape)


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
        yield self.fig


@block(tag='vis', max_queue=1, data_type='plot')
class plt_scatter(plot):
    def draw(self, x, y):
        return self.ax.scatter(x, y)

    def update_draw(self, x, y):
        return self.pl.set_offsets(list(zip(x, y)))


@block(tag='vis', max_queue=1, data_type='plot')
class plt_plot(plot):
    def draw(self, x, y):
        return self.ax.plot(x, y)[0]

    def update_draw(self, x, y):
        self.pl.set_data(x, y)
