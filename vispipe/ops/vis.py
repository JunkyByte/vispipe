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


@block(tag='vis', max_queue=1, data_type='plot')
class plt_plot:
    def __init__(self):
        self.fig = plt.Figure(figsize=[1.2, 1.2])
        self.ax = self.fig.gca()
        #self.ax.axis('off')
        self.ax.margins(0)
        self.ax.tick_params(axis='x', pad=2, labelsize=6, direction='in', length=2)
        self.ax.tick_params(axis='y', pad=1, labelsize=6, direction='in', length=2, rotation=90)
        self.fig.tight_layout(pad=0.1)
        self.pl, = self.ax.plot([], [])

    def run(self, x, y):
        #self.ax.cla()
        self.pl.set_xdata(x)
        self.pl.set_ydata(y)
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        yield self.fig
