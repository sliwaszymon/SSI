import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


line_settings = [
    {
        'line_color': 'blue',
        'line_style': '-',
        'point_style': 'o',
        'point_color': 'red',
    },
    {
        'line_color': 'green',
        'line_style': '--',
        'point_style': '^',
        'point_color': 'purple',
    },
    {
        'line_color': 'orange',
        'line_style': '-.',
        'point_style': 's',
        'point_color': 'blue',
    },
    {
        'line_color': 'red',
        'line_style': ':',
        'point_style': 'd',
        'point_color': 'green',
    },
    {
        'line_color': 'purple',
        'line_style': '-',
        'point_style': 'x',
        'point_color': 'orange',
    },
    {
        'line_color': 'blue',
        'line_style': '--',
        'point_style': 'o',
        'point_color': 'red',
    },
    {
        'line_color': 'green',
        'line_style': '-.',
        'point_style': '^',
        'point_color': 'purple',
    },
    {
        'line_color': 'orange',
        'line_style': ':',
        'point_style': 's',
        'point_color': 'blue',
    },
    {
        'line_color': 'red',
        'line_style': '-',
        'point_style': 'd',
        'point_color': 'green',
    },
    {
        'line_color': 'purple',
        'line_style': '--',
        'point_style': 'x',
        'point_color': 'orange',
    },
]


class Plot:
    def __init__(self, nrows: int = 1, ncols: int = 1) -> None:
        self.nrows, self.ncols = nrows, ncols
        self.series = 0

        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols)
        self.multiplot = False
        if ncols > 1 or nrows > 1:
            self.x_ax, self.y_ax, self.multiplot = 0, 0, True

    def set_title(self, title: str) -> None:
        if self.multiplot:
            self.ax[self.x_ax, self.y_ax].set_title(title)
        else:
            self.ax.set_title(title)

    def set_xlabel(self, xlabel: str) -> None:
        if self.multiplot:
            self.ax[self.x_ax, self.y_ax].set_xlabel(xlabel)
        else:
            self.ax.set_xlabel(xlabel)

    def set_ylabel(self, ylabel: str) -> None:
        if self.multiplot:
            self.ax[self.x_ax, self.y_ax].set_ylabel(ylabel)
        else:
            self.ax.set_ylabel(ylabel)

    def swap_subplot(self, x: int, y: int) -> None:
        self.x_ax, self.y_ax = x, y

    def draw_from_y(self, y: list[int], label: str = '') -> None:
        settings = line_settings[self.series % 10]
        if self.multiplot:
            self.ax[self.x_ax, self.y_ax].plot(y, color=settings['line_color'],
                                               linestyle=settings['line_style'], label=label)
        else:
            self.ax.plot(y, color=settings['line_color'],
                         linestyle=settings['line_style'], label=label)
        self.series += 1

    def draw_from_xy(self, x: list[int], y: list[int], label: str = '') -> None:
        settings = line_settings[self.series % 10]
        if self.multiplot:
            self.ax[self.x_ax, self.y_ax].plot(x, y, color=settings['line_color'],
                                               linestyle=settings['line_style'], label=label)
        else:
            self.ax.plot(x, y, color=settings['line_color'],
                         linestyle=settings['line_style'], label=label)
        self.series += 1

    def draw_points(self, x: list[int], y: list[int], label: str = '') -> None:
        settings = line_settings[self.series % 10]
        if self.multiplot:
            self.ax[self.x_ax, self.y_ax].scatter(x, y, color=settings['point_color'],
                                                  marker=settings['point_style'], label=label)
        else:
            self.ax.scatter(x, y, color=settings['point_color'],
                            marker=settings['point_style'], label=label)
        self.series += 1

    def clean(self) -> None:
        self.__init__(self.nrows, self.ncols)

    def grid(self, on: bool = True, ticks: float | int = 1) -> None:
        if self.multiplot:
            self.ax[self.x_ax, self.y_ax].grid(on)
            self.ax[self.x_ax, self.y_ax].set_xticks(np.arange(-100, 100, ticks))
            self.ax[self.x_ax, self.y_ax].set_yticks(np.arange(-100, 100, ticks))
        else:
            self.ax.grid(on)
            self.ax.set_xticks(np.arange(-100, 100, ticks))
            self.ax.set_yticks(np.arange(-100, 100, ticks))

    def set_legend(self, loc: str = 'upper left', title: str = '', fontsize: str = 'small') -> None:
        if self.multiplot:
            self.ax[self.x_ax, self.y_ax].legend(loc=loc, title=title, fontsize=fontsize)
        else:
            self.ax.legend(loc=loc, title=title, fontsize=fontsize)

    @staticmethod
    def ylim(ylim: tuple[float, int]) -> None:
        plt.ylim(ylim[0], ylim[1])

    @staticmethod
    def xlim(xlim: tuple[float, int]) -> None:
        plt.xlim(xlim[0], xlim[1])

    @staticmethod
    def show() -> None:
        plt.show()


def zad3() -> None:
    plot = Plot(1, 1)
    plot.set_title('Nowy plot')
    plot.set_xlabel('x')
    plot.set_ylabel('y')
    plot.grid()

    # points
    plot.draw_points([-1, 0, 1], [1, 0, 1])

    # x^2-1
    fun_xs = np.linspace(-1, 1, 10)
    fun_ys = fun_xs**2 - 1
    plot.draw_from_xy(list(fun_xs), fun_ys)

    # ellipse
    ellipse_xs = [0, 0.8, 1.4, 1.8, 2, 1.8, 1.4, 0.8, 0, -0.8, -1.4, -1.8, -2, -1.8, -1.4, -0.8, 0]
    ellipse_ys = [2, 1.8, 1.4, 0.8, 0, -0.8, -1.4, -1.8, -2, -1.8, -1.4, -0.8, 0, 0.8, 1.4, 1.8, 2]
    plot.draw_from_xy(ellipse_xs, ellipse_ys)
    plot.show()


def zad4() -> None:
    plot = Plot(ncols=2, nrows=2)

    class_names = ['Setosa', 'Versicolour', 'Virginica']
    labels = ['sepal_length_in_cm', 'sepal_width_in_cm', 'petal_length_in_cm', 'petal_width_in_cm']

    # pozyskanie danych
    df = pd.read_csv('../odczyt_probek/iris.txt', delimiter=r'\s+', header=None)
    class_1 = df[df[4] == 1].values.tolist()
    class_2 = df[df[4] == 2].values.tolist()
    class_3 = df[df[4] == 3].values.tolist()

    # komórka 0-0
    plot.set_title(f'{labels[2]} & {labels[3]}')
    plot.set_xlabel(labels[2])
    plot.set_ylabel(labels[3])
    plot.draw_points([x[2] for x in class_1], [y[3] for y in class_1], class_names[0])
    plot.draw_points([x[2] for x in class_2], [y[3] for y in class_2], class_names[1])
    plot.draw_points([x[2] for x in class_3], [y[3] for y in class_3], class_names[2])
    plot.set_legend()

    # komórka 0-1
    plot.swap_subplot(0, 1)
    plot.set_title(f'{labels[1]} & {labels[3]}')
    plot.set_xlabel(labels[0])
    plot.set_ylabel(labels[3])
    plot.draw_points([x[1] for x in class_1], [y[3] for y in class_1], class_names[0])
    plot.draw_points([x[1] for x in class_2], [y[3] for y in class_2], class_names[1])
    plot.draw_points([x[1] for x in class_3], [y[3] for y in class_3], class_names[2])
    plot.set_legend()

    # komórka 1-0
    plot.swap_subplot(1, 0)
    plot.set_title(f'{labels[0]} & {labels[3]}')
    plot.set_xlabel(labels[0])
    plot.set_ylabel(labels[3])
    plot.draw_points([x[0] for x in class_1], [y[3] for y in class_1], class_names[0])
    plot.draw_points([x[0] for x in class_2], [y[3] for y in class_2], class_names[1])
    plot.draw_points([x[0] for x in class_3], [y[3] for y in class_3], class_names[2])
    plot.set_legend()

    # komórka 1-1
    plot.swap_subplot(1, 1)
    plot.set_title(f'{labels[1]} & {labels[2]}')
    plot.set_xlabel(labels[1])
    plot.set_ylabel(labels[2])
    plot.draw_points([x[1] for x in class_1], [y[2] for y in class_1], class_names[0])
    plot.draw_points([x[1] for x in class_2], [y[2] for y in class_2], class_names[1])
    plot.draw_points([x[1] for x in class_3], [y[2] for y in class_3], class_names[2])
    plot.set_legend()

    plot.show()


def main() -> None:
    zad3()
    zad4()


if __name__ == '__main__':
    main()
