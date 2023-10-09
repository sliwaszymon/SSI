import matplotlib.pyplot as plt


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
    def __init__(self, title: str = None, xlabel: str = None, ylabel: str = None) -> None:
        self.fig, self.ax = plt.subplots()
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.series = 0
        if title:
            self.ax.set_title(title)
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)

    def draw(self, x: list[int], y: list[int]) -> None:
        settings = line_settings[self.series % 10]
        self.ax.plot(x, y, color=settings['line_color'], linestyle=settings['line_style'])
        self.series += 1

    def clean(self) -> None:
        self.__init__(self.title, self.xlabel, self.ylabel)

    @staticmethod
    def show() -> None:
        plt.show()


def main() -> None:
    plot = Plot('Nowy plot', 'x', 'y')
    plot.draw([1, 2, 3, 4, 5], [10, 12, 5, 8, 6])
    plot.show()
    plot.clean()
    plot.draw([10, 12, 5, 8, 6], [1, 2, 3, 4, 5])
    plot.show()


if __name__ == '__main__':
    main()
