import random
import matplotlib.pyplot as plt
import numpy as np


class OnePlusOne:
    dispersion: int
    growth_factor: float
    iterations: int
    variability: tuple[int | float, int | float]
    x: float
    y: float

    def __init__(self, dispersion: int, growth_factor: float, iterations: int,
                 variability: tuple[int | float, int | float] = (0, 100)) -> None:
        self.dispersion = dispersion
        self.growth_factor = growth_factor
        self.iterations = iterations
        self.variability = variability

    def initialize_values(self) -> None:
        self.x = random.uniform(self.variability[0], self.variability[1])
        self.y = self.function(self.x)

    def __call__(self, visualize=False) -> None:
        self.initialize_values()

        x_pot, y_pot = None, None
        fig, ax = plt.subplots()
        for i in range(self.iterations):
            x_pot = self.x + random.uniform(-self.dispersion, self.dispersion)
            if x_pot < self.variability[0]:
                x_pot = 0.0
            if x_pot > self.variability[1]:
                x_pot = 100.0
            y_pot = self.function(x_pot)
            ax.scatter(x_pot, y_pot, s=100, color='b', marker='o')
            print(f'Iteracja: {i}, ({x_pot}, {y_pot})')

            if y_pot >= self.y:
                self.x, self.y = x_pot, y_pot
                self.dispersion *= self.growth_factor
            else:
                self.dispersion /= self.growth_factor
        ax.scatter(x_pot, y_pot, s=250, color='r', marker='x')
        ax.set_title('1+1 algorithm')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if visualize:
            plt.show()

    def get_max_spot(self) -> tuple[float, float]:
        return self.x, self.y

    @staticmethod
    def function(x: float | int) -> float:
        return np.sin(x / 10) * np.sin(x / 200)


class MuPlusLambda:
    mu: int
    lmbd: int
    iterations: int
    competition_size: int
    mutation: float
    variability: tuple[int | float, int | float]
    parents: list[list[float]]
    kids: list[list[float]]

    def __init__(self, mu: int, lmbd: int, iterations: int, competition_size: int, mutation: float,
                 variability: tuple[int | float, int | float] = (0, 100)) -> None:
        self.mu = mu
        self.lmbd = lmbd
        self.iterations = iterations
        self.competition_size = competition_size
        self.mutation = mutation
        self.variability = variability

    def initialize_parents(self) -> None:
        self.parents = [[random.uniform(self.variability[0], self.variability[1]),
                         random.uniform(self.variability[0], self.variability[1])] for _ in range(self.mu)]

    def visualize(self, iteration: int | str) -> None:
        if iteration == 0:
            iteration = 'INITIAL'
        x_1, x_2 = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
        z = np.sin(x_1 * 0.05) + np.sin(x_2 * 0.05) + 0.4 * np.sin(x_1 * 0.15) * np.sin(x_2 * 0.15)
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(x_1, x_2, z, cmap='viridis')
        plt.colorbar(contour)
        plt.scatter([x[0] for x in self.parents], [x[1] for x in self.parents], marker='o', color='b', s=50,
                    label='Parents')
        plt.scatter([x[0] for x in self.kids], [x[1] for x in self.kids], marker='x', color='r', s=50, label='Kids')
        plt.title(f'mu+lambda iteration {iteration}')
        plt.legend()
        plt.show()

    def __call__(self, visualize: bool = True, what_which: int = 1) -> None:
        self.initialize_parents()
        for i in range(self.iterations):
            kids = []
            for _ in range(self.lmbd):
                kid = self.competition(random.sample(self.parents, self.competition_size))
                kids.append([x + random.uniform(-self.mutation, self.mutation) for x in kid])

            self.kids = kids
            if visualize and i % what_which == 0:
                self.visualize(i)
            elif visualize and i % what_which != 0 and i == self.iterations - 1:
                self.visualize('LAST')

            new_parents = self.parents + self.kids
            new_vals = [self.function(parent[0], parent[1]) for parent in new_parents]
            while len(new_parents) != self.mu:
                pop_idx = new_vals.index(min(new_vals))
                new_vals.pop(pop_idx)
                new_parents.pop(pop_idx)
            self.parents = new_parents

    def get_best_parent(self) -> list[float, float]:
        vals = [self.function(parent[0], parent[1]) for parent in self.parents]
        ans_idx = vals.index(max(vals))
        return self.parents[ans_idx]

    def competition(self, vals: list[list[float, float]]) -> list[float, float]:
        ys = [self.function(x[0], x[1]) for x in vals]
        return vals[ys.index(max(ys))]

    @staticmethod
    def function(x1: float | int, x2: float | int) -> float:
        return np.sin(x1 * 0.05) + np.sin(x2 * 0.05) + 0.4 * np.sin(x1 * 0.15) * np.sin(x2 * 0.15)


def function(x1: float | int, x2: float | int) -> float:
    return np.sin(x1 * 0.05) + np.sin(x2 * 0.05) + 0.4 * np.sin(x1 * 0.15) * np.sin(x2 * 0.15)


def euklides(x_1: list[int | float, int | float], x_2: list[int | float, int | float]) -> float:
    return ((x_2[0] - x_1[0]) ** 2 + (x_2[1] - x_1[1]) ** 2) ** (1/2)


def swietliki(n: int, beta_zero: float, gamma_zero: float, mu_zero: float,
              x_min_i: int | float = 0, x_max_i: int | float = 100, iterations: int = 100) -> tuple:

    gamma = gamma_zero / (x_max_i - x_min_i)
    mu_i = (x_max_i - x_min_i) * mu_zero
    xes = np.random.uniform(x_min_i, x_max_i, (n, 2))
    fs = np.array([function(x[0], x[1]) for x in xes])

    best_point = xes[np.argmax(fs)]
    best_value = np.max(fs)

    for _ in range(iterations):
        for a in random.sample(range(0, n), n):
            for b in random.sample(range(0, n), n):
                if fs[b] > fs[a]:
                    beta = beta_zero * np.exp(-gamma * euklides(xes[a], xes[b]) ** 2)
                    xes[a] += beta * (xes[b] - xes[a])
            xes[a] += np.random.uniform(-mu_i, mu_i, 2)
            fs[a] = function(xes[a, 0], xes[a, 1])
            if fs[a] > best_value:
                best_point = xes[a]
                best_value = fs[a]

    return best_point, best_value


def zad1() -> None:
    opo = OnePlusOne(10, 1.1, 100)
    opo(visualize=True)
    print("Ostateczne najlepszy punkt (x, y):\n", opo.get_max_spot())


def zad2() -> None:
    mpl = MuPlusLambda(4, 10, 20, 2, 10)
    mpl(visualize=True, what_which=5)
    print("Ostateczne najlepszy punkt (x1, x2):\n", mpl.get_best_parent())


def zad3() -> None:
    ans = swietliki(4, 0.3, 0.1, 0.05)
    print(ans)


def main() -> None:
    # zad1()  # 1+1
    # zad2()  # mu+lambda
    zad3()  # świetliki


if __name__ == '__main__':
    main()