from math import sin
import random
import matplotlib.pyplot as plt
import numpy as np


def foo(x: float) -> float:
    return sin(x/10) * sin(x/200)


def foo2(x1: float, x2: float) -> float:
    return sin(x1 * 0.05) + sin(x2 * 0.05) + 0.4 * sin(x1 * 0.15) * sin(x2 * 0.15)


def one_plus_one(dispersion, growth_factor, iterations, variability: list[int | float, int | float] = (0, 100)):
    x = random.uniform(variability[0], variability[1])
    y = foo(x)
    x_pot, y_pot = None, None
    fig, ax = plt.subplots()
    for i in range(iterations):
        x_pot = x + random.uniform(-dispersion, dispersion)
        if x_pot < variability[0]:
            x_pot = 0.0
        if x_pot > variability[1]:
            x_pot = 100.0
        y_pot = foo(x_pot)
        ax.scatter(x_pot, y_pot, s=100, color='b', marker='o')
        print(f'Iteracja: {i}, ({x_pot}, {y_pot})')

        if y_pot >= y:
            x, y = x_pot, y_pot
            dispersion *= growth_factor
        else:
            dispersion /= growth_factor
    ax.scatter(x_pot, y_pot, s=250, color='r', marker='x')
    ax.set_title('1+1 algorithm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def competition(vals: list[list[float, float]]) -> list[float, float]:
    ys = [foo2(x[0], x[1]) for x in vals]
    return vals[ys.index(max(ys))]


def mi_plus_lambda(mu: int, lmbd: int,  iterations: int, competition_size: int, mutation: float,
                   variability: list[int | float, int | float] = (0, 100)):
    parents = [[random.uniform(variability[0], variability[1]), random.uniform(variability[0], variability[1])]
               for _ in range(mu)]
    for _ in range(iterations):
        kids = []
        for _ in range(lmbd):
            kid = competition(random.sample(parents, competition_size))
            kids.append([x+random.uniform(-mutation, mutation) for x in kid])

        x = np.linspace(0, 100, 100)
        x_1, x_2 = np.meshgrid(x, x)
        z = np.sin(x_1 * 0.05) + np.sin(x_2 * 0.05) + 0.4 * np.sin(x_1 * 0.15) * np.sin(x_2 * 0.15)
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(x_1, x_2, z, cmap='viridis')
        plt.colorbar(contour)
        plt.scatter([x[0] for x in parents], [x[1] for x in parents], marker='o', color='b', s=50, label='Parents')
        plt.scatter([x[0] for x in kids], [x[1] for x in kids], marker='x', color='r', s=50, label='Parents')

        plt.show()

        new_parents = parents + kids
        new_vals = [foo2(parent[0], parent[1]) for parent in new_parents]
        while len(new_parents) != mu:
            pop_idx = new_vals.index(min(new_vals))
            new_vals.pop(pop_idx)
            new_parents.pop(pop_idx)
        parents = new_parents

    vals = [foo2(parent[0], parent[1]) for parent in parents]
    ans_idx = vals.index(max(vals))
    return parents[ans_idx]


def zad1():
    one_plus_one(10, 1.1, 100)


def zad2():
    ans = mi_plus_lambda(4, 10, 20, 2, 10)
    print(ans)


def main():
    # zad1() # 1+1
    zad2() # mu+lambda


if __name__ == '__main__':
    main()
