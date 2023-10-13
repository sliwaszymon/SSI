from math import sin
import random
import matplotlib.pyplot as plt


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


def mi_plus_lambda(mu: int, lmbd: int,  iterations: int, variability: list[int | float, int | float] = (0, 100)):
    parents = [[random.uniform(variability[0], variability[1]), random.uniform(variability[0], variability[1])]
               for _ in range(mu)]
    print(parents)
    for _ in range(iterations):
        values = [foo2(parent[0], parent[1]) for parent in parents]
        print(values)
        childs = []


def zad1():
    one_plus_one(10, 1.1, 100)


def zad2():
    mi_plus_lambda(4, 20)


def main():
    zad1() # 1+1
    # zad2() # mu+lambda


if __name__ == '__main__':
    main()
