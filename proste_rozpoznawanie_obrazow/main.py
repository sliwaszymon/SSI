import numpy as np


def iterate(table: list[list[int]]) -> list[tuple]:
    return [(row_idx, col_idx) for row_idx, row in enumerate(table) for col_idx, _ in enumerate(row)]


def manhattan(p1: list[float | int], p2: list[float | int]) -> float:
    if len(p1) != len(p2):
        raise ValueError("Points must have exact same number of dimensions!")
    return sum([abs(x1 - x2) for x1, x2 in zip(p1, p2)])


def euclidean_distance(p1: list[float | int], p2: list[float | int]) -> float:
    if len(p1) != len(p2):
        raise ValueError("Points must have exact same number of dimensions!")
    return sum([(x1 - x2) ** 2 for x1, x2 in zip(p1, p2)]) ** .5


def miara_niepodobienstwa(bit_a: list[list[int]], bit_b: list[list[int]]) -> float:
    miara = 0
    for pay, pax in iterate(bit_a):
        if bit_a[pay][pax] == 1:
            odl_min = np.inf
            for pby, pbx in iterate(bit_b):
                if bit_b[pby][pbx] == 1:
                    odl_akt = euclidean_distance([pax, pay], [pbx, pby])
                    odl_min = min(odl_min, odl_akt)
            miara += odl_min
    return miara


def miara_podobienstwa_obustronnego(bit_a: list[list[int]], bit_b: list[list[int]]) -> float:
    return -(miara_niepodobienstwa(bit_a, bit_b) + miara_niepodobienstwa(bit_b, bit_a))


def zad1() -> None:
    tab1 = [[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
    tab2 = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
    print(miara_podobienstwa_obustronnego(tab1, tab2))


def main() -> None:
    zad1()


if __name__ == '__main__':
    main()
