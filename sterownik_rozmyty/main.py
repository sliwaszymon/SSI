import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import math
import numpy as np
from matplotlib.transforms import Affine2D


class SimulationParameters:
    x_min: int
    x_max: int
    x_range: int
    y_min: int
    y_max: int
    y_range: int
    lramp: int
    rramp: int
    tramp: int
    bramp: int
    ramp_min_target_angle: int
    ramp_max_target_angle: int
    rotation_min: int
    rotation_max: int
    angle_min: int
    angle_max: int
    angle_range: int
    move_step: float

    def __init__(self, x: tuple[int, int] = (-100, 100), y: tuple[int, int] = (-100, 0),
                 ramp: tuple[int, int, int, int] = (0, 30, -10, -30), ramp_target_angle: tuple[int, int] = (-20, 20),
                 rotation: tuple[int, int] = (-20, 20), angle: tuple[int, int] = (-180, 180), angle_range: int = 360,
                 move_step: float = 5.0) -> None:
        self.x_min = min(x)
        self.x_max = max(x)
        self.x_range = sum([abs(r) for r in x])
        self.y_min = min(y)
        self.y_max = max(y)
        self.y_range = sum([abs(r) for r in y])
        self.tramp = ramp[0]
        self.rramp = ramp[1]
        self.bramp = ramp[2]
        self.lramp = ramp[3]
        self.ramp_min_target_angle = min(ramp_target_angle)
        self.ramp_max_target_angle = max(ramp_target_angle)
        self.rotation_min = min(rotation)
        self.rotation_max = max(rotation)
        self.angle_min = min(angle)
        self.angle_max = max(angle)
        self.angle_range = angle_range
        self.move_step = move_step


class VehicleParameters:
    x: int
    y: int
    rotation: int

    def __init__(self, x: int, y: int, rotation: int) -> None:
        self.x = x
        self.y = y
        self.rotation = rotation


class FuzzyController:
    simulation_parameters: SimulationParameters
    vehicle: VehicleParameters
    route: list[list[int, int]]

    def __init__(self, simulation_parameters: SimulationParameters, vehicle: VehicleParameters) -> None:
        self.simulation_parameters = simulation_parameters
        self.vehicle = vehicle
        self.route = [[vehicle.x, vehicle.y]]

    def calculate_angle_to_destination(self) -> float:
        destination_x = (self.simulation_parameters.lramp + self.simulation_parameters.rramp) / 2
        destination_y = self.simulation_parameters.tramp
        dx, dy = destination_x - self.vehicle.x, destination_y - self.vehicle.y
        angle_radians = math.atan2(dx, dy)

        return math.degrees(angle_radians)

    def takagi_sugeno_driver(self, angle: float) -> float:
        if angle < self.vehicle.rotation:
            if (self.vehicle.rotation - angle) >= self.simulation_parameters.rotation_max:
                return self.simulation_parameters.rotation_min
            if self.simulation_parameters.rotation_min < (self.vehicle.rotation - angle) \
                    < self.simulation_parameters.rotation_max:
                return - (self.vehicle.rotation - angle)
        if angle > self.vehicle.rotation:
            if (angle - self.vehicle.rotation) >= self.simulation_parameters.rotation_max:
                return self.simulation_parameters.rotation_max
            if self.simulation_parameters.rotation_min < (angle - self.vehicle.rotation) \
                    < self.simulation_parameters.rotation_max:
                return angle - self.vehicle.rotation
        return 0

    def visualize(self) -> None:
        fig, ax = plt.subplots()
        ramp = np.array([[self.simulation_parameters.lramp, self.simulation_parameters.tramp],
                         [self.simulation_parameters.rramp, self.simulation_parameters.tramp],
                         [self.simulation_parameters.rramp, self.simulation_parameters.bramp],
                         [self.simulation_parameters.lramp, self.simulation_parameters.bramp],
                         [self.simulation_parameters.lramp, self.simulation_parameters.tramp]])

        destination = np.array([(self.simulation_parameters.lramp + self.simulation_parameters.rramp) / 2,
                                self.simulation_parameters.tramp])

        route = np.array(self.route)

        vehicle = np.array([self.vehicle.x, self.vehicle.y])
        m = MarkerStyle(marker='^', transform=Affine2D().rotate_deg(-self.vehicle.rotation))

        ax.plot(ramp.T[0], ramp.T[1], color='red', linestyle='dotted')
        ax.plot(route.T[0], route.T[1], color='green', linestyle='dotted')
        ax.scatter(destination.T[0], destination.T[1], color='blue', marker='s')
        ax.scatter(vehicle.T[0], vehicle.T[1], color='black', marker=m, s=100)

        ax.set_xlim(self.simulation_parameters.x_min, self.simulation_parameters.x_max)
        ax.set_ylim(self.simulation_parameters.y_min, self.simulation_parameters.y_max)
        plt.show()

    @staticmethod
    def euclidean_distance(p1: list[float | int], p2: list[float | int]) -> float:
        if len(p1) != len(p2):
            raise ValueError("Points must have exact same number of dimensions!")
        return sum([(x1 - x2) ** 2 for x1, x2 in zip(p1, p2)]) ** .5

    def __call__(self, error: int = 0, visualize: bool = False) -> None:
        destination_x = (self.simulation_parameters.lramp + self.simulation_parameters.rramp) / 2
        destination_y = self.simulation_parameters.tramp

        while not (destination_x - error <= self.vehicle.x <= destination_x + error
                   and destination_y - error <= self.vehicle.y <= destination_y + error):
            angle = self.calculate_angle_to_destination()
            rotation = self.takagi_sugeno_driver(angle)
            self.vehicle.rotation += int(rotation)

            distance_to_destination = self.euclidean_distance([self.vehicle.x, self.vehicle.y],
                                                              [destination_x, destination_y])

            if distance_to_destination > self.simulation_parameters.move_step:
                move_distance = self.simulation_parameters.move_step
            else:
                move_distance = distance_to_destination

            radians = math.radians(
                90 - self.vehicle.rotation if self.vehicle.rotation <= 90 else self.vehicle.rotation - 90)
            self.vehicle.x += int(move_distance * math.cos(radians))
            self.vehicle.y += int(move_distance * math.sin(radians))
            self.route.append([self.vehicle.x, self.vehicle.y])

            print(f'Vehicle: (x: {self.vehicle.x}, y: {self.vehicle.y})')
            if visualize:
                self.visualize()
        print('Vehicle reached destination!')


def zad1() -> None:
    sim_params = SimulationParameters()
    vehicle = VehicleParameters(10, -100, 170)
    fuzzy_controller = FuzzyController(sim_params, vehicle)
    fuzzy_controller(error=1, visualize=True)


def main() -> None:
    zad1()  # sterownik rozmyty


if __name__ == '__main__':
    main()
