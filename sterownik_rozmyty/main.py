import matplotlib.pyplot as plt


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
                 ramp: tuple[int, int, int, int] = (0, 30, -10, -30), ramp_target_angle: tuple[int, int] =  (-20, 20),
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

    def __init__(self, simulation_parameters: SimulationParameters, vehicle: VehicleParameters) -> None:
        self.simulation_parameters = simulation_parameters
        self.vehicle = vehicle


def zad1() -> None:
    sim_params = SimulationParameters()
    vehicle = VehicleParameters(100, 0, 90)
    fuzzy_controller = FuzzyController(sim_params, vehicle)


def main() -> None:
    zad1()  # sterownik rozmyty


if __name__ == '__main__':
    main()

