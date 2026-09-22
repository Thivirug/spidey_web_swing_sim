import numpy as np
from ursina import Ursina, Entity, color, camera, time, Vec3

from src.controller import PolicyParams
from sim.sim_live import LiveSimulator
from sim.assets.city import City

# 1. Simulation configuratio

PHYSICS_DT = 0.01

policy = PolicyParams(
    theta_r=np.radians(44.082),
    L=12.064,
)

anchors = [
    (8.0, 25.0),
    (13.0, 20.0),
    (19.0, 28.0),
    (24.0, 23.0),
    (31.0, 30.0),
    (39.0, 22.0),
    (44.0, 27.0),
    (50.0, 19.0),
    (57.0, 26.0),
    (64.0, 31.0),
    (70.0, 23.0),
    (77.0, 28.0),
    (85.0, 20.0),
    (91.0, 26.0),
    (98.0, 32.0),
    (105.0, 24.0),
    (113.0, 29.0),
    (120.0, 21.0),
    (128.0, 27.0),
    (135.0, 33.0),
    (143.0, 25.0),
    (150.0, 30.0),
    (158.0, 22.0),
    (166.0, 28.0),
]

simulation = LiveSimulator(
    theta0=np.radians(-45),
    omega0=1.2,
    L=10.0,
    anchor=(0.0, 22.0),
    policy_params=policy,
    T=50.0,
    dt=PHYSICS_DT,
    y_ground=0.0,
    drag_params=None,
    anchors=anchors,
    L_min=5.0,
    L_max=15.0,
    min_flight_time=0.25,
)

# 2. Initialise Ursina

app = Ursina()

initial_state = simulation.get_state()

spiderman = Entity(
    model="sphere",
    color=color.red,
    scale=1.0,
    position=(
        initial_state["x"],
        initial_state["y"],
        0,
    ),
)

# Create the city from the SAME anchors passed to LiveSimulator.
city = City(
    anchors=anchors,
    y_ground=0.0,
)

# Perspective camera positioned at an angle to the XY plane.
camera.orthographic = False
camera.fov = 65
camera.position = (10, 24, -55)
camera.look_at((20, 12, 0))

# 3. Live update loo

accumulator = 0.0


def update():
    global accumulator

    if simulation.finished:
        return

    # Accumulate elapsed real-world time.
    accumulator += time.dt

    # Execute the required number of fixed RK4 steps.
    while accumulator >= PHYSICS_DT and not simulation.finished:
        simulation.step()
        accumulator -= PHYSICS_DT

    # Render the most recently computed state.
    state = simulation.get_state()

    spiderman.position = Vec3(
        state["x"],
        state["y"],
        -2,
    )

# 4. Start the applicatio

app.run()