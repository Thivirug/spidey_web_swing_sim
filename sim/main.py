import numpy as np
from ursina import Ursina, Entity, color, camera, time, Vec3, Sky

from src.controller import PolicyParams
from sim.sim_live import LiveSimulator
from sim.assets.city import City
from sim.assets.web import Web

# 1. Simulation configuratio

PHYSICS_DT = 0.01

# policy = PolicyParams(  # no drag
#     theta_r=np.radians(50.911),
#     L=5.809,
# )

policy = PolicyParams( # with drag
    theta_r=np.radians(47.161),
    L=5.963,
)

anchors = [
    # Initial building / starting anchor
    (0.0, 30.0), # if this is changed change the below anchor arg asw

    (4.0, 13.0),
    (8.0, 25.0),
    (13.0, 20.0),
    (19.0, 25.0),
    (24.0, 23.0),
    (31.0, 28.0),
    (35.0, 25.0),
    (39.0, 22.0),
    (44.0, 27.0),
    (50.0, 19.0),
    (54.0, 23.0),
    (57.0, 22.0),
    (64.0, 31.0),
    (70.0, 23.0),
    (77.0, 29.0),
    (81.0, 12.0),
    (85.0, 20.0),
    (91.0, 26.0),
    (98.0, 32.0),
    (105.0, 24.0),
    (109.0, 27.0),
    (113.0, 29.0),
    (120.0, 21.0),
    (128.0, 27.0),
    (135.0, 30.0),
    (139.0, 28.0),
    (143.0, 25.0),
    (150.0, 30.0),
    (158.0, 22.0),
    (162.0, 25.0),
    (166.0, 28.0),
]

simulation = LiveSimulator(
    theta0=np.radians(-70),
    omega0=0.0,
    L=10.0,
    anchor=(0.0, 30.0), # if this changes anchor initial in list above needs to be changed asw
    policy_params=policy,
    T=30.0,
    dt=PHYSICS_DT,
    y_ground=0.0,
    # drag_params=None,
    drag_params=(75.0, 1.225, 1.0, 0.7), 
    anchors=anchors,
    L_min=5.0,
    L_max=15.0,
    min_flight_time=1.2,
)

# Initialise Ursina

app = Ursina()

sky = Sky(
    color=color.rgb32(
        120,
        165,
        210,
    )
)

CAMERA_Y = 24
CAMERA_Z = -55
CAMERA_X_OFFSET = -8
CAMERA_LOOK_AHEAD = 8
CAMERA_FOLLOW_SPEED = 3.0

initial_state = simulation.get_state()

spiderman = Entity(
    model="assets/models/spidey/spiderman.obj",
    scale=1.8,
    position=(
        initial_state["x"],
        initial_state["y"],
        -3,
    ),
)

hand_anchor = Entity(
    parent=spiderman,
    position=(0.77, 1.14, -0.1),
    visible=False,
)

web = Web()

# Create the city from the SAME anchors passed to LiveSimulator.
city = City(
    anchors=anchors,
    y_ground=0.0,
)

# Perspective camera positioned at an angle to the XY plane.
camera.orthographic = False
camera.fov = 65
camera.position = (
    initial_state["x"] + CAMERA_X_OFFSET,
    CAMERA_Y,
    CAMERA_Z,
)

camera.look_at(
    Vec3(
        initial_state["x"] + CAMERA_LOOK_AHEAD,
        15,
        0,
    )
)

# Live update loop
accumulator = 0.0
    
def update():
    global accumulator

    if not simulation.finished:

        accumulator += time.dt

        while (
            accumulator >= PHYSICS_DT
            and not simulation.finished
        ):
            simulation.step()
            accumulator -= PHYSICS_DT

    state = simulation.get_state()

    # Spider-Man
    spiderman.position = Vec3(
        state["x"],
        state["y"],
        -2,
    )

    # Web
    web.update(state, hand_anchor.world_position,)

    # Camera follow
    target_x = state["x"] + CAMERA_X_OFFSET

    camera.x += (
        target_x - camera.x
    ) * min(
        CAMERA_FOLLOW_SPEED * time.dt,
        1.0,
    )

    # Keep camera altitude/depth stable
    camera.y = CAMERA_Y
    camera.z = CAMERA_Z

    # Look slightly ahead of Spider-Man
    camera.look_at(
        Vec3(
            state["x"] + CAMERA_LOOK_AHEAD,
            15,
            0,
        )
    )

    print(
        state["time"],
        state["attached"],
        state["anchor"],
    )

# Start the application
app.run()