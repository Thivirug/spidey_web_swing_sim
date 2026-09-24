from ursina import Entity, color


class City:
    """
    Visual city generated from the simulation's anchor positions.

    Physics remains 2D:
        x = horizontal
        y = vertical

    z is only used for visual depth.
    """

    def __init__(self, anchors, y_ground=0.0):
        self.anchors = anchors
        self.y_ground = y_ground

        self.buildings = []
        self.windows = []
        self.anchor_markers = []

        self.create_ground()
        self.create_buildings()
        self.create_anchor_markers()

    def create_ground(self):
        xs = [anchor[0] for anchor in self.anchors]

        left = min(xs) - 30
        right = max(xs) + 30
        length = right - left
        centre_x = (left + right) / 2

        # Main road
        self.road = Entity(
            model="cube",
            color=color.rgb32(55, 58, 63),
            position=(
                centre_x,
                self.y_ground - 0.30,
                -5,
            ),
            scale=(
                length,
                0.4,
                12,
            ),
        )

        # Sidewalk closer to buildings
        self.sidewalk = Entity(
            model="cube",
            color=color.rgb32(145, 145, 145),
            position=(
                centre_x,
                self.y_ground - 0.12,
                2,
            ),
            scale=(
                length,
                0.25,
                3,
            ),
        )

        # Background ground behind buildings
        self.background_ground = Entity(
            model="cube",
            color=color.rgb32(85, 90, 95),
            position=(
                centre_x,
                self.y_ground - 0.35,
                12,
            ),
            scale=(
                length,
                0.5,
                18,
            ),
        )

    def create_buildings(self):

        building_colors = [
            color.rgb32(75, 82, 95),
            color.rgb32(90, 95, 105),
            color.rgb32(110, 105, 100),
            color.rgb32(65, 72, 82),
            color.rgb32(100, 108, 115),
        ]

        for i, (x, y) in enumerate(self.anchors):

            # Slight variation in building shape
            width = 5.0 + (i % 3) * 0.8
            height = max(5.0, y - self.y_ground)
            depth = 8.0 + (i % 2) * 2.0

            # Buildings remain behind Spider-Man
            z = 6.0

            building = Entity(
                model="cube",
                color=building_colors[i % len(building_colors)],
                position=(
                    x,
                    self.y_ground + height / 2,
                    z,
                ),
                scale=(
                    width,
                    height,
                    depth,
                ),
            )

            self.buildings.append(building)

            self.create_windows(
                x=x,
                height=height,
                width=width,
                depth=depth,
                z=z,
                building_index=i,
            )

    def create_windows(
        self,
        x,
        height,
        width,
        depth,
        z,
        building_index,
    ):
        """
        Add simple windows to the front face of each building.
        """

        # Front face is toward the camera (negative z)
        front_z = z - depth / 2 - 0.03

        window_color = color.rgb32(185, 205, 215)

        floor_spacing = 3.0

        y = self.y_ground + 2.0

        while y < self.y_ground + height - 1.5:

            # Two window columns
            for x_offset in (-width * 0.22, width * 0.22):

                # Make some windows dark for variation
                if (int(y) + building_index) % 4 == 0:
                    current_color = color.rgb32(
                        55,
                        65,
                        75,
                    )
                else:
                    current_color = window_color

                window = Entity(
                    model="cube",
                    color=current_color,
                    position=(
                        x + x_offset,
                        y,
                        front_z,
                    ),
                    scale=(
                        0.9,
                        1.2,
                        0.08,
                    ),
                )

                self.windows.append(window)

            y += floor_spacing

    def create_anchor_markers(self):
        """
        Small roof attachment markers.
        """

        for x, y in self.anchors:

            marker = Entity(
                model="sphere",
                color=color.rgb32(210, 210, 210),
                position=(
                    x,
                    y,
                    -0.2,
                ),
                scale=0.22,
            )

            self.anchor_markers.append(marker)