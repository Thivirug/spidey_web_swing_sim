from ursina import Entity, color

class City:
    """
        Visual city generated from the simulation's anchor positions.

        Physics coordinates:
            x = horizontal position
            y = vertical position

        The z-axis is used only to give the scene depth.
    """

    def __init__(self, anchors, y_ground=0.0):
        self.anchors = anchors
        self.y_ground = y_ground

        self.buildings = []
        self.anchor_markers = []

        self.create_ground()
        self.create_buildings()
        self.create_anchor_markers()

    def create_ground(self):
        # Cover the region occupied by the anchors.
        xs = [anchor[0] for anchor in self.anchors]

        left = min(xs) - 30
        right = max(xs) + 30

        self.ground = Entity(
            model="cube",
            color=color.rgb(65, 70, 75),
            position=(
                (left + right) / 2,
                self.y_ground - 0.25,
                0,
            ),
            scale=(
                right - left,
                0.5,
                35,
            ),
        )

    def create_buildings(self):
        """
            Create one simple building behind each anchor.
        """

        for i, (x, y) in enumerate(self.anchors):

            width = 5.0
            height = max(5.0, y - self.y_ground)
            depth = 8.0

            # Keep buildings behind Spider-Man's motion plane.
            z = 5.0

            shade = 105 + (i % 4) * 18

            building = Entity(
                model="cube",
                color=color.rgb(shade, shade, shade + 8),
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

    def create_anchor_markers(self):
        """
            Display the exact anchor coordinates used by physics.
        """

        for x, y in self.anchors:

            marker = Entity(
                model="sphere",
                color=color.yellow,
                position=(x, y, 0),
                scale=0.45,
            )

            self.anchor_markers.append(marker)