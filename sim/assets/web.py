from ursina import Entity, color, Vec3

class Web:
    def __init__(self, thickness=0.13, z=-2.2):
        self.z = z

        self.entity = Entity(
            model="cube",
            color=color.rgb(35, 35, 35), # dark
            scale=(thickness, thickness, 1.0),
            enabled=False,
        )

    def update(self, state):
        if not state["attached"] or state["anchor"] is None:
            self.entity.enabled = False
            return

        self.entity.enabled = True

        spider_pos = Vec3(
            state["x"],
            state["y"],
            self.z,
        )

        anchor_x, anchor_y = state["anchor"]

        anchor_pos = Vec3(
            anchor_x,
            anchor_y,
            self.z,
        )

        direction = anchor_pos - spider_pos
        distance = direction.length()

        self.entity.position = (
            spider_pos + anchor_pos
        ) / 2

        self.entity.scale = (
            self.entity.scale_x,
            self.entity.scale_y,
            distance,
        )

        self.entity.look_at(anchor_pos)

    def hide(self):
        self.entity.enabled = False