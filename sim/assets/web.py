from ursina import Entity, color, Vec3


class Web:
    def __init__(self, thickness=0.10, z=-3.0):
        self.z = z
        self.thickness = thickness

        self.entity = Entity(
            model="cube",
            color=color.rgb32(35, 35, 35),
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

        # Put the web exactly halfway between the endpoints
        self.entity.position = (
            spider_pos + anchor_pos
        ) / 2

        # Point the entity's Z axis toward the anchor
        self.entity.look_at(anchor_pos)

        # THEN apply the length
        self.entity.scale = Vec3(
            self.thickness,
            self.thickness,
            distance,
        )