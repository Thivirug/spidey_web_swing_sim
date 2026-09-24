from ursina import Entity, color, Vec3

class Web:
    def __init__(
        self,
        thickness=0.10,
        z=-3.0,
        hand_offset=Vec3(0.75, 0.35, 0.0),
    ):
        self.z = z
        self.thickness = thickness
        self.hand_offset = hand_offset

        self.entity = Entity(
            model="cube",
            color=color.rgb32(35, 35, 35),
            enabled=False,
        )

    def update(self, state, hand_position):
        if not state["attached"] or state["anchor"] is None:
            self.entity.enabled = False
            return

        self.entity.enabled = True

        # Start web at Spider-Man's actual hand marker
        spider_pos = Vec3(
            hand_position.x,
            hand_position.y,
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

        self.entity.look_at(anchor_pos)

        self.entity.scale = Vec3(
            self.thickness,
            self.thickness,
            distance,
        )