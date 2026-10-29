"""Load the T800 floor-only variant from its checked-in MJCF and height data."""

from pathlib import Path

import mujoco

TERRAIN_XML = Path(__file__).with_name("terrains") / "t800_smooth_random_2cm.xml"


def replace_floor_from_xml(spec: mujoco.MjSpec) -> None:
    """Change only the floor geometry, retaining its material and contact settings.

    mjlab still creates its original plane terrain and environment origins. This
    scene hook runs after the robot, terrain, friction and materials are assembled.
    Robot assets, physics options, origins and reference heights stay unchanged.
    """
    floor = spec.geom("terrain")
    if floor is None or floor.type != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("expected the original plane geom named 'terrain'")
    definition = mujoco.MjSpec.from_file(str(TERRAIN_XML))
    replacement = definition.geom("terrain")
    source = definition.hfield(replacement.hfieldname)
    field = spec.add_hfield(
        name=source.name,
        file=str((TERRAIN_XML.parent / source.file).resolve()),
        size=source.size,
    )
    floor.type = replacement.type
    floor.hfieldname = field.name
    floor.pos[:] = replacement.pos
    # The original plane contributes no mass or inertia.
    floor.mass = 0.0
