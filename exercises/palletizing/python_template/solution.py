"""Palletizing — reference solution.

This is the code a student writes in the web editor. It drives the UR10 + suction
gripper through the palletizing task: a conveyor feeds boxes one at a time to a
fixed pickup point, and the robot stacks them into an ordered grid on the pallet.

Coordination with the feeder (box_spawner.py) is a two-message handshake:
  - the feeder publishes the box name on /box_ready when a box is stopped at the
    pickup point, and then waits;
  - this code picks and stacks the box, then publishes the name on /box_done,
    which releases the next box.

IMPORTANT — coordinate frames and tuning
========================================
All HAL motion targets are in the robot BASE frame. MoveIt's planning frame is
the robot base_link (SRDF virtual_joint world->base_link, no offset), but the
robot is mounted at Gazebo world (0, 0, 0.9). So a point at Gazebo world height
z is at base-frame height (z - 0.9).

The cup-face reach below tool0 is not a clean analytic value (the tool0->cup
transform has a -90 deg Y rotation plus a compound EE offset), so the pick/place
heights are expressed as tunable constants below and need ONE calibration pass in
sim: jog the TCP to the box top, read off the pose, and adjust CUP_REACH so the
cup just touches the box. Everything marked `# TUNE` is a candidate to adjust.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import HAL_Harmonic as HAL

# --- Geometry ---------------------------------------------------------------
# Robot base mount height in Gazebo world. base_z = world_z - BASE_MOUNT_Z.
BASE_MOUNT_Z = 0.9

# Pickup point on the belt, in Gazebo WORLD coordinates (matches box_spawner:
# spawn_x = belt centre X, center_y = where the belt stops the box).
PICK_WORLD_X = 0.6
PICK_WORLD_Y = -0.15
# Box is 0.40 x 0.30 x 0.20 (tall). Measured natively: top surface sits at
# robot-base z=0.300, i.e. world z=1.20 (belt centre ~1.10, box half-height 0.10).
BOX_HALF_HEIGHT = 0.10
PICK_BOX_TOP_WORLD_Z = 1.20  # measured — box top surface at the pickup point

# Distance from tool0 (what MoveIt positions) down to the cup face. The cup must
# touch the box top, so tool0 target z = box_top_z + CUP_REACH.
# Measured natively: cup flush on box top at base z=0.375, box top base z=0.300.
CUP_REACH = 0.075  # measured — tool0 sits 0.075 m above the cup-face contact

# Clearance to lift to / approach from, above the box top (avoids dragging).
APPROACH_CLEARANCE = 0.30  # TUNE: Increased to avoid dipping into the box during PTP moves

# Cup pointing straight down: tool0 oriented so the suction axis faces -Z world.
# HAL's MoveLinear/MoveJoint take abs_ypr as (roll, pitch, yaw) in DEGREES despite
# the name. The cup points along tool0's +X axis and ls_vgr_joint is -90 deg about
# Y, so tool0 must be rotated 180 deg about Y (cup frame = Ry(180)*Ry(-90)=Ry(90),
# X-axis straight down). MEASURED at the flush grasp pose: tool0 quat (xyzw)
# ~(0,1,0,0) == 180 deg about Y == ypr [0,179,0]. (An earlier [0,90,0] guess put
# the cup horizontal — wrong; the live tf2_echo reading corrected it.)
# Ry(179) avoids qw=0 in the quaternion which can lead to NaN/fail in KDL.
CUP_DOWN_YPR = [0.0, 179.0, 0.0]  # (roll, pitch, yaw) deg — cup faces the floor

# --- Pallet grid (target pattern), in Gazebo WORLD coordinates ---------------
# 2 cols x 2 rows x 2 layers = 8 boxes. Mirrors the grid box_spawner used to
# teleport to, so the stack lands on the pallet table. Boxes fill layer by
# layer, within a layer row by row, column by column.
GRID_COLS = 2
GRID_ROWS = 2
GRID_LAYERS = 2
GRID_ORIGIN_WORLD_X = -0.70   # near-corner column X
GRID_ORIGIN_WORLD_Y = -0.22   # near-corner row Y
GRID_BASE_WORLD_Z = 0.86      # bottom-layer box CENTRE z (table 0.76 + box half 0.10)
PITCH_X = 0.33                # column spacing
PITCH_Y = 0.43                # row spacing
PITCH_Z = 0.20                # layer spacing (== box height)
PLACE_YAW = 90.0              # box long axis along world Y (deg)

# --- Motion tuning ----------------------------------------------------------
SPEED = 0.3        # fraction of max [0,1]
SETTLE = 1.0       # wait after each move (s)
GRIP_PAUSE = 0.5   # wait for the suction attach/detach to take (s)


def world_to_base_z(world_z):
    """Convert a Gazebo world height to the robot base-frame height."""
    return world_z - BASE_MOUNT_Z


def grid_cell(index):
    """Map box index (0-based) to its (x, y, z) box-centre in WORLD coords."""
    per_layer = GRID_COLS * GRID_ROWS
    layer = index // per_layer
    within = index % per_layer
    row = within // GRID_COLS
    col = within % GRID_COLS

    x = GRID_ORIGIN_WORLD_X + col * PITCH_X
    y = GRID_ORIGIN_WORLD_Y + row * PITCH_Y
    z = GRID_BASE_WORLD_Z + layer * PITCH_Z
    return x, y, z, layer, row, col


class FeederLink(Node):
    """Handshake node: receive /box_ready, acknowledge with /box_done."""

    def __init__(self):
        super().__init__("palletizing_solution")
        self._ready_box = None
        self._processed_boxes = set()
        self.create_subscription(String, "/box_ready", self._on_ready, 10)
        self._done_pub = self.create_publisher(String, "/box_done", 10)

    def _on_ready(self, msg):
        if msg.data not in self._processed_boxes:
            self._ready_box = msg.data

    def wait_for_box(self):
        """Block until a box is announced at the pickup point; return its name."""
        self._ready_box = None
        while rclpy.ok() and self._ready_box is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self._processed_boxes.add(self._ready_box)
        return self._ready_box

    def box_done(self, name):
        """Tell the feeder the box has been palletized; release the next."""
        self._done_pub.publish(String(data=name))


def pick(link):
    """Lower onto the box at the pickup point, grip, and lift clear."""
    bx = PICK_WORLD_X
    by = PICK_WORLD_Y
    grip_z = world_to_base_z(PICK_BOX_TOP_WORLD_Z + CUP_REACH)
    approach_z = grip_z + APPROACH_CLEARANCE

    # Swing gracefully from home to a safe high clearance point directly over the box
    HAL.MoveJoint([bx, by, approach_z + 0.3], CUP_DOWN_YPR, SPEED, SETTLE)
    
    # Smooth, single vertical descent onto the box
    HAL.MoveLinear([bx, by, grip_z], CUP_DOWN_YPR, SPEED, SETTLE)
    
    HAL.SuctionSet(True, GRIP_PAUSE)                                   # vacuum on
    
    # Lift the box clear in a single motion
    HAL.MoveLinear([bx, by, approach_z + 0.3], CUP_DOWN_YPR, SPEED, SETTLE)


def place(link, index):
    """Carry the held box to grid cell `index` and release it."""
    wx, wy, wz, layer, row, col = grid_cell(index)
    # tool0 sits CUP_REACH + box height above the target box centre.
    place_z = world_to_base_z(wz + BOX_HALF_HEIGHT + CUP_REACH)
    approach_z = place_z + APPROACH_CLEARANCE
    place_ypr = [CUP_DOWN_YPR[0], CUP_DOWN_YPR[1], PLACE_YAW]

    HAL.MoveJoint([wx, wy, approach_z], place_ypr, SPEED, SETTLE)   # above cell
    HAL.MoveLinear([wx, wy, place_z], place_ypr, SPEED, SETTLE)     # down onto stack
    HAL.SuctionSet(False, GRIP_PAUSE)                              # vacuum off -> release
    HAL.MoveLinear([wx, wy, approach_z], place_ypr, SPEED, SETTLE)  # retreat up
    print(f"placed box {index} at grid L{layer} R{row} C{col}")


def main():
    # rclpy is already initialised by HAL; just add our coordination node.
    link = FeederLink()

    # Natural 'up' home pose: [pan, lift, elbow, w1, w2, w3] in DEGREES
    # This keeps the arm safely clear of the workspace without awkward IK contortions.
    home_joints = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]
    HAL.MoveAbsJ(home_joints, SPEED, SETTLE)

    total = GRID_COLS * GRID_ROWS * GRID_LAYERS
    for index in range(total):
        name = link.wait_for_box()
        if name is None:
            break
        print(f"box {name} ready -> placing as #{index}")
        
        pick(link)
        place(link, index)
        link.box_done(name)
        
        # Return to home pose before waiting for the next box to ensure a clean sweep
        HAL.MoveAbsJ(home_joints, SPEED, SETTLE)

    print("palletizing complete")
    link.destroy_node()


if __name__ == "__main__":
    main()
