"""Calibration of the cell: everything that turns measurements into the numbers the robot and
the cameras rely on. Operator infrastructure (scripts/calibrate.sh), not an Astra tool.

    board      ChArUco board: detection, board pose, camera intrinsics
    hand_eye   where a camera is: on the flange (board views) or fixed (through the board, or a
               marker on the gripper with an unknown offset)
    cameras    the procedure for both cameras: capture, aim and orbit, solve, gates, save
    survey     the cell by hand-guided touches: gripper contact model, table, points, keep-out zones
    table      teach the table height by lowering the fingertips
    workcell   the stored table / cube-top heights the motion guards use
    geometry   rigid-transform helpers shared by all of them
    files      where every result is written
"""
