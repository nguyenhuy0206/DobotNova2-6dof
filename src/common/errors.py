"""Exception hierarchy.

Everything raised for a robot/controller reason derives from ``RobotError`` so
callers (UI, agent harness, scripts) can catch a single type.
"""


class RobotError(RuntimeError):
    """Controller rejected a command, timed out, or returned bad data."""


class ValidationError(RobotError):
    """A request was outside the configured limits (speed, step, workspace…)."""


class SafetyError(RobotError):
    """The robot is not in a state where motion is allowed."""


class MotionStopped(SafetyError):
    """A move was stopped by the safety layer (not arrived in time, or an alarm during it).
    The harness ends the episode on it: the model gets no further turn."""


class ContactError(ValidationError):
    """A slow approach move was blocked by something (the robot stopped and backed off)."""
