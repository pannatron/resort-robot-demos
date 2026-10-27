Added
^^^^^

* Added the ``IsaacContrib-Velocity-{Flat,Rough}-DobotQuad`` and ``IsaacContrib-Velocity-{Flat,Rough}-DobotQuadWheel``
  velocity-tracking environments. The wheel-legged variant drives the legs with position targets and the wheels
  with velocity targets.
* Added ``-Deploy`` variants of the four Dobot environments for hardware deployment. Their actor observes only
  what the robot's low-level DDS interface measures (no base linear velocity), the critic keeps it as a
  privileged observation, and command delay (0–20 ms), PD gains (±20 %), and ground friction are randomized.
