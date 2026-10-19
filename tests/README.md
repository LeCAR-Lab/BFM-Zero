# Correctness regression tests

After installing the project dependencies, run from the repository root:

```bash
python -m unittest discover -s tests -v
```

These CPU tests exercise production code with small tensors. Inference tests mock
simulator, model and renderer boundaries; MuJoCo tests run the real engine with an
inline XML model. No checkpoint, motion dataset or GPU is needed.

- Default-pose joint positions, zero velocities and backend quaternion ordering.
- Manual and automatic resets refresh observations and history after writing state;
  untouched environments and ordinary steps retain the correct observations.
- The actual push flag with domain randomization enabled and disabled.
- Local-frame angular velocity and observation scaling in expert loading and both
  reference-observation builders, including zero velocities used by goal inference.
- Virtual-head world velocity and privileged observations match position finite
  differences across XYZ rotations, including tilted/inverted poses and zero speed.
- Tracking every requested motion to its full length, distinct videos for multiple
  motions and the existing `tracking.mp4` filename for a single motion.
- Joint-angle metrics along the time axis with physical units; short clips and
  invalid inputs; all joints in observation-space EMD and unchanged main EMD.
- MuJoCo root/joint reset, caller input preservation, empty resets, body-origin
  velocity checked against position finite differences, and COM randomization
  without changing mass, accumulating offsets or modifying live qpos/qvel/time.
- State refresh after physics substeps without advancing simulation time.
- COM offsets take effect when the original inertial and body frames coincide,
  including rotated roots, repeated randomization and restoring defaults.
- Same-step autoreset retains physical successors, history and terminal flags;
  short episodes, evaluation cuts, ring wrap and the last legal sampling start.
- Explicit replay survives save/load and reward inference; legacy replay is
  rejected before environment creation or overwriting saved configuration.
- Model/agent strict loading, shared parameter aliases, actual optimizer updates,
  resume configuration checks, reproducible seeds and distinct expert environments.
- The legacy tracking evaluator uses all robot joints and the same time-based
  joint-angle metrics as the main evaluator, excluding the floating root.
- Isaac Sim adapter uses link-origin velocities for observations and root resets;
  push velocities appear in the same observation and history sample.
- PD randomization updates both PhysX and actuator gains; sampled torque noise
  reaches the effort command and is cleared when disabled.
- Auxiliary critic architecture is independent, including update and save/load;
  legacy checkpoints retain the architecture actually used to train their weights.
- Checkpoint write/publish failures preserve the previous version; recovery handles
  interruption between directory renames. Training saves its final completed batch
  without an extra batch and retains the final scheduled evaluation.

The training evaluator replaces `mpjpe_l`, `vel_dist` and `accel_dist` with
`joint_angle_l2_rad`, `joint_velocity_l2_rad_s` and `joint_acceleration_l2_rad_s2`.
It reports time-averaged L2 norms over joints, not Cartesian position errors.
Velocity needs at least two frames and acceleration needs three; unavailable
derivative metrics are omitted. New auxiliary metrics and `obs_state_emd` are not
directly comparable to old logs. The main `emd` calculation is unchanged.

The corrected reference encoding changes the inputs to the backward model;
existing weights remain loadable, but identical policy behavior is not guaranteed.
These tests do not validate full Isaac Sim training, rendered policy quality,
GPU performance, ONNX execution or real-robot behavior.

New online replay stores explicit successors, including pre-reset timeout states.
Existing complete model weights remain loadable; replay from older checkpoints
cannot reconstruct the missing physical successors and is rejected for resume.
The default 5.12M buffer needs approximately 19 GiB more float payload. The policy
architecture, reward weights and configured DR ranges are unchanged.
