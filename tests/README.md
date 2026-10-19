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
