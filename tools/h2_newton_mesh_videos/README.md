# Unitree H2 Newton-Style Mesh Videos

This helper exports review videos that resemble the Newton OpenGL viewer: a
white Unitree H2 mesh beside the yellow SOMA skinned mesh on a checker ground.
It supports both `unitree_h2` and `unitree_h2_sonic`.

Run from the repository root. The first step retargets BVH files and caches H2
body transforms plus SOMA skinned mesh vertices. The second step renders MP4s
with Blender and ffmpeg.

```bash
uv run python tools/h2_newton_mesh_videos/generate_mesh_video_data.py \
  --robot-type unitree_h2 \
  --fps 24 \
  --out-dir outputs/h2_newton_videos/data

blender --background \
  --python tools/h2_newton_mesh_videos/render_mesh_videos_blender.py -- \
  --out-dir outputs/h2_newton_videos \
  --width 1280 \
  --height 720 \
  --fps 24 \
  outputs/h2_newton_videos/data/*.npz
```

By default, the data export uses five representative motions from
`assets/motions/bvh/`. Pass explicit `.bvh` paths after the options to render a
different set. The generated files are written under `outputs/`, which is ignored
by Git.

For Sonic H2 training assets, switch the robot type and output directory:

```bash
uv run python tools/h2_newton_mesh_videos/generate_mesh_video_data.py \
  --robot-type unitree_h2_sonic \
  --fps 24 \
  --out-dir outputs/h2_sonic_newton_videos/data

blender --background \
  --python tools/h2_newton_mesh_videos/render_mesh_videos_blender.py -- \
  --out-dir outputs/h2_sonic_newton_videos \
  --width 1280 \
  --height 720 \
  --fps 24 \
  outputs/h2_sonic_newton_videos/data/*.npz
```
