import sys
import time
from pathlib import Path

import viser
from matplotlib import colormaps
from tqdm import tqdm

from vggt4d.visualization.scene import Scene4D

viser_port = 8080

bg_downsample_factor = 0.1
point_size = 0.001
# point_size = 0.02
fg_downsample_factor = 1.0
# fg_downsample_factor = 0.5
camera_frustum_scale = 0.01
# camera_frustum_scale = 0.1
axes_scale = 0.05
cam_line_width = 5
fps = 24


if __name__ == "__main__":
    scene_dir = sys.argv[1]
    scene_dir = Path(scene_dir)
    scene = Scene4D(scene_dir)
    server = viser.ViserServer(port=viser_port)

    server.scene.set_up_direction('+y')
    n_frame = scene.num_frame

    print("Loading frames...")
    fg_pts_handles = []
    bg_pts_handles = []
    frame_handles = []
    cam_handles = []
    for i in tqdm(range(n_frame), unit="frame"):
        frame_handles.append(server.scene.add_frame(
            f"/scene/frame_{i:04d}", show_axes=False, visible=True))

        # Load background points for this frame
        bg_pts, bg_rgb = scene.get_background_points(
            frame_id=i, downsample_factor=bg_downsample_factor, filter=False)
        bg_pts_handle = server.scene.add_point_cloud(
            name=f"/scene/frame_{i:04d}/background",
            points=bg_pts,
            colors=bg_rgb,
            point_size=point_size,
            point_shape="rounded",
        )
        bg_pts_handle.visible = True
        bg_pts_handles.append(bg_pts_handle)

        # Load dynamic points for this frame
        dyn_pts, dyn_rgb = scene.get_dynamic_points(
            i, downsample_factor=fg_downsample_factor)
        fg_pts_handle = server.scene.add_point_cloud(
            name=f"/scene/frame_{i:04d}/foreground",
            points=dyn_pts,
            colors=dyn_rgb,
            point_size=point_size,
            point_shape="rounded",
        )
        fg_pts_handle.visible = (i == 0)  # Only show first frame initially
        fg_pts_handles.append(fg_pts_handle)

        fov, aspect, wxyz, pos = scene.get_cam_frustum(i)
        cmap = colormaps.get_cmap("viridis")
        color = cmap(i / (n_frame - 1))[:3]
        cam_handles.append(server.scene.add_camera_frustum(
            f"/scene/frame_{i:04d}/frustum",
            fov=fov,
            scale=camera_frustum_scale,
            aspect=aspect,
            wxyz=wxyz,
            position=pos,
            color=color,
            line_width=cam_line_width
        ))
        server.scene.add_frame(
            f"/scene/frame_{i:04d}/frustum/axes",
            axes_length=camera_frustum_scale * axes_scale * 10,
            axes_radius=camera_frustum_scale * axes_scale,
        )

    with server.gui.add_folder("Playback"):
        # gui_playing = server.gui.add_checkbox("Playing", False)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=n_frame - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=fps
        )
        gui_point_size = server.gui.add_number(
            "Point size",
            min=0.0001,
            max=0.02,
            step=0.0002,
            initial_value=point_size,
        )

    with server.gui.add_folder("Background"):
        gui_show_all_bg = server.gui.add_button("Show All Background")
        gui_hide_all_bg = server.gui.add_button("Hide All Background")

    def update_point_size(new_size: float) -> None:
        new_size = max(0.0004, min(0.02, new_size))
        with server.atomic():
            for h in bg_pts_handles:
                h.point_size = new_size
            for h in fg_pts_handles:
                h.point_size = new_size

    @gui_point_size.on_update
    def _(_) -> None:
        update_point_size(gui_point_size.value)

    @gui_timestep.on_update
    def _(_) -> None:
        curr_timestep = gui_timestep.value
        with server.atomic():
            for i, fg_h in enumerate(fg_pts_handles):
                fg_h.visible = i == curr_timestep

    @gui_show_all_bg.on_click
    def _(_) -> None:
        with server.atomic():
            for bg_h in bg_pts_handles:
                bg_h.visible = True

    @gui_hide_all_bg.on_click
    def _(_) -> None:
        with server.atomic():
            for bg_h in bg_pts_handles:
                bg_h.visible = False

    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value

    # record scene
    record_fps = 30
    serializer = server.get_scene_serializer()
    for i in range(n_frame):
        with server.atomic():
            for j, fg_h in enumerate(fg_pts_handles):
                fg_h.visible = j == i
            for j, cam_h in enumerate(cam_handles):
                cam_h.visible = j == i
            serializer.insert_sleep(1.0 / record_fps)
    record_data = serializer.serialize()
    with open(scene_dir / "record.viser", "wb") as f:
        f.write(record_data)

    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % n_frame
        time.sleep(1.0 / gui_framerate.value)
