import rerun as rr
import rerun.blueprint as rrb
import depth_pro
import subprocess

import torch
import os
import gradio as gr
from gradio_rerun import Rerun
import spaces
from PIL import Image
import tempfile
import cv2

# Run the script to get pretrained models
if not os.path.exists("./checkpoints/depth_pro.pt"):
    print("downloading pretrained model")
    subprocess.run(["bash", "get_pretrained_models.sh"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and preprocessing transform
print("loading model...")
model, transform = depth_pro.create_model_and_transforms()
model = model.to(device)
model.eval()


def resize_image(image_buffer, max_size=256):
    with Image.fromarray(image_buffer) as img:
        # Calculate the new size while maintaining aspect ratio
        ratio = max_size / max(img.size)
        new_size = tuple([int(x * ratio) for x in img.size])

        # Resize the image
        img = img.resize(new_size, Image.LANCZOS)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            img.save(temp_file, format="PNG")
            return temp_file.name


@spaces.GPU(duration=20)
def predict_depth(input_image):
    # Preprocess the image
    result = depth_pro.load_rgb(input_image)
    image = result[0]
    f_px = result[-1]  # Assuming f_px is the last item in the returned tuple
    image = transform(image)
    image = image.to(device)

    # Run inference
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m]
    focallength_px = prediction["focallength_px"]  # Focal length in pixels

    # Convert depth to numpy array if it's a torch tensor
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()

    # Convert focal length to a float if it's a torch tensor
    if isinstance(focallength_px, torch.Tensor):
        focallength_px = focallength_px.item()

    # Ensure depth is a 2D numpy array
    if depth.ndim != 2:
        depth = depth.squeeze()

    # Clip depth values to 0m - 10m
    depth = depth.clip(0, 10)

    return depth, focallength_px


@rr.thread_local_stream("rerun_example_ml_depth_pro")
def run_rerun(path_to_video):
    print("video path:", path_to_video)
    stream = rr.binary_stream()

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="/"),
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin="/world/camera/depth",
                ),
                rrb.Spatial2DView(origin="/world/camera/frame"),
            ),
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)

    yield stream.read()

    video_asset = rr.AssetVideo(path=path_to_video)
    rr.log("world/video", video_asset, static=True)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns = video_asset.read_frame_timestamps_ns()

    cap = cv2.VideoCapture(path_to_video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    # limit the number of frames to 10 seconds of video
    max_frames = min(10 * fps_video, num_frames)

    for i in range(len(frame_timestamps_ns)):
        if i >= max_frames:
            raise gr.Error("Reached the maximum number of frames to process")

        ret, frame = cap.read()
        if not ret:
            break

        temp_file = None
        try:
            # Resize the image to make the inference faster
            temp_file = resize_image(frame, max_size=256)

            depth, focal_length = predict_depth(temp_file)

            # find x and y scale factors, which can be applied to image
            x_scale = depth.shape[1] / frame.shape[1]
            y_scale = depth.shape[0] / frame.shape[0]

            rr.set_time_nanos("video_time", frame_timestamps_ns[i])
            rr.log(
                "world/camera/depth",
                rr.DepthImage(depth, meter=1),
            )

            rr.log(
                "world/camera/frame",
                rr.VideoFrameReference(
                    timestamp=rr.components.VideoTimestamp(nanoseconds=frame_timestamps_ns[i]),
                    video_reference="world/video",
                ),
                rr.Transform3D(scale=(x_scale, y_scale, 1)),
            )

            rr.log(
                "world/camera",
                rr.Pinhole(
                    focal_length=focal_length,
                    width=depth.shape[1],
                    height=depth.shape[0],
                    principal_point=(depth.shape[1] / 2, depth.shape[0] / 2),
                    camera_xyz=rr.ViewCoordinates.FLU,
                    image_plane_distance=depth.max(),
                ),
            )

            yield stream.read()
        except Exception as e:
            raise gr.Error(f"An error has occurred: {e}")
        finally:
            # Clean up the temporary file
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    yield stream.read()


with gr.Blocks() as interface:
    gr.Markdown(
        """
        # DepthPro Rerun Demo

        [DepthPro](https://huggingface.co/apple/DepthPro) is a fast metric depth prediction model. Simply upload a video to visualize the depth predictions in real-time.

        High resolution videos will be automatically resized to 256x256 pixels, to speed up the inference and visualize multiple frames.
        """
    )
    with gr.Row():
        with gr.Column(variant="compact"):
            video = gr.Video(format="mp4", interactive=True, label="Video", include_audio=False)
            visualize = gr.Button("Visualize ML Depth Pro")
        with gr.Column():
            viewer = Rerun(
                streaming=True,
            )
        visualize.click(run_rerun, inputs=[video], outputs=[viewer])


if __name__ == "__main__":
    interface.launch()
