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


def resize_image(image_path, max_size=1536):
    with Image.open(image_path) as img:
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
def run_rerun(path_to_image):
    stream = rr.binary_stream()

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="/"),
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin="/world/camera/depth",
                ),
                rrb.Spatial2DView(origin="/world/camera/image"),
            ),
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)

    yield stream.read()

    temp_file = None
    try:
        temp_file = resize_image(path_to_image)
        rr.log("world/camera/image", rr.EncodedImage(path=temp_file))
        yield stream.read()

        depth, focal_length = predict_depth(temp_file)

        rr.log(
            "world/camera/depth",
            rr.DepthImage(depth, meter=1),
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
    except Exception as e:
        rr.log(
            "error",
            rr.TextLog(f"An error has occurred: {e}", level=rr.TextLogLevel.ERROR),
        )
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

    yield stream.read()


# Example images
example_images = [
    "examples/lemur.jpg",
    "examples/silly-cat.png",
]

with gr.Blocks() as interface:
    gr.Markdown(
        """
        # DepthPro Rerun Demo

        [DepthPro](https://huggingface.co/apple/DepthPro) is a fast metric depth prediction model. Simply upload an image to predict its inverse depth map and focal length. Large images will be automatically resized to 1536x1536 pixels.
        """
    )
    with gr.Row():
        with gr.Column(variant="compact"):
            image = gr.Image(type="filepath", interactive=True, label="Image")
            visualize = gr.Button("Visualize ML Depth Pro")
            examples = gr.Examples(
                example_images,
                label="Example Images",
                inputs=[image],
            )
        with gr.Column():
            viewer = Rerun(
                streaming=True,
            )
        visualize.click(run_rerun, inputs=[image], outputs=[viewer])


if __name__ == "__main__":
    interface.launch()
