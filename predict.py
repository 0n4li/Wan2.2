# predict.py
import tempfile
from cog import BasePredictor, Input, Path
import subprocess
import os
import glob

class Predictor(BasePredictor):
    def setup(self):
        """
        This method is called once when the model is loaded.
        The model weights are already downloaded in the build step (cog.yaml),
        so we just confirm the path here.
        """
        self.model_ckpt_dir = "./Wan2.2-S2V-14B/"
        print("Setup complete. Model weights are located at:", self.model_ckpt_dir)

    def predict(
        self,
        image: Path = Input(description="Reference image for the video."),
        audio: Path = Input(description="Driving audio file (e.g., .wav, .mp3)."),
        prompt: str = Input(
            description="Text prompt to guide the video generation.",
            default="A person is talking."
        ),
        size: str = Input(
            description="Area budget preset for s2v (affects max resolution).",
            default="1024*704",
            choices=[
                "720*1280",
                "1280*720",
                "480*832",
                "832*480",
                "1024*704",
                "704*1024",
                "704*1280",
                "1280*704",
            ],
        ),
        pose_video: Path = Input(
            description="Optional video file for pose guidance.",
            default=None
        )
    ) -> Path:
        """
        This method is called for each prediction request.
        It runs the model's inference script using the provided inputs.
        """
        
        # Clean up any previous outputs
        output_dir = Path(tempfile.mkdtemp())
        
        # Construct the command-line arguments for the inference script.
        # This uses the single-GPU command from the model's documentation.
        save_path = os.path.join(output_dir, "result.mp4")
        command = [
            "python", "generate.py",
            "--task", "s2v-14B",
            "--size", size,
            "--ckpt_dir", self.model_ckpt_dir,
            "--save_file", save_path,
            "--offload_model", "True",
            "--convert_model_dtype",
            "--prompt", prompt,
            "--image", str(image),
            "--audio", str(audio)
        ]

        # Add the optional pose_video argument if it's provided.
        if pose_video:
            command.extend(["--pose_video", str(pose_video)])

        print(f"Running command: {' '.join(command)}")

        # Execute the inference script.
        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            print("Inference script stdout:", process.stdout)
        except subprocess.CalledProcessError as e:
            # If the script fails, provide the error logs for debugging.
            print("Inference script stderr:", e.stderr)
            raise RuntimeError(f"Inference failed: {e.stderr}") from e

        # Find the generated video file in the output directory.
        output_files = glob.glob(os.path.join(output_dir, "*.mp4"))
        if not output_files:
            raise FileNotFoundError("Generation script finished, but no output .mp4 file was found.")
        
        # Return the path to the first video file found.
        return Path(output_files[0])