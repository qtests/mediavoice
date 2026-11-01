

# 1. Install spandrel (modern model runner, zero dependency conflicts) and PyAV / tqdm
# !pip install -q spandrel av tqdm ffmpeg-python

import time
import subprocess
from pathlib import Path
import torch
import av
import torchvision.transforms.functional as TF
from spandrel import ModelLoader
import requests
from tqdm import tqdm

# =====================================================================
# 1. PURE PYTHON DOWNLOADER (Replaces !wget)
# =====================================================================
def download_model_if_missing(url: str, dest_path: Path):
    """Downloads a file in pure Python with a progress bar."""
    if dest_path.exists():
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ Downloading weights: {dest_path.name}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest_path.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("✅ Model downloaded successfully!\n")


# =====================================================================
#  RUN SCRIPT
# =====================================================================
if __name__ == "__main__":

    INPUT_VIDEO = "input.mp4"
    # MODEL_PATH = Path("./videomodels/RealESRGAN_x4plus_anime_6B.pth")

    MODELS_DIR = Path("videomodels")
    # MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    # MODEL_PATH = MODELS_DIR / "RealESRGAN_x4plus_anime_6B.pth"    
    
    MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    MODEL_PATH = MODELS_DIR / "realesr-animevideov3.pth"

    # 1. Download model if you don't have it locally yet
    download_model_if_missing(MODEL_URL, MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelLoader().load_from_file(str(MODEL_PATH)).to(device).eval()

    container = av.open(INPUT_VIDEO)
    video_stream = container.streams.video[0]
    orig_w, orig_h = video_stream.width, video_stream.height
    out_w, out_h = 1880, 1080

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
        "-r", "24", "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "temp_debug.mp4"
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    print("=" * 60)
    print("🔬 PROFILING 10 FRAMES (Milliseconds per operation)")
    print("=" * 60)

    with torch.no_grad():
        for i, frame in enumerate(container.decode(video=0)):
            if i >= 10:
                break

            # 1. Decode & Convert to PIL
            t0 = time.perf_counter()
            img = frame.to_image()
            t1 = time.perf_counter()

            # 2. Host to GPU copy
            tensor = TF.to_tensor(img).unsqueeze(0).to(device)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            # 3. AI Model Inference
            upscaled = model(tensor)
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            # 4. Interpolation
            upscaled = torch.nn.functional.interpolate(upscaled, size=(out_h, out_w), mode="bicubic")
            torch.cuda.synchronize()
            t4 = time.perf_counter()

            # 5. GPU to CPU copy
            out_bytes = (upscaled.squeeze(0).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy().tobytes()
            t5 = time.perf_counter()

            # 6. FFmpeg Pipe Write
            process.stdin.write(out_bytes)
            t6 = time.perf_counter()

            print(
                f"Frame {i+1:02d} | "
                f"Decode: {(t1-t0)*1000:5.1f}ms | "
                f"To GPU: {(t2-t1)*1000:5.1f}ms | "
                f"AI Model: {(t3-t2)*1000:6.1f}ms | "
                f"Downscale: {(t4-t3)*1000:5.1f}ms | "
                f"To CPU: {(t5-t4)*1000:5.1f}ms | "
                f"FFmpeg: {(t6-t5)*1000:6.1f}ms"
            )

    process.stdin.close()
    process.wait()
    container.close()
    if Path("temp_debug.mp4").exists():
        Path("temp_debug.mp4").unlink()