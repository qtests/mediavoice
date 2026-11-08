import subprocess
import torch
import av
import torchvision.transforms.functional as TF
from spandrel import ModelLoader
from tqdm import tqdm
from pathlib import Path


def upscale_cartoon_video(
    input_video_path: str,
    output_video_path: str,
    model_path: Path,
    mode: str = "1080p",         # "1080p", "4k", or "custom"
    target_scale: float = 2.0,   # used only if mode == "custom"
    crf: int = 18,               # 18 = visually lossless H.264
):
    """
    Upscales 2D animation using RealESRGAN Anime 6B.
    Preserves exact aspect ratio, locks variable framerates to audio,
    and uses GPU acceleration on your RTX A4000.
    """
    in_path = Path(input_video_path).resolve()
    out_path = Path(output_video_path).resolve()

    if not in_path.is_file():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Active Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load RealESRGAN Anime model into GPU memory
    print("🧠 Loading RealESRGAN Anime 6B into GPU VRAM...")
    model = ModelLoader().load_from_file(str(model_path)).to(device).eval()

    # 2. Inspect video properties
    container = av.open(str(in_path))
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames or int(
        video_stream.duration * video_stream.time_base * video_stream.average_rate
    )
    # Use exact rational framerate (e.g. 2387/100) to eliminate audio drift
    rate_str = f"{video_stream.average_rate.numerator}/{video_stream.average_rate.denominator}"
    orig_w, orig_h = video_stream.width, video_stream.height

    # 3. Proportional dimension calculation (No horizontal stretching!)
    if mode == "1080p":
        # Locks height to 1080 and scales width proportionally (864x496 -> 1880x1080)
        out_h = 1080
        out_w = int(orig_w * (1080 / orig_h))
    elif mode == "4k":
        # Native AI 4x resolution (864x496 -> 3456x1984, zero interpolation needed)
        out_w = orig_w * 4
        out_h = orig_h * 4
    else:
        out_w = int(orig_w * target_scale)
        out_h = int(orig_h * target_scale)

    # Ensure even dimensions for H.264 encoder
    out_w -= out_w % 2
    out_h -= out_h % 2

    aspect_ratio = out_w / out_h
    print(f"🎬 Source Video   : {orig_w}x{orig_h} @ {float(video_stream.average_rate):.2f} fps ({total_frames} frames)")
    print(f"🚀 Output Target  : {out_w}x{out_h} [Aspect Ratio: {aspect_ratio:.2f}:1] ({mode.upper()})")

    # Temporary raw video stream
    temp_silent_video = in_path.parent / f"temp_{out_path.stem}_silent.mp4"

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",          # 👈 1. Suppresses all banner/codec text
        "-nostats",          
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", rate_str,
        "-i", "-",                     # Standard input pipe
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", str(crf),
        str(temp_silent_video),
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # 4. Frame-by-frame GPU Upscaling
    with torch.no_grad():
        for frame in tqdm(container.decode(video=0), total=total_frames, desc="Upscaling episode"):
            img = frame.to_image()
            tensor = TF.to_tensor(img).unsqueeze(0).to(device)

            # Native 4x AI Pass
            upscaled = model(tensor)

            # Only downsample if target is NOT native 4x
            if (out_w, out_h) != (orig_w * 4, orig_h * 4):
                upscaled = torch.nn.functional.interpolate(
                    upscaled,
                    size=(out_h, out_w),
                    mode="bicubic",
                    antialias=True,
                )

            # Convert to uint8 bytes and stream directly to FFmpeg
            out_bytes = (
                (upscaled.squeeze(0).clamp(0, 1) * 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .tobytes()
            )
            process.stdin.write(out_bytes)

    process.stdin.close()
    process.wait()
    container.close()

    # 5. Mux original audio back in while locking sync
    print("🔊 Muxing original audio track and verifying sync...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", str(temp_silent_video),
        "-i", str(in_path),
        "-map", "0:v:0",
        "-map", "1:a?",               # Keeps audio if present, skips cleanly if silent
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Cleanup temp file
    if temp_silent_video.exists():
        temp_silent_video.unlink()

    print(f"\n🎉 Finished! Undistorted upscaled video saved to:\n   👉 {out_path}")


if __name__ == "__main__":

    MODELS_DIR = Path("videomodels")
    # ANIME_MODEL_PATH = MODELS_DIR / "RealESRGAN_x4plus_anime_6B.pth" 
    ANIME_MODEL_PATH = MODELS_DIR / "realesr-animevideov3.pth"   


    TARGET_SCALE = "1080p"                                # "1080p", "4k", Converts 480p (854x480) -> 1080p (1920x1080)
    INPUT_CLIP   = "input.mp4"                            # Your stitched cartoon clip
    OUTPUT_CLIP  = f"output_upscaled_{TARGET_SCALE}.mp4"

    upscale_cartoon_video (
        input_video_path    = INPUT_CLIP,
        output_video_path   = OUTPUT_CLIP,
        model_path          = str(ANIME_MODEL_PATH),
        mode                = TARGET_SCALE,                   
        crf                 = 18
    )

    print("Finito!")
