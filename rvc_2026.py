
# pip install infer_rvc_python
#   https://pypi.org/project/infer_rvc_python/
# Alternative: 
#   https://github.com/nakshatra-garg/rvc-no-gui

# Competitor: github.com/Jerrister/X-VC

# Find voices: 
#   https://huggingface.co/models?search=rvc
#   https://voice-models.com
#   https://aivoices.gg/


import os
import subprocess
from infer_rvc_python import BaseLoader


def normalize_loudness(input_path: str, output_path: str):
    """
    Extracts audio from video, applies EBU R128 
    loudness normalization, and saves as WAV/MP4.
    """

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")    

    if output_path[-4:].lower() == ".wav":
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vn",                                  # Drop video stream
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", "pcm_s16le",                    # Uncompressed 16-bit WAV
            "-ar", "44100",
            output_path
        ]
    else:        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v", "copy",
            output_path
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("FFmpeg Error:\n", result.stderr)
        raise RuntimeError("FFmpeg failed!")
    
    print(f"[✓] Normalized audio to: {output_path}")


def run_modern_rvc_pipeline(source_wav, model_pth, index_file, model_folder, output_wav, tag="my_voice", quality="natural"):
    print("[*] Initializing Modern RVC Core Engine...")

    # Quality presets
    # index_influence:              0.55 - 0.70: Higher = more like the trained voice, but more artifacts. Lower = cleaner but less similar
    # consonant_breath_protection:  0.25 - 0.35: Lower  = stronger protection of consonants & breaths (less robotic). Too low makes it sound muted.
    # envelope_ratio:               0.4  - 0.6:  Higher = keeps more of the original volume dynamics (more natural).
    # respiration_median_filtering:    3 - 5:    Higher = smoother pitch, less breathiness.

    presets = {
        "clean": {          # cleaner, fewer artifacts, less similar
            "index_influence"               : 0.50,
            "respiration_median_filtering"  : 5,
            "envelope_ratio"                : 0.55,
            "consonant_breath_protection"   : 0.25,
        },
        "natural": {        # balanced (recommended starting point)
            "index_influence"               : 0.65,
            "respiration_median_filtering"  : 4,
            "envelope_ratio"                : 0.45,
            "consonant_breath_protection"   : 0.28,
        },
        "similar": {        # maximum similarity to the trained voice
            "index_influence"               : 0.78,
            "respiration_median_filtering"  : 3,
            "envelope_ratio"                : 0.30,
            "consonant_breath_protection"   : 0.33,
        },
    }

    if quality not in presets:
        raise ValueError(f"Unknown quality preset: {quality}. Use 'clean', 'natural' or 'similar'")

    conf = presets[quality]
    print(f"[*] Using quality preset: {quality}")

    local_rmvpe = os.path.join(model_folder, "rmvpe.pt")
    rmvpe_arg = local_rmvpe if os.path.isfile(local_rmvpe) else None    

    converter = BaseLoader(only_cpu=False, rmvpe_path=rmvpe_arg)

    model_pth = os.path.join(model_folder, model_pth)
    index_file = os.path.join(model_folder, index_file)
    index_file = index_file if os.path.isfile(MODEL_INDEX) else ""
    
    converter.apply_conf(
        tag         = tag,
        file_model  = model_pth,
        file_index  = index_file,
        pitch_algo  = "rmvpe+",         # "rmvpe", "rmvpe+"
        pitch_lvl   = 0,

        index_influence                 = conf["index_influence"],
        respiration_median_filtering    = conf["respiration_median_filtering"],
        envelope_ratio                  = conf["envelope_ratio"],
        consonant_breath_protection     = conf["consonant_breath_protection"],
        resample_sr                     = 0,
    )    

    print(f"[*] Loading Target Model: {os.path.basename(model_pth)}")

    result = converter(
        audio_files = source_wav,
        tag_list    = tag,               # ← fixed here
        overwrite   = False,
        parallel_workers = 1
    )

    print(f"[✓] Processing successful!")
    print("Result:", result)

    # Optional: rename the generated file to your desired output name
    if result and isinstance(result, list) and len(result) > 0:
        generated = result[0]
        if generated != output_wav and os.path.exists(generated):
            os.replace(generated, output_wav)
            print(f"Saved as: {output_wav}")


if __name__ == "__main__":

    SOURCE_MP4       = "input.mp4"
    SOURCE_WAV       = "dialogue.wav"

    normalize_loudness(SOURCE_MP4, SOURCE_WAV)

    MODEL_FOLDER  = "voimodels"
    MODEL_NAME    = "model_donald3"
    MODEL_WEIGHTS = f"{MODEL_NAME}.pth"
    MODEL_INDEX   = f"{MODEL_NAME}.index"
    MODEL_TAG     = f"{MODEL_NAME}_voice"
    OUTPUT        = f"{MODEL_NAME}_output.wav"

    run_modern_rvc_pipeline(
        source_wav   = SOURCE_WAV,
        model_pth    = MODEL_WEIGHTS,
        index_file   = MODEL_INDEX,
        model_folder = MODEL_FOLDER,
        output_wav   = OUTPUT,
        tag          = MODEL_TAG
    )