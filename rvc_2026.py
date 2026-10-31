
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
from infer_rvc_python import BaseLoader

def run_modern_rvc_pipeline(source_wav, model_pth, index_file, output_wav):
    print("[*] Initializing Modern RVC Core Engine...")

    converter = BaseLoader(only_cpu=False)

    tag = "my_voice"

    converter.apply_conf(
        tag=tag,
        file_model=model_pth,
        file_index=index_file if index_file else "",
        pitch_algo="rmvpe",
        pitch_lvl=0,
        index_influence=0.75,
        consonant_breath_protection=0.33,
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

    MODEL_NAME    = "model_markzuckerberg"
    MODEL_WEIGHTS = f"{MODEL_NAME}.pth"
    MODEL_INDEX   = f"{MODEL_NAME}.index"

    SOURCE        = "sample_dialogue.wav"
    OUTPUT        = f"{MODEL_NAME}_output.wav"

    run_modern_rvc_pipeline(
        source_wav=SOURCE,
        model_pth=MODEL_WEIGHTS,
        index_file=MODEL_INDEX if os.path.isfile(MODEL_INDEX) else None,
        output_wav=OUTPUT
    )