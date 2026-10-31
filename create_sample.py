import asyncio
import edge_tts
from pathlib import Path

INPUTS_DIR = Path(".")

# Dialogue from Episode 1: Oval Office Outbreak
SAMPLE_TEXT = "E-lon! We have a problem. A huge problem. The launch was supposed to be in three dimensions, not twenty-seven!"
SAMPLE_WAV = str(INPUTS_DIR / "sample_dialogue.wav")

async def generate_sample_speech():
    # Generate generic neutral speech to be converted
    communicate = edge_tts.Communicate(SAMPLE_TEXT, voice="en-US-GuyNeural")
    await communicate.save(SAMPLE_WAV)

asyncio.run(generate_sample_speech())