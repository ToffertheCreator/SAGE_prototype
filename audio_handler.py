import torch
from transformers import pipeline
from pydub import AudioSegment
import numpy as np
import io
import tempfile
import os

def convert_bytes_to_array(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    
    # Convert bytes to AudioSegment
    audio = AudioSegment.from_file(audio_file)
    
    # Debugging statements
    print("Original sample width:", audio.sample_width)
    print("Original channels:", audio.channels)
    print("Original frame rate:", audio.frame_rate)
    
    # Ensure the audio is in the correct format
    if audio.sample_width != 2:
        # Convert to 16-bit depth
        audio = audio.set_sample_width(2)
    
    if audio.channels != 1:
        # Convert to mono
        audio = audio.set_channels(1)
    
    # Debugging statements
    print("Adjusted sample width:", audio.sample_width)
    print("Adjusted channels:", audio.channels)
    
    # Convert to numpy array
    audio_array = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    return audio_array, sample_rate


def transcribe_audio(audio_bytes):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize the ASR pipeline
    try:
        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device=device,
        )
    except Exception as e:
        raise RuntimeError("Error initializing ASR pipeline: {}".format(e))

    # Convert bytes to audio array and sample rate
    audio_array, sample_rate = convert_bytes_to_array(audio_bytes)

    # Save the array to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_filename = temp_file.name
        # Save audio array to a file
        audio = AudioSegment(
            data=audio_array.tobytes(),
            sample_width=2,  # 16-bit depth
            frame_rate=sample_rate,
            channels=1
        )
        audio.export(temp_filename, format='wav')
    
    try:
        # Use the pipeline for transcription with the temporary file
        prediction = pipe(temp_filename, batch_size=1)
    except IndexError as e:
        raise IndexError("Model generated an out-of-bounds token index: {}".format(e))
    except Exception as e:
        raise RuntimeError("Error during transcription: {}".format(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return prediction['text']
