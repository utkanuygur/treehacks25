from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import random


def text_to_speech(
    comment: str,
    stochastic: bool = False,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
) -> bytes:
    """
    Convert the given comment into speech using the ElevenLabs API.

    :@param comment: The text to be converted into speech.
    :@param stochastic: If True, choose a random voice from the voice list. 
                       If False, use the default voice (George).
    :@param model_id: Model ID for speech generation (default: "eleven_multilingual_v2").
    :@param output_format: Audio format (default: "mp3_44100_128").
    :@return: Raw audio bytes.
    """
    load_dotenv()
    # Predefined voices: name -> ID
    ALL_VOICES = {
        "Aria": "9BWtsMINqrJLrRacOk9x",
        "Roger": "CwhRBWXzGAHq8TQ4Fs17",
        "Sarah": "EXAVITQu4vr4xnSDxMaL",
        "Laura": "FGY2WhTYpPnrIDTdsKH5",
        "Charlie": "IKne3meq5aSn9XLyUdCD",
        "George": "JBFqnCBsd6RMkjVDRZzb",  # default
        "Callum": "N2lVS1w4EtoT3dr4eOWO",
        "River": "SAz9YHcvj6GT2YYXdXww",
        "Liam": "TX3LPaxmHKxFdv7VOQHJ",
        "Charlotte": "XB0fDUnXU5powFXDhCwa",
        "Matilda": "XrExE9yKIg1WjnnlVkGX",
        "Will": "bIHbv24MWmeRgasZH58o",
        "Jessica": "cgSgspJ2msm6clMCkdW9",
        "Eric": "cjVigY5qzO86Huf0OWal",
        "Chris": "iP95p4xoKVk53GoZ742B",
        "Brian": "nPczCjzI2devNBz1zQrb",
        "Daniel": "onwK4e9ZLuTAKqWW03F9",
        "Lily": "pFZP5JQG7iQjIQuC4Bku",
        "Bill": "pqHfZKP75CvOlQylNhV4",
    }

    ALL_VOICE_IDS = list(ALL_VOICES.values())
    # Determine the voice ID
    if stochastic:
        voice_id = random.choice(ALL_VOICE_IDS)
    else:
        voice_id = ALL_VOICES["George"]  # default voice

    # Initialize the ElevenLabs client
    client = ElevenLabs(api_key='sk_3fc41945004fb9fdc53ce89471617c5e8fea9384510810bd')

    # Convert text to speech
    audio_data = client.text_to_speech.convert(
        text=comment,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )
    return audio_data


if __name__ == "__main__":

    # Generate audio with a random voice
    audio_data = text_to_speech(
        comment="Hello from a random voice!",
        stochastic=True
    )

    # Print raw bytes for quick verification
    print(audio_data)

    # Play the audio in Python
    play(audio_data)