print("Pipeline runs successfully")

if __name__ == "__main__":
    word = "she"
    phoneme = "SH IY"
    audio = phoneme_to_audio(phoneme)
    embedding = audio_to_embedding(audio)
    print("Embedding:", embedding)


# src/main_pipeline.py

import argparse

from phoneme_converter import text_to_phoneme
from audio_generator import phoneme_to_audio
from audio_embedding import audio_to_embedding


def main(word: str):
    phoneme = text_to_phoneme(word)
    audio = phoneme_to_audio(phoneme)
    embedding = audio_to_embedding(audio)

    print("Word:", word)
    print("Phoneme:", phoneme)
    print("Embedding:", embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text → Phoneme → Audio → Embedding pipeline"
    )
    parser.add_argument(
        "--word",
        type=str,
        required=True,
        help="Input word for processing"
    )

    args = parser.parse_args()

    main(args.word)
