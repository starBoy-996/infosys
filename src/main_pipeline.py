from audio_embedding import audio_to_embedding  # <-- singular


embedding = audio_to_embedding("data/user_audio/test.wav")
print("Embedding length:", len(embedding))

"""
import argparse
from audio_embedding import audio_to_embedding

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
args = parser.parse_args()

embedding = audio_to_embedding(args.audio)

print(f"Embedding dimension: {len(embedding)}")


import argparse
from audio_embedding import audio_to_embedding

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
args = parser.parse_args()

embedding = audio_to_embedding(args.audio)

print("Embedding length:", len(embedding))
print("Embedding vector:", embedding)
"""


import argparse
from audio_embedding import audio_to_embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    args = parser.parse_args()

    embedding = audio_to_embedding(args.audio)
    print("Embedding shape:", embedding.shape)

if __name__ == "__main__":
    main()



import argparse
from audio_embedding import audio_to_embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    args = parser.parse_args()

    embedding = audio_to_embedding(args.audio)
    print("Embedding shape:", embedding.shape)

if __name__ == "__main__":
    main()
