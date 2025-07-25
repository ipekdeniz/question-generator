import argparse
from sentence_transformers import SentenceTransformer
import os

def download_and_save(model_name: str, output_dir: str):
    print(f"Downloading model '{model_name}'...")
    model = SentenceTransformer(model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model to '{output_dir}'...")
    model.save(output_dir)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a Sentence Transformers model for offline use.")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name, e.g. 'bge-small-en-v1.5-sbert'")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save the model")
    args = parser.parse_args()
    download_and_save(args.model, args.output) 