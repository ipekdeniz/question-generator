from sentence_transformers import SentenceTransformer
import os


def download_and_save():
    model_name = "BAAI/bge-small-en-v1.5"
    output_dir = "storage/embedding_cache/bge-small-en-v1.5-sbert"

    print(f"Downloading model '{model_name}'...")
    model = SentenceTransformer(model_name)
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    print(f"Saving model to '{output_dir}'...")
    model.save(output_dir)
    print("Done.")


if __name__ == "__main__":
    download_and_save()