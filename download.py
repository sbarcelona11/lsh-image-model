# This file runs during container build time to get model weights built into the container
# In this example: A Huggingface BERT model
from diffusers import StableDiffusionPipeline


def download_model():
    model_id = "sbarcelona11/KIDS-ILLUSTRATION-LSH"
    StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True, safety_checker=None)
    print("Model downloaded")


if __name__ == "__main__":
    download_model()
