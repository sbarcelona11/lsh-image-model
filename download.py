# This file runs during container build time to get model weights built into the container
# In this example: A Huggingface BERT model
from transformers import pipeline
def download_model():
    # model_id = "sbarcelona11/KIDS-ILLUSTRATION-LSH"
    # pipeline('fill-mask', model=model_id)
    print("Model downloaded")

if __name__ == "__main__":
    download_model()
