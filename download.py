# This file runs during container build time to get model weights built into the container
# In this example: A Huggingface BERT model
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler


def download_model():
    repo_id = "sbarcelona11/KIDS-ILLUSTRATION-LSH"
    # StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True, safety_checker=None)
    # ddpm = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")
    DiffusionPipeline.from_pretrained(
        repo_id,
        use_safetensors=True,
        torch_dtype=torch.float16,
        # scheduler=ddpm
    ).to("cuda")
    print("Model downloaded")


if __name__ == "__main__":
    download_model()
