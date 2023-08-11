import torch
from potassium import Potassium, Request, Response
from diffusers import DiffusionPipeline
from io import BytesIO
import base64

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model_id = "sbarcelona11/KIDS-ILLUSTRATION-LSH"
    model = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True, safety_checker=None)
    model.to(device)

    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    negative_prompt = "(worst quality, low quality:1.4), monochrome, zombie, (interlocked fingers), cleavage, nudity, naked, nude"
    image = model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7,
        num_inference_steps=request.json.get("steps", 30),
        width=512,
        height=512,
    ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=80)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return Response(
        json = {"output": img_str},
        status=200
    )

if __name__ == "__main__":
    app.serve()