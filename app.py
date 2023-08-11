import torch
from potassium import Potassium, Request, Response
from diffusers import StableDiffusionPipeline
from auth_token import auth_token

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model_id = "sbarcelona11/KIDS-ILLUSTRATION-LSH"
    model = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)
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
    outputs = model(prompt, num_images_per_prompt=1, height=512, width=512).images

    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()