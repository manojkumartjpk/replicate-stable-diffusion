import os
from typing import List
import base64

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from cog import BasePredictor, Input, Path

# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        )
    ) -> List[str]:
        """Run a single prediction on the model"""
        image = self.pipe(prompt).images[0]
            
        image.save("output.png")
        my_output_string = ""
        with open("output.png", "rb") as img_file:
            my_output_string = base64.b64encode(img_file.read())
    
        return [my_output_string]
