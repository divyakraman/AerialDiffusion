import requests
from PIL import Image
from io import BytesIO
import torch
import os
from diffusers import DiffusionPipeline, DDIMScheduler
import PIL
import cv2
import numpy as np 
from scipy import ndimage #rotation angle in degree
#import matplotlib.pyplot as plt 

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
torch.hub.set_dir('/scratch0/')

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
        safety_checker=None,
    use_auth_token=False,
    custom_pipeline='./models/aerialDiffusion', cache_dir = 'dir_name',
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
).to(device)


generator = torch.Generator("cuda").manual_seed(0)
seed = 0

prompt = "a pizza and garlic bread on a dining table."

init_image = PIL.Image.open('dataset/images/image1.png').convert("RGB")
#init_image = init_image.resize((512, 512))
init_image = init_image.resize((256, 256))
image_orig = np.array(init_image)
image_orig = PIL.Image.fromarray(image_orig)


init_image = init_image.resize((256, 256))
init_image = np.array(init_image)
H = 256
W = 256 
pts1 = np.float32([[0,0],[H,0],[H,W],[0,W]])
pts2 = np.float32([[0,W],[H,0],[H,W],[0,2*W]])
M1 = cv2.getPerspectiveTransform(pts1,pts2)
init_image = cv2.warpPerspective(init_image,M1,(2*W,2*H))
init_image = PIL.Image.fromarray(init_image)
init_image = init_image.resize((512, 512))

res = pipe.train(
    prompt,
    image=init_image,
    generator=generator, image_orig = image_orig, text_embedding_optimization_steps = 500,
        model_fine_tuning_optimization_steps = 1000)

res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.1.png')
res = pipe(alpha=0.15, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.15.png')
res = pipe(alpha=0.2, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.2.png')
res = pipe(alpha=0.25, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.25.png')
res = pipe(alpha=0.3, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.3.png')
res = pipe(alpha=0.35, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.35.png')
res = pipe(alpha=0.4, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.4.png')
res = pipe(alpha=0.45, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.45.png')
res = pipe(alpha=0.5, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.5.png')
res = pipe(alpha=0.55, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.55.png')
res = pipe(alpha=0.6, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.6.png')
res = pipe(alpha=0.65, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.65.png')
res = pipe(alpha=0.7, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.7.png')
res = pipe(alpha=0.75, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.75.png')
res = pipe(alpha=0.8, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.8.png')
res = pipe(alpha=0.85, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.85.png')
res = pipe(alpha=0.9, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.9.png')
res = pipe(alpha=0.95, guidance_scale=7.5, num_inference_steps=50, image_orig = image_orig)
image = res.images[0]
image.save('./dataset/images/alpha_0.95.png')
