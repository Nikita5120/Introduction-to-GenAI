from flask import Flask, render_template, request, redirect, url_for
import torch
from model import Generator
from PIL import Image
import os
import random

app = Flask(__name__)

device = torch.device('cpu')

# Initialize and load the trained generator
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.eval()

@app.route('/')
def index():
    return render_template('index.html', random=random)

@app.route('/generate', methods=['POST'])
def generate():
    noise = torch.randn(1, 100).to(device)
    with torch.no_grad():
        generated_image = generator(noise)

    generated_image = (generated_image + 1) / 2  # De-normalize
    generated_image = generated_image.view(28, 28).cpu().detach().numpy()
    image = Image.fromarray((generated_image * 255).astype('uint8'), mode='L')
    image = image.resize((280, 280))  # Optional: upscale for UI
    image_path = 'static/generated_image.png'
    image.save(image_path)

    return render_template('index.html', image_url='generated_image.png', random=random)

if __name__ == '__main__':
    app.run(debug=True)
