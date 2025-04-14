import torch
from generator import Generator
from config import noise_dim, device
from torchvision.utils import save_image

def generate_and_save_image(model_path='models/generator.pth', image_path='static/generated.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    z = torch.randn(1, noise_dim).to(device)
    img = model(z).view(1, 1, 28, 28)
    save_image(img, image_path)
