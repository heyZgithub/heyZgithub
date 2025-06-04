import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

from diffusers import UNet2DModel, DDPMScheduler


def custom_metric(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute normalized mean absolute error as a custom metric."""
    diff = torch.abs(original - reconstructed)
    norm = torch.abs(original).mean()
    return float(diff.mean() / (norm + 1e-8))


class DiffusionTrainer:
    def __init__(self, data_dir: str, out_dir: str, img_size: int = 32, batch_size: int = 16,
                 lr: float = 1e-4, epochs: int = 1, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.CIFAR10(root=data_dir, download=True, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        self.model = UNet2DModel(
            sample_size=img_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 128),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            attention_head_dim=4,
        ).to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for images, _ in self.dataloader:
                images = images.to(self.device)
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps,
                                          (images.shape[0],), device=self.device).long()
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

                noise_pred = self.model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            print(f"Epoch {epoch+1}/{self.epochs} - loss: {loss.item():.4f}")

        self.model.save_pretrained(self.out_dir)

    def reconstruct(self, num_images: int = 8):
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=num_images, shuffle=True)
        images, _ = next(iter(loader))
        images = images.to(self.device)
        with torch.no_grad():
            noisy_images = images
            timesteps = torch.arange(self.noise_scheduler.num_train_timesteps - 1, -1, -1, device=self.device)
            for t in timesteps:
                noisy_images = self.noise_scheduler.step(
                    self.model(noisy_images, t).sample, t, noisy_images).prev_sample
        metric = custom_metric(images.cpu(), noisy_images.cpu())
        print(f"Reconstruction metric d: {metric:.4f}")
        torchvision.utils.save_image(noisy_images.cpu(), self.out_dir / "reconstructions.png")

    def sample(self, num_images: int = 8):
        self.model.eval()
        sample = torch.randn(num_images, 3, self.model.sample_size, self.model.sample_size, device=self.device)
        timesteps = torch.arange(self.noise_scheduler.num_train_timesteps - 1, -1, -1, device=self.device)
        with torch.no_grad():
            for t in timesteps:
                sample = self.noise_scheduler.step(
                    self.model(sample, t).sample, t, sample).prev_sample
        torchvision.utils.save_image(sample.cpu(), self.out_dir / "samples.png")


def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model with multi-head attention")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    trainer = DiffusionTrainer(args.data_dir, args.out_dir, epochs=args.epochs)
    trainer.train()
    trainer.reconstruct()
    trainer.sample()


if __name__ == "__main__":
    main()
