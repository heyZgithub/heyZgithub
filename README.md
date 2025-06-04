## Hi there 👋

<!--
**heyZgithub/heyZgithub** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

## Diffusion model training

This repository includes `train_diffusion.py` which trains a diffusion model with multi-head attention on CIFAR-10.

```bash
pip install torch torchvision diffusers
python train_diffusion.py --data_dir ./data --out_dir ./outputs --epochs 1
```

The script also reconstructs a few images using the trained model and generates new samples. Reconstruction quality is measured using a custom normalized mean absolute error function.
