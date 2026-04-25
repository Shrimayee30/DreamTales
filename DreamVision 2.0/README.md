# DreamVision 2.0

Fresh workspace for the rebuilt DreamVision pipeline.

Planned setup:
- Jupyter notebook workflow
- PyTorch 2.7.0 environment on HiPerGator
- New dataset and training pipeline

Current focus:
- background-only GAN training
- Places365 as the scene dataset
- animation/cartoon styling as a later post-processing step

Notebook:
- [background_gan_places365.ipynb](/Users/shrimayee/sdeshpan/ADL/DreamTales/DreamVision%202.0/notebooks/background_gan_places365.ipynb)

Expected dataset layout on HiPerGator:

```text
DreamVision 2.0/
  data/
    places365/
      train/
        abbey/
        airport_terminal/
        ...
      val/
        abbey/
        airport_terminal/
        ...
  outputs/
    backgrounds/
      checkpoints/
      samples/
```
