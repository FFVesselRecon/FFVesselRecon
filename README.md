# Fourier Feature Network for 3D Vessel Reconstruction from Biplane Angiograms

This repository hosts the code and resources for the research project conducted by Sean Wu, Naoki Kaneko, David S. Liebeskind, and Fabien Scalzo, collaborating between Pepperdine University and UCLA. This project aims to tackle the challenging problem of 3D vessel reconstruction from biplane cerebral angiograms using Fourier feature networks.

## Abstract

3D reconstruction of biplane cerebral angiograms remains a significant challenge due to depth information loss and unknown pixel-wise correlations between input images. Occlusions from only two views further complicate the reconstruction of fine vessel details. This research uses a coordinate-based neural network with a deterministic Fourier feature mapping to reconstruct more accurate cerebral angiogram slices. Key metrics from our results include a peak signal-to-noise ratio (PSNR) of 26.32±0.36, and a structural similarity index measure (SSIM) of 61.38±1.79.


## Justification of Gaussian Fourier Features

We demonstrate the necessity of Gaussian Fourier Features with a scaling parameter B of 10.0, highlighting their ability to model higher frequency functions and enhance learning over basic or no positional encoding setups.



https://github.com/FFVesselRecon/FFVesselRecon/assets/167462507/829ab3bd-ec34-4c24-a0c6-10b80bedc34b




## Related Research

- [Backprojection-based 3D Reconstruction](#)
- [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](#)
- [2D to 3D reconstruction of biplane angiograms with conditional GAN](#)

## BibTeX Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{wu2024fourier,
  author={Wu, Sean and Kaneko, Naoki and Liebeskind, David and Scalzo, Fabien},
  title={Fourier Feature Network for 3D Vessel Reconstruction from Biplane Angiograms},
  journal={(In Review) Journal of Machine Vision and Applications},
  year={2024}
}
