# Fourier Feature Network for 3D Vessel Reconstruction from Biplane Angiograms

This repository hosts the code and resources for the research project by Sean Wu, Naoki Kaneko, David S. Liebeskind, and Fabien Scalzo

## Usage

### Code Availability

This repository provides complete code for both the training and inference phases of our Fourier feature network. Users can replicate our results or adapt the methodology for their own datasets with the following resources:

- `training/fourier_features_training.py`: Script for training the network on your data.
- `infererence/fourier_feature_inference.py`: Script for running inference using a trained model to reconstruct 3D slices.

### Data Privacy

To respect patient privacy, the dataset used for this research will not be provided publicly. We encourage researchers to apply our methods to their own cerebra angiogram datasets.

### Getting Started

To get started with the provided scripts, you should clone this repository and install the required dependencies:

```bash
git clone [https://github.com/your-repository-url](https://github.com/FFVesselRecon/FFVesselRecon.git)
cd FFVesselRecon
pip install -r requirements.txt
```


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
