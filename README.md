# GSoC Evaluation – CMS Tasks
## For Super resolution at the CMS detector

This repository contains solutions for the GSoC (Google Summer of Code) evaluation tasks involving two distinct challenges related to particle physics data:

1. **Electron/Photon Classification** – Using a ResNet-15-based model to classify electrons and photons based on detector data.  
2. **Super-Resolution** – Training a Enhanced Super Resolution Generative Adversarial Network (ESRGAN) to enhance the resolution of particle collision images.  

---

## Repository Structure
```
GSoC_Evaluation_CMS/
├── Task1/                             # Electron/Photon Classification
│   ├── models.py                     # ResNet-15 model definition
│   ├── train.py                      # Training and evaluation script
│   ├── comparison.png                # Performance comparison plot
│   ├── data/                         # Directory for dataset files (not included)
│   ├── expt_20_adam/                 # Model weights for 20-epoch run with Adam optimizer
│   ├── expt_20_adamw/                # Model weights for 20-epoch run with AdamW optimizer
│   ├── expt_50_adam/                 # Model weights for 50-epoch run with Adam optimizer
│   └── README.md                     # Task 1 details
│
├── Task2/                             # Super-Resolution with ESRGAN
│   ├── datasets.py                   # Custom dataset loader
│   ├── esrgan.py                     # Training and evaluation script
│   ├── models.py                     # Generator, Discriminator, Feature Extractor
│   ├── utils.py                      # Utility functions
│   ├── images/training/              # Output generated images
│   ├── saved_models/                 # Saved model checkpoints
│   └── README.md                     # Task 2 details
│
└── README.md                         # Project overview
```

---

## Tasks Overview
### 1. **Electron/Photon Classification with ResNet-15**
This task involves developing a ResNet-15-based binary classifier to distinguish between electrons and photons using 32×32 matrices representing hit energy and time.

#### **Dataset:**
- **SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5** – Electron dataset  
- **SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5** – Photon dataset  

#### **Model:**
- Based on ResNet-18 but modified to accept 2-channel inputs (hit energy and time).  
- Removal of the maxpool layer to retain spatial resolution.  
- Adjusted final output layer for binary classification.  

#### **Results:**
| Model Name     | Optimizer | Epochs | Test Accuracy (%) | Test Loss |
|---------------|-----------|--------|-------------------|-----------|
| expt_50_adam   | Adam      | 50     | 50.40%             | 1.01       |
| expt_20_adamw  | AdamW     | 20     | 74.70%             | 0.50       |
| expt_20_adam   | Adam      | 20     | 75.50%             | 0.51       |

---

### 2. **Super-Resolution with ESRGAN**
This task involves training an Enhanced Super-Resolution GAN (ESRGAN) to increase the resolution of low-resolution (LR) particle collision images.

#### **Dataset:**
- **X_jets_LR** – Low-resolution images `(3, 64, 64)`  
- **X_jets** – High-resolution images `(3, 125, 125)`  

#### **Model:**
- Generator: Uses Residual-in-Residual Dense Blocks (RRDB) with pixel shuffle for upsampling.  
- Discriminator: Uses convolutional layers and relativistic average GAN loss.  
- Feature Extractor: Based on VGG19 for content loss calculation.  

#### **Results:**
- Trained for 5 epochs using an **L40 GPU** from Lightning AI's free credits.  
- Initial results show promising output quality but further training is needed for better convergence.  

---

## Author
- **Kartik Bhatt**  
- 📧 [kartikbhtt7@gmail.com](mailto:kartikbhtt7@gmail.com)  

---