# Comparative Analysis of Adversarial Attack Methods

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📚 Overview

This repository contains the implementation and analysis of various adversarial attack methods on deep neural networks. The project provides a comprehensive comparative study of three prominent adversarial attack techniques: **FGSM (Fast Gradient Sign Method)**, **BIM (Basic Iterative Method)**, and **PGD (Projected Gradient Descent)**.

The research explores how these attacks affect model robustness, accuracy degradation, and the perceptual quality of adversarial examples across different attack parameters.

## 🎯 Key Features

- **Multiple Attack Implementations**: FGSM, BIM, and PGD adversarial attacks
- **Comprehensive Metrics**: Evaluation using accuracy drop, success rate, confidence degradation, L2 norm, PSNR, and SSIM
- **Visual Analysis**: Extensive visualizations comparing attack effectiveness and image quality
- **Class-wise Analysis**: Investigation of how class frequency impacts adversarial robustness
- **Complete Pipeline**: From data loading to attack generation and evaluation

## 📂 Repository Structure

```
.
├── ComparativeAnalyse.ipynb          # Main Jupyter notebook with complete analysis
├── Thesis/
│   └── Comparative_Analysis_of_Adverserial_Attacks_Methods_final_version.pdf
├── Images/                            # Visualization outputs and results
│   ├── BildBIM.png                   # BIM attack visualizations
│   ├── BildFGSM.png                  # FGSM attack visualizations
│   ├── PGDPicture.png                # PGD attack visualizations
│   ├── attacksucessrate.png          # Attack success rate comparison
│   ├── confidencedrop.png            # Model confidence degradation
│   ├── L2Norm.png                    # L2 norm analysis
│   ├── PSNR.png                      # Peak Signal-to-Noise Ratio metrics
│   ├── SSIM.png                      # Structural Similarity Index metrics
│   ├── execution_pipeline.png        # Overall execution pipeline
│   └── ...                           # Additional analysis visualizations
└── README.md                          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- CUDA-capable GPU (recommended for faster processing)
- Kaggle account (for dataset download)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NKWILI/Comparative-Analysis-Of-Adverserial-Attack-Method-Thesis-.git
   cd Comparative-Analysis-Of-Adverserial-Attack-Method-Thesis-
   ```

2. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib seaborn torch torchvision pillow ipython kagglehub scikit-image scikit-learn
   ```

3. **Set up Kaggle API credentials**
   - Download your `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Place it in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Running the Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook ComparativeAnalyse.ipynb
```

The notebook is organized into the following sections:
1. **Install & Import Libraries** - Setup environment
2. **Download & Prepare Dataset** - Load image dataset from Kaggle
3. **Data Exploration** - Analyze dataset distribution
4. **Load & Preprocess Images** - Prepare images for attack
5. **Attack Implementation** - FGSM, BIM, and PGD attacks with various epsilon values
6. **Class Frequency Analysis** - Impact of class distribution on robustness
7. **Comparative Evaluation** - Comprehensive comparison of attack methods

## 🔬 Attack Methods

### 1. Fast Gradient Sign Method (FGSM)
- Single-step attack using the sign of gradients
- Fast and computationally efficient
- Explores multiple epsilon (ε) values for perturbation strength

### 2. Basic Iterative Method (BIM)
- Iterative extension of FGSM
- Multiple small steps in the gradient direction
- Better attack success rate than FGSM

### 3. Projected Gradient Descent (PGD)
- Iterative attack with random initialization
- Projection step to maintain perturbation bounds
- Considered one of the strongest first-order adversarial attacks

## 📊 Evaluation Metrics

The project evaluates attacks using multiple metrics:

- **Accuracy Drop**: Reduction in model classification accuracy
- **Attack Success Rate**: Percentage of successfully fooled samples
- **Confidence Drop**: Degradation in model confidence scores
- **L2 Norm**: Euclidean distance between original and adversarial images
- **PSNR (Peak Signal-to-Noise Ratio)**: Perceptual quality metric
- **SSIM (Structural Similarity Index)**: Structural similarity measure

## 📈 Results

The analysis reveals:

- Trade-off between attack strength and perceptual quality
- PGD generally achieves higher success rates than FGSM and BIM
- Class frequency significantly impacts adversarial robustness
- Different attacks show varying effectiveness across epsilon values

Detailed results and visualizations are available in the `Images/` directory.

## 📄 Thesis Document

The complete thesis document is available in the `Thesis/` folder:
- [Comparative_Analysis_of_Adverserial_Attacks_Methods_final_version.pdf](Thesis/Comparative_Analysis_of_Adverserial_Attacks_Methods_final_version.pdf)

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **torchvision** - Pre-trained models and image transformations
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **scikit-image** - Image quality metrics (PSNR, SSIM)
- **scikit-learn** - Machine learning utilities

## 📚 References

Key papers and resources:
- Goodfellow, I. J., et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015.
- Kurakin, A., et al. "Adversarial examples in the physical world." ICLR 2017.
- Madry, A., et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or discussions about this research, please open an issue in this repository.

## 📝 License

This project is available for academic and research purposes.

---

**Note**: This is a research project for academic purposes. The adversarial attack implementations are intended for understanding model vulnerabilities and improving robustness, not for malicious use.
