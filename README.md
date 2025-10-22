# Brain Tumor Classifier (PyTorch)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Unlicense-blue.svg)](#license)

A proof-of-concept brain MRI image classifier built with Python and PyTorch for educational and research purposes.

## ⚠️ IMPORTANT

**THIS IS NOT A MEDICAL DEVICE** — This is a research/demo project. See the [DISCLAIMER](#disclaimer) section before proceeding.

---

## Features

- **4-class classification**: glioma, meningioma, pituitary tumor, no tumor
- **CNN architecture** built with PyTorch
- **Data augmentation** pipeline for improved generalization
- **Complete workflow**: preprocessing, training, validation, and testing
- **Performance metrics** and visualization tools
- **Jupyter notebook** for interactive experimentation

---

## Project Structure

```
brain-tumor-classifier/
├── Main.ipynb              # Primary notebook (preprocessing, training, evaluation)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── check_data_structure.py # Data validation script
└── data/                  # Dataset folder (see below)
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

Each class folder contains MRI images in JPG/PNG format.

---

## Dataset

**Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (Kaggle)

### Download Instructions

#### Option 1: Kaggle CLI (Recommended)

```bash
# Requires Kaggle CLI configured with your API token
# Setup: https://github.com/Kaggle/kaggle-api#api-credentials

kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

#### Option 2: Manual Download

1. Visit the [dataset page](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Click "Download" (requires Kaggle account)
3. Extract to `data/` folder
4. Ensure the folder structure matches the layout above

### Verify Data Structure

Run this script to check your setup:

```bash
python check_data_structure.py
```

**check_data_structure.py**:
```python
import os

expected = ["Training", "Testing"]
classes = ["glioma", "meningioma", "notumor", "pituitary"]

for split in expected:
    split_path = os.path.join("data", split)
    if not os.path.isdir(split_path):
        raise SystemExit(f"❌ Missing folder: data/{split}")
    
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            raise SystemExit(f"❌ Missing class folder: {cls_path}")
        
        num_images = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"✓ {split}/{cls}: {num_images} images")

print("\n✅ Data structure verified. Ready to proceed!")
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** should include:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
Pillow>=9.5.0
jupyter>=1.0.0
tqdm>=4.65.0
```

### 4. Download Dataset

Follow the [Dataset](#dataset) section above.

---

## Usage

### Start Jupyter Notebook

```bash
jupyter notebook Main.ipynb
```

### Notebook Sections

The `Main.ipynb` notebook is organized into sequential sections:

1. **Setup & Configuration**
   - Import libraries
   - Set device (CPU/GPU)
   - Define hyperparameters

2. **Data Loading & Preprocessing**
   - Load images from folders
   - Apply transformations (resize, normalize, augment)
   - Create data loaders

3. **Model Definition**
   - Define CNN architecture
   - Initialize model and optimizer

4. **Training**
   - Train model on training set
   - Validate on validation set
   - Save checkpoints

5. **Evaluation**
   - Test on test set
   - Generate confusion matrix
   - Display sample predictions

**Run cells sequentially** or jump to specific sections as needed.

---

## Results

### Performance Metrics

- **Test Accuracy**: ~95% (on provided test split)
- **Classes**: 4 (glioma, meningioma, pituitary, no tumor)
- **Dataset Split**: 80% training, 20% testing (as provided)

### Important Notes

⚠️ **These results are specific to:**
- The exact dataset used (Kaggle Brain Tumor MRI Dataset)
- Controlled experimental conditions
- Specific train/test split provided

⚠️ **Performance may vary significantly with:**
- Different MRI scanners or protocols
- Real-world clinical images
- Different patient populations
- Varying image quality and acquisition parameters

**See the [DISCLAIMER](#disclaimer) section for critical information.**

---

## <a name="disclaimer"></a>⚠️ DISCLAIMER

### THIS IS NOT A MEDICAL DEVICE

This software is a **research and educational project ONLY**.

### DO NOT USE FOR:

- ❌ Clinical diagnosis
- ❌ Treatment decisions
- ❌ Patient care decisions
- ❌ Any medical purpose

### Important Limitations

1. **Not Validated**: This model has NOT undergone clinical validation or regulatory approval
2. **Dataset Specific**: Performance metrics are specific to the training dataset and may not generalize
3. **No Clinical Testing**: Has not been tested on real-world clinical data
4. **Research Only**: Intended for learning and research purposes only

### Factors Affecting Real-World Performance

Results may differ significantly due to:
- MRI scanner manufacturer and model differences
- Image acquisition protocols and parameters
- Patient demographics and tumor characteristics
- Image quality, resolution, and artifacts
- Slice selection and positioning
- Preprocessing variations

### Legal

**THE AUTHORS ASSUME NO RESPONSIBILITY FOR:**
- Misuse of this software
- Incorrect interpretations of results
- Any harm resulting from use of this software
- Decisions made based on model outputs

**For all medical decisions, always consult qualified healthcare professionals and use validated medical devices.**

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended for training)
- 8GB+ RAM recommended
- ~5GB disk space for dataset

---

## License

This project is released into the **public domain** — use it freely for any purpose.

**No warranty**: This software is provided "as-is" without any warranty. See [DISCLAIMER](#disclaimer) for medical use restrictions.

**Dataset**: The dataset has its own license. Please review the [dataset license](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed changes.

---

## Citation

If you use this code in your research or project, please cite:

```bibtex
@misc{brain_tumor_classifier_2025,
  author = {Your Name},
  title = {Brain Tumor Classifier: A PyTorch Implementation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/brain-tumor-classifier}
}
```

**Dataset Citation**:
```bibtex
@misc{nickparvar_brain_tumor_mri,
  author = {Masoud Nickparvar},
  title = {Brain Tumor MRI Dataset},
  year = {2021},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}
}
```

---

## Acknowledgments

- Dataset: [Masoud Nickparvar](https://www.kaggle.com/masoudnickparvar) via Kaggle
- Framework: [PyTorch](https://pytorch.org/)
- Inspiration: Medical imaging research community

---

## Contact

For questions, issues, or suggestions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/brain-tumor-classifier/issues)
- **Email**: your.email@example.com

---

## Roadmap

Potential future improvements:
- [ ] Add model explainability (Grad-CAM, attention maps)
- [ ] Implement additional architectures (ResNet, EfficientNet)
- [ ] Add cross-validation
- [ ] Create web interface for demo purposes
- [ ] Add more comprehensive testing suite
- [ ] Improve documentation with tutorials

---

**Remember**: This is a learning project. Always prioritize patient safety and use validated, approved medical devices for clinical applications.
