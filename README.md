# CXR Localization: Chest X-Ray Abnormality Detection

This project implements a robust deep learning pipeline for localizing abnormalities in chest X-rays. It utilizes the **Swin Transformer V2 (SwinV2)** architecture, adapted for object detection tasks (classification and bounding box regression), and is built using **PyTorch Lightning**.

## 🚀 Key Features

- **Swin Transformer V2 Backbone**: Utilizes the state-of-the-art SwinV2-B architecture for powerful image feature extraction.
- **Multi-task Learning**: A unified model head for simultaneous classification of 16 abnormality types and precise bounding box localization.
- **PyTorch Lightning Integration**: Clean, modular code for training, validation, and logging (TensorBoard support).
- **DICOM Support**: Built-in handling for DICOM image files commonly used in medical imaging.
- **Automatic Optimization**: Custom training loop with separate optimizers for classification and regression tasks.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vijay-K-2003/CXR_Localization.git
   cd CXR_Localization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The project is designed to work with the **VinBigData Chest X-ray Abnormalities Detection** dataset. 

- **Images**: Located in `VinBigDataCXR/train/` (DICOM format).
- **Annotations**: Provided in `VinBigDataCXR/train.csv` including labels and bounding box coordinates.
- **Classes**: Supports 16 distinct classes of abnormalities.

## 🏗️ Model Architecture

The `SwinV2ObjectDetector` consists of:
- **Backbone**: Swin Transformer V2 (384x384 input).
- **Classifier**: A linear layer mapping features to 16 classes.
- **Regressor**: A linear layer predicting normalized bounding box coordinates $[x, y, w, h]$.

## 📖 Usage

### Training

To start the training process, ensure your dataset is placed in the expected directory structure and run:

```bash
python main.py
```

> [!NOTE]
> Update the `img_dir` and `csv_file` paths in `main.py` if your dataset is stored in a different location.

### Logging

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir logs
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.