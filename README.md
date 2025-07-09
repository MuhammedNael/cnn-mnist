# CNN for Digit and Rotation Prediction

**CSE4097.1 - Introduction to Deep Learning**  
**Homework Project**  
**Authors**: Mohamad Nael Ayoubi, Enes Torluoğlu

---

## Purpose
This project develops a lightweight Convolutional Neural Network (CNN) using PyTorch to predict both the digit and its rotation angle from a modified MNIST dataset, balancing efficiency and high accuracy.

---

## Dataset
- **Source**: Modified MNIST (`torchvision.datasets`)
- **Modifications**:
  - Digits: 1, 2, 3, 4, 5, 7 (labels 0–5)
  - Rotations: 0°, 90°, 180°, 270° (labels 0–3)
- **Sample Structure**:
  - Image: 1x28x28 grayscale
  - Labels: Digit (categorical), Rotation (categorical)
- **Size**: 145,436 samples
  - **Training Set**: 116,348 (80%)
  - **Test Set**: 29,088 (20%)
- **Split Method**: `StratifiedShuffleSplit` for balanced digit/rotation distributions

*Refer to Figures 2 and 3 in the report for train/test data distribution charts.*

---

## Neural Network Architecture
The CNN processes 1x28x28 images and outputs digit and rotation predictions via two heads:

1. **Input**: 1x28x28
2. **Conv1**: 8x28x28, 3x3 filters, 8 filters, stride 1, padding 1
3. **Pool1**: 8x14x14, 2x2 max pooling, stride 2
4. **Conv2**: 16x14x14, 3x3 filters, 16 filters, stride 1, padding 1
5. **Pool2**: 16x7x7, 2x2 max pooling, stride 2
6. **Flattening**: 784 features
7. **Fully Connected**:
   - `fc_digit`: Predicts digit (6 classes)
   - `fc_rotation`: Predicts rotation (4 classes)

*Refer to Figure 1 in the report for architecture visualization.*

---

## Training Process
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss**: Combined digit and rotation losses
- **Sample Losses**:
  - Epoch 1: Digit: 0.2437, Rotation: 0.0354, Total: 0.2791
  - Epoch 10: Digit: 0.1909, Rotation: 0.0009, Total: 0.1917
  - Epoch 20: Digit: 0.0956, Rotation: 0.0016, Total: 0.0972

---

## Evaluation
### Training Set
- **Digit Accuracy**: 97.45% (F1: 0.9745 weighted, 0.9742 macro)
- **Rotation Accuracy**: 99.76% (F1: 0.9976 weighted/macro)
- **Overall Accuracy**: 97.26%

### Test Set
- **Digit Accuracy**: 96.68% (F1: 0.9668 weighted, 0.9664 macro)
- **Rotation Accuracy**: 99.52% (F1: 0.9952 weighted/macro)
- **Overall Accuracy**: 96.27%

*Refer to Figure 4 in the report for prediction samples.*

---

## Dependencies
- PyTorch
- torchvision
- scikit-learn (`StratifiedShuffleSplit`)

---

**Authors**: Mohamad Nael Ayoubi, Enes Torluoğlu  
**Course**: CSE4097.1 - Introduction to Deep Learning
