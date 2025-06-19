# Classification-model-for-MNIST

A Convolutional Neural Network (CNN) built in PyTorch to classify handwritten digits from 0 to 9 using the **MNIST** dataset. The model achieves high accuracy by using a 3-layer CNN with dropout, normalization, and proper regularization techniques.

---

## 📌 Project Overview

- **Dataset**: MNIST (from Keras)
- **Model**: 3-layer CNN
- **Framework**: PyTorch
- **Goal**: Multiclass classification (10 classes: digits 0–9)
- **Accuracy**: Achieves >99% test accuracy in 5 epochs

---

## 📁 Dataset: MNIST

The MNIST dataset consists of:
- 🖼️ 60,000 training images
- 🧪 10,000 test images
- Grayscale digits (28x28 pixels)

Data is loaded using:
```python
from keras.datasets import mnist
```

---

## 🧠 Model Architecture

```text
Input: (1 x 28 x 28)
│
├── Conv2D(1 → 32, 3x3) + ReLU
├── MaxPool(2x2)
│
├── Conv2D(32 → 64, 3x3) + ReLU
├── MaxPool(2x2)
│
├── Conv2D(64 → 128, 3x3) + ReLU
├── MaxPool(2x2)
│
├── Flatten (128 * 3 * 3)
├── Dense(256) + ReLU + Dropout(0.5)
└── Dense(10) → Softmax (via CrossEntropyLoss)
```

---

## ⚙️ Training Configuration

| Setting           | Value               |
|------------------|---------------------|
| Optimizer        | Adam                |
| Loss Function    | CrossEntropyLoss     |
| Learning Rate    | 0.0005              |
| Weight Decay     | 1e-4                |
| Batch Size       | 128                 |
| Epochs           | 5 (configurable)   |
| Scheduler        | StepLR (gamma=0.5)  |

Training and test accuracy are evaluated per epoch.

---

## 📉 Visualizations

### 🔺 Accuracy Plot
- Compares **training vs. test accuracy** across epochs.
- 
![Train VS TEST](https://github.com/Shadowsaber76/Classification-model-for-MNIST/blob/main/Data/Train%20vs.%20Test%20Accuracy%20Curve.png?raw=true)

### 📉 Smoothed Loss Curve
- Shows **loss vs. iterations** with smoothing using moving average.
- Log-scaled Y-axis to visualize exponential drop.

![LOSS VS ITERATIONS](https://github.com/Shadowsaber76/Classification-model-for-MNIST/blob/main/Data/Loss%20vs.%20Iterations.png?raw=true)

---

## 📦 Sample Prediction Output

10 random test samples with model predictions:

```
True: 7 | Predicted: 7
True: 2 | Predicted: 2
True: 1 | Predicted: 1
...
```
![Sample Predictions](https://github.com/Shadowsaber76/Classification-model-for-MNIST/blob/main/Data/Results.png?raw=true)

---

## 👨‍💻 Author

- Name: *Gaurav Sonawane*
- GitHub: [@Shadowsaber76](https://github.com/Shadowsaber76)
- [E-mail](mailto:f20241310@pilani.bits-pilani.ac.in?subject=[GitHub]%20MNIST%20Classifier)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
