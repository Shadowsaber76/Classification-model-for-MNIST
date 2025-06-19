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
| Epochs           | 10 (configurable)   |
| Scheduler        | StepLR (gamma=0.5)  |

Training and test accuracy are evaluated per epoch.

---

## 📉 Visualizations

### 🔺 Accuracy Plot
- Compares **training vs. test accuracy** across epochs.

### 📉 Smoothed Loss Curve
- Shows **loss vs. iterations** with smoothing using moving average.
- Log-scaled Y-axis to visualize exponential drop.

<img src="loss_plot_example.png" alt="Loss vs. Iterations" width="600">

---

## 🚀 How to Run

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/mnist-cnn-classifier.git
   cd mnist-cnn-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision scikit-learn matplotlib keras
   ```

3. **Run the training script**
   ```bash
   python train.py
   ```

4. **Output**
   - Accuracy plots
   - Loss curve
   - Sample predictions

---

## 📦 Sample Prediction Output

10 random test samples with model predictions:

```
True: 7 | Predicted: 7
True: 2 | Predicted: 2
True: 1 | Predicted: 1
...
```

![Sample Predictions](sample_predictions.png)

---

## 📚 Learnings & Takeaways

- Mini-batch training can be noisy; use smoothing for clarity.
- Log scaling helps visualize exponential loss decay.
- Dropout helps reduce overfitting significantly in this task.

---

## 🏁 Future Improvements

- Add early stopping
- Add validation split
- Visualize confusion matrix
- Export model with `torch.save()`

---

## 👨‍💻 Author

- Name: *Your Name*
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn / Blog / Portfolio (optional)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
