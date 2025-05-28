
# 🧠 Hybrid AI Classifier - Tabular + Image Data (Fashion MNIST)

This project demonstrates a **hybrid deep learning model** that combines **image data** (Fashion MNIST) and **tabular data** (synthetic features like price, rating, and category) to classify fashion items using **TensorFlow/Keras**. It showcases how to merge multiple data modalities into a single predictive system.

---

## 🚀 Project Overview

- **Image Input**: Fashion MNIST grayscale images (28x28), converted to RGB (32x32x3) for MobileNetV2.
- **Tabular Input**: Synthetic numerical (`price`, `rating`) and categorical (`category`) features.
- **Architecture**:
  - Pretrained **MobileNetV2** handles image features.
  - Fully connected layers process tabular data.
  - Both branches are concatenated and fed into dense layers for classification.

---

## 📁 Folder Structure

```
hybrid-ai-classifier/
├── data/
│   └── tabular.csv          # Synthetic tabular data
├── notebooks/
│   └── training.ipynb       #  Notebook with full code 
├── README.md

```

---

## 🧠 Model Architecture

```text
                +---------------------+       +---------------------+
                |   Image Input (CNN) |       |  Tabular Input (DNN)|
                +---------------------+       +---------------------+
                           ↓                            ↓
             Pretrained MobileNetV2         Dense → Dense
                           ↓                            ↓
                   GlobalAvgPooling                 Dense
                           ↓                            ↓
                      Concatenate Both Modalities → Dense → Output (Softmax)
```

---

## 📊 Results

- **Loss Function**: `Sparse Categorical Crossentropy`
- **Optimizer**: `Adam`
- **Accuracy**: ~85% after tuning
- **Epochs**: 10–50 (configurable)

---

## 🛠️ Setup Instructions

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

### 🔹 Run the Notebook

Use Jupyter Notebook or Google Colab to run:

```
notebooks/training.ipynb
```

It loads data, builds the model, trains it, and saves the `.h5` file in `/models`.

---

## 📈 Sample Metrics Plot

You can optionally add this section after plotting loss/accuracy graphs:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label="train acc")
plt.plot(history.history['val_accuracy'], label="val acc")
plt.title("Training Accuracy")
plt.legend()
plt.show()
```

---

## 📦 Future Improvements

- Add dropout/batch norm for regularization
- Use YOLOv8 for real-time image streams
- Integrate real tabular datasets (e.g., product metadata)
- Export model to ONNX or TF Lite

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.



## 🙋‍♂️ Author

**K H Srujan Gowda**  
📧 srujangowda582003@gmail.com  
📞 +91 8105740961
