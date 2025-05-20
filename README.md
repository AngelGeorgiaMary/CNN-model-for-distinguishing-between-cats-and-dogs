# CNN-model-for-distinguishing-between-cats-and-dogs
Developed and optimized a CNN model for distinguishing between cats and dogs using a dataset from Kaggle. Implemented Conv2D and MaxPooling2D layers, employing a sigmoid function for binary classification. Successfully attained high validation accuracy
# 🐱🐶 Image Classification: Cat vs Dog

This project is a simple binary image classification task using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The model distinguishes between images of cats and dogs.

## 📁 Dataset

The dataset is stored in CSV format and loaded using NumPy. Each image is 100x100 pixels with 3 RGB channels.

- `input.csv` — Training image data  
- `labels.csv` — Training labels (0 = dog, 1 = cat)  
- `input_test.csv` — Test image data  
- `labels_test.csv` — Test labels  

## 🧠 Model Architecture

CNN model architecture:

- `Conv2D(32, (3,3), ReLU)` + `MaxPooling2D(2,2)`
- `Conv2D(32, (3,3), ReLU)` + `MaxPooling2D(2,2)`
- `Flatten`
- `Dense(64, ReLU)`
- `Dense(1, Sigmoid)`

Compiled using:

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

🚀 Training
model.fit(X_train, Y_train, epochs=5, batch_size=64)

📈 Evaluation
model.evaluate(X_test, Y_test)
model.evaluate(X_test, Y_test)

Prediction

idx = random.randint(0, len(X_test))
img = X_test[idx]
y_pred = model.predict(img.reshape(1, 100, 100, 3))

Display
plt.imshow(img)
