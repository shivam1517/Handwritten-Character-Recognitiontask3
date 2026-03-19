import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# -------------- Create artificial dataset 0-15 ----------------
num_classes = 16
num_samples = 1000

# Random "images" 28x28
x_data = np.random.rand(num_samples, 28, 28).astype('float32')

# Random labels 0-15
y_data = np.random.randint(0, num_classes, num_samples)

# Split into train and test
split = int(0.8 * num_samples)
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

# ---------------- Build Model ----------------
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # 16 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- Train Model ----------------
model.fit(x_train, y_train, epochs=5, verbose=2)

# ---------------- Evaluate ----------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# ---------------- Predict Example ----------------
prediction = model.predict(x_test)
predicted_class = prediction[0].argmax()
print("Predicted Class:", predicted_class)
print("Actual Class:", y_test[0])

# Show image
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {predicted_class}, Actual: {y_test[0]}")
plt.show()
