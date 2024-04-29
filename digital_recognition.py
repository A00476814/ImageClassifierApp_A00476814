import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to [0, 1] range
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape the images to (28, 28, 1) because the CNN expects 3D images
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

model.save('my_mnist_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Predict and visualize test images
def visualize_predictions(images, labels, num_samples=10):
    indices = np.random.choice(np.arange(len(images)), num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        prediction = np.argmax(model.predict(np.expand_dims(images[idx], axis=0)))
        ax.set_title(f'Pred: {prediction}\nTrue: {labels[idx]}')
        ax.axis('off')

# Call the function to visualize predictions
visualize_predictions(test_images, test_labels)
