import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Charger MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0
x_test = x_test.reshape(-1,28,28,1)/255.0

# Créer un modèle simple CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner
print("Entraînement du modèle...")
model.fit(x_train, y_train, epochs=3, batch_size=64)

# Évaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Précision sur test : {accuracy:.2f}")

# Sauvegarder
model.save("models/mnist_cnn.h5")
print("Modèle sauvegardé dans models/mnist_cnn.h5")
