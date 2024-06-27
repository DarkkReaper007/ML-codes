import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(trainimages, trainlabels), (testimages, testlabels) = datasets.mnist.load_data()

trainimages = trainimages.reshape((60000,28, 28, 1)).astype('float32') / 255
testimages = testimages.reshape((10000, 28, 28, 1)).astype('float32') / 255

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.75))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainimages, trainlabels, epochs=20, batch_size=128, validation_split=0.1)

test_loss, test_acc = model.evaluate(testimages, testlabels)
print(f"Accuracy: {test_acc * 100:.2f}")
