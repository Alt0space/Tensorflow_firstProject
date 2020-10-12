import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

names = ['Футболка', 'Штаны', 'Пуловер', 'Платье', 'Плащ', 'Сандалии', 'Рубашка',
         'Кросовок', 'Сумка', 'Ботинок']

train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)


prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Настоящая картинка: " + names[test_labels[i]])
    plt.title("Предсказание: " + names[np.argmax(prediction[i])])
    plt.show()


#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print('Тестовая точность:', test_acc)
