import pandas as pd
import tensorflow as tf
import skimage
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir("..")
x_test = np.array(np.load("./Datos/DatosTestConv.npy"))
x_train = np.array(np.load('./Datos/DatosTrainCONV.npy'))


j = 0
k = -1
y_train = np.array([])
for i in range(0,195768): #cambiar

    if(j % 6992 == 0):
        k += 1

    j+=1
    y_train = np.append(y_train,k)

j = 0
k = -1
y_test = np.array([])
for i in range(0,1409): #cambiar

    if(j % 51 == 0):
        k += 1

    j+=1
    y_test = np.append(y_test,k)

# definir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(300, 3, padding='same', activation='relu', input_shape=(70,70,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(100, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(100, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(28, activation='softmax')
])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(195768,reshuffle_each_iteration=True).batch(32)
#train_dataset = train_dataset.shuffle(195768,reshuffle_each_iteration=True).batch(100)

# compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# entrenar el modelo
history = model.fit(train_dataset, epochs=5, use_multiprocessing=True, validation_data=(x_test,y_test)) #, validation_data=test


# guardar el modelo entrenado
model.save('./ModeloTrain.keras')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()