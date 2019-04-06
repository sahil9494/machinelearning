import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, LeakyReLU, MaxPooling2D
from tensorflow.python.keras.models import Sequential

plt.rcParams['figure.figsize'] = (16.0, 4.0)


def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']


X_train, y_train = load_data('train_32x32.mat')
X_test, y_test = load_data('test_32x32.mat')

print("Training Set", X_train.shape, y_train.shape)
print("Test Set", X_test.shape, y_test.shape)

# Transpose the image arrays
X_train, y_train = X_train.transpose((3, 0, 1, 2)), y_train[:, 0]
X_test, y_test = X_test.transpose((3, 0, 1, 2)), y_test[:, 0]

print("Training Set", X_train.shape)
print("Test Set", X_test.shape)
print('')

class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9', '10']

# Calculate the total number of images
num_images = X_train.shape[0] + X_test.shape[0]

print("Total Number of Images", num_images)


def plot_images(img, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i, :, :, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(labels[i])
    plt.show()


# Plot some training set images
plot_images(X_train, y_train, 2, 8)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)

ax1.hist(y_train, bins=10)
ax1.set_title("Training set")
ax1.set_xlim(1, 10)

ax2.hist(y_test, color='g', bins=10)
ax2.set_title("Test set")

fig.tight_layout()
plt.show()

X_train = X_train / 255.0

X_test = X_test / 255.0


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(32, 32, 3), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(11, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print("model summary", model.summary())

model.fit(X_train, y_train, epochs=5)

print("Model fitting completed, feeding test data for predictions")

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(X_test)

print("predictions are made")

model.save('a4-model.h5')

print("saving the model")


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(11), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# # Plot the first X test images, their predicted label, and the true label
# # Color correct predictions in blue, incorrect predictions in red
num_rows = 2
num_cols = 4
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, y_test, X_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, y_test)
plt.show()