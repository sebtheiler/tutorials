from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import save_model
import tensorflow as tf

import tempfile

# Authenticate your Google account (this will open a window in a web brower)
from pydrive.auth import GoogleAuth

gauth = GoogleAuth()
gauth.LocalWebserverAuth()

# Create GoogleDrive instance
from pydrive.drive import GoogleDrive
drive = GoogleDrive(gauth)


# Callback class for saving to GDrive
class GoogleDriveSaver(Callback):
    def __init__(self, folder_name, frequency=1):
        super().__init__()
        self.frequency = frequency

        # Search for folder to save in
        file_list = drive.ListFile({'q': f"title='{folder_name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"}).GetList()
        if len(file_list) > 1:
            raise ValueError('There are multiple folders with that specified folder name')
        elif len(file_list) == 0:
            raise ValueError('No folders match that specified folder name')

        # Save the folder's ID
        self.folder_id = file_list[0]['id']

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            # Unfortunately we can't get the raw save file output of model.save, so we need
            # to save it to a tempfile and store that tempfile in Google Drive
            temp_save = tempfile.NamedTemporaryFile(suffix='.hdf5')
            self.model.save(temp_save.name)

            file = drive.CreateFile({'title': f'model-save_epoch-{epoch}.hdf5', 'parents': [{'id': self.folder_id}]})
            file.SetContentFile(temp_save.name)
            file.Upload()

            temp_save.close()



google_drive_saver = GoogleDriveSaver('test-saves')


### Regular setup for MNIST ###
# Adapted from: https://keras.io/examples/mnist_cnn/
batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[google_drive_saver]) # callback for saving in Google Drive
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
