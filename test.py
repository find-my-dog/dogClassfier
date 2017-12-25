from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

LABEL_PATH = 'data/labels.csv'
DEST_PATH = 'data/'
SAMPLE_PATH = 'data/sample_submission.csv'


df = pd.read_csv(LABEL_PATH)
df.head()

n = len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))


width = 299
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)
for i in tqdm(range(n)):
    im = Image.open(DEST_PATH+'train/%s.jpg' % df['id'][i])
    X[i] = im.resize((width,width))
    #X[i] = cv2.resize(cv2.imread(DEST_PATH+'train/%s.jpg' % df['id'][i]), (width, width))
    y[i][class_to_num[df['breed'][i]]] = 1


import random
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i in range(8):
    random_index = random.randint(0, n-1)
    plt.subplot(2, 4, i+1)
    plt.imshow(X[random_index][:,:,::-1])
    plt.title(num_to_class[y[random_index].argmax()])
plt.savefig("1.jpg")

from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input

def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')

    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features


inception_features = get_features(InceptionV3, X)
xception_features = get_features(Xception, X)
features = np.concatenate([inception_features, xception_features], axis=-1)

inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(features, y, batch_size=128, epochs=10, validation_split=0.1)

try:
    import cPickle as pickle
except ImportError:  # Python 3
    import pickle
with open('net.pickle', 'wb') as f:
    pickle.dump(model, f, -1)

filename = open('model_loss.csv', 'wb')
np.savetxt(filename, h.history['loss'])
np.savetxt(filename, h.history['val_losss'])
np.savetxt(filename, h.history['acc'])
np.savetxt(filename, h.history['val_acc'])

from IPython.display import SVG
from keras.utils import plot_model
plot_model(model, to_file='model.jpg')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')
plt.savefig('2.jpg')


df2 = pd.read_csv(SAMPLE_PATH)
n_test = len(df2)
X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    im = Image.open((DEST_PATH+'test/%s.jpg' % df2['id'][i]))
    X_test[i] = im.resize((width,width))
    #X_test[i] = cv2.resize(cv2.imread(DEST_PATH+'test/%s.jpg' % df2['id'][i]), (width, width))

inception_features = get_features(InceptionV3, X_test)
xception_features = get_features(Xception, X_test)
features_test = np.concatenate([inception_features, xception_features], axis=-1)


y_pred = model.predict(features_test, batch_size=128)
for b in breed:
    df2[b] = y_pred[:,class_to_num[b]]

df2.to_csv('pred.csv', index=None)