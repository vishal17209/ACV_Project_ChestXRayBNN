# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# !pip install larq larq_zoo
# !pip install imgaug

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# !pip install utils
from utils import *
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import chain
from datetime import datetime
import statistics
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras.backend as kb
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import shutil
import warnings
import json
import pickle

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from sklearn.metrics import roc_curve, auc, roc_auc_score
import tensorflow.keras as keras 
import larq
from larq_zoo.sota import QuickNetLarge
from larq_zoo.literature import BinaryDenseNet45, BinaryDenseNet37Dilated
from larq.layers import QuantDense, QuantConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Conv2D, Activation, GlobalAveragePooling2D, UpSampling2D, GlobalMaxPooling2D, Reshape, GlobalAveragePooling1D
from tensorflow.keras.activations import relu
from tensorflow.linalg import matmul
import tensorflow as tf

import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import Sequence
from PIL import Image
from skimage.transform import resize


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

print('Import Complete')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
# For Data Sequencce

# Disease Names / Class Labels 
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

folders = glob("../data/*")
folders = [i for i in folders if "images_" in i]

images = []
for f in folders:
    image_names = glob(os.path.join(f, "images/*"))
    images += image_names

print(len(images))

# %% [code]
# For Data Sequencce

# %% [code]
# For Data Sequencce

train_map = {}
val_map = {}

output_dir = "../data/"


train_csv = pd.read_csv(os.path.join(output_dir, "train.csv"))
dev_csv = pd.read_csv(os.path.join(output_dir, "dev.csv"))

# for name in tqdm(train_csv['Image Index']):
#     for i in images:
#         if(name in i):
#             train_map[name] = i
#             break

# for name in tqdm(dev_csv['Image Index']):
#     for i in images:
#         if(name in i):
#             val_map[name] = i
#             break

# %% [code]
# For Data Sequencce


class AugmentedImageSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, class_names, source_image_dir, batch_size=16,
                 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=1):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param class_names: list of str
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_image(self, image_file):
        image = Image.open(image_file)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size, :]

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.x_path, self.y = df["Image Index"].to_numpy(), df[self.class_names].to_numpy()
        for i in range(len(self.x_path)):
            self.x_path[i] = os.path.join(self.source_image_dir, self.x_path[i])

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

print("** create image generators **")

output_dir = "../data/"
class_names = disease_labels
image_source_dir = "../data/"
batch_size = 4
image_dimension = 224

from imgaug import augmenters as iaa

augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
    ],
    random_order=True,
)

train_steps = "auto"
train_count = pd.read_csv(os.path.join(output_dir, "train.csv")).shape[0]/batch_size
dev_count = pd.read_csv(os.path.join(output_dir, "dev.csv")).shape[0]/batch_size

train_sequence = AugmentedImageSequence(
    dataset_csv_file=os.path.join(output_dir, "train.csv"),
    class_names=class_names,
    source_image_dir=image_source_dir,
    batch_size=batch_size,
    target_size=(image_dimension, image_dimension),
    augmenter=augmenter,
    steps=train_count,
)

validation_sequence = AugmentedImageSequence(
    dataset_csv_file=os.path.join(output_dir, "dev.csv"),
    class_names=class_names,
    source_image_dir=image_source_dir,
    batch_size=batch_size,
    target_size=(image_dimension, image_dimension),
    augmenter=augmenter,
    steps=dev_count,
    shuffle_on_epoch_end=False,
)

# %% [code]

# %% [code]
print("TensorFlow version: ", tf.__version__)


# %% [code]
IMG_SHAPE = (224,224,3)

EPOCHS = 5

kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1,mode='min',cooldown=0,min_lr=1e-8)

checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

OPTIMIZER = Adam(learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999)

LOSS = BinaryCrossentropy()

METRICS = ['BinaryAccuracy']

# %% [code]
def FPAModule(x, y, z, num_channels):
    inp = Input(shape=(x,y,z))
    height, width = inp.shape[1], inp.shape[2]

    gap_branch = GlobalAveragePooling2D()(inp)
    gap_branch = Reshape((1, 1, gap_branch.shape[1]))(gap_branch)
    gap_branch = Conv2D(num_channels, kernel_size=1, strides=1)(gap_branch)
    gap_branch = BatchNormalization()(gap_branch)
    gap_branch = Activation(relu)(gap_branch)
    gap_branch = UpSampling2D(size=(height, width), interpolation='bilinear')(gap_branch)

    mid_branch = Conv2D(num_channels, kernel_size=1, strides=1)(inp)
    mid_branch = BatchNormalization()(mid_branch)
    mid_branch = Activation(relu)(mid_branch)

    #Downsample layers
    scale1 = Conv2D(1, kernel_size=7, strides=2, padding="same")(inp)
    scale1 = BatchNormalization()(scale1)
    scale1 = Activation(relu)(scale1)

    scale2 = Conv2D(1, kernel_size=5, strides=2, padding="same")(scale1)
    scale2 = BatchNormalization()(scale2)
    scale2 = Activation(relu)(scale2)

    scale3 = Conv2D(1, kernel_size=3, strides=2, padding="same")(scale2)
    scale3 = BatchNormalization()(scale3)
    scale3 = Activation(relu)(scale3)

    #Scale layers
    scale3 = Conv2D(1, kernel_size=3, padding="same")(scale3)
    scale3 = BatchNormalization()(scale3)
    scale3 = Activation(relu)(scale3)
    scale3 = UpSampling2D(size=(height//(4*scale3.shape[1]), width//(4*scale3.shape[2])), interpolation='bilinear')(scale3)

    scale2 = Conv2D(1, kernel_size=5, padding="same")(scale2)
    scale2 = BatchNormalization()(scale2)
    scale2 = Activation(relu)(scale2)
    scale2 = scale2 + scale3
    scale2 = UpSampling2D(size=(height//(2*scale2.shape[1]), width//(2*scale2.shape[2])), interpolation='bilinear')(scale2)

    scale1 = Conv2D(1, kernel_size=7, padding="same")(scale1)
    scale1 = BatchNormalization()(scale1)
    scale1 = Activation(relu)(scale1)
    scale1 = scale1 + scale2
    scale1 = UpSampling2D(size=(height//scale1.shape[1], width//scale1.shape[2]), interpolation='bilinear')(scale1)

    z = tf.math.multiply(scale1, mid_branch) + gap_branch

    out = Model(inputs=inp, outputs=z)

    return out

# %% [code]
# add this code ROC

class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """
    def __init__(self, generator, class_names, weights_path, stats=None):
        super(Callback, self).__init__()
        self.generator = generator
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            f"best_{os.path.split(weights_path)[1]}",
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calcula el promedio de las Curvas ROC y guarda el mejor grupo de pesos
        de acuerdo a esta metrica
        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print(f"Learning Rate actual: {self.stats['lr']}")

        """
        y_hat shape: (#ejemplos, len(etiquetas))
        y: [(#ejemplos, 1), (#ejemplos, 1) ... (#ejemplos, 1)]
        """
        y_hat = self.model.predict_generator(self.generator,steps=self.generator.n/self.generator.batch_size)
        y = self.generator.labels

        print(f"*** epoch#{epoch + 1} Curvas ROC Fase Entrenamiento ***")
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:][i], y_hat[:][i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print(f"{i+1}. {self.class_names[i]}: {score}")
        print("*********************************")

        mean_auroc = np.mean(current_auroc)
        print(f"Promedio Curvas ROC: {mean_auroc}")
        if mean_auroc > self.stats["best_mean_auroc"]:
            print(f"Actualización del resultado de las Curvas de ROC de: {self.stats['best_mean_auroc']} a {mean_auroc}")

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print(f"Actualización del archivo de logs: {self.best_auroc_log_path}")
            with open(self.best_auroc_log_path, "a") as f:
                f.write(f"(epoch#{epoch + 1}) auroc: {mean_auroc}, lr: {self.stats['lr']}\n")

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print(f"Actualización del grupo de pesos: {self.weights_path} -> {self.best_weights_path}")
            self.stats["best_mean_auroc"] = mean_auroc
            print("*********************************")
        return

# %% [code]
# add this code for ROC
training_stats = {}
auroc = MultipleClassAUROC(
    generator=validation_sequence,
    class_names=disease_labels,
    weights_path=checkpoint_path,
    stats=training_stats
)

# %% [code]
# print("Num GPUs Used: ", len(tf.config.experimental.list_physical_devices('GPU')))


base_model = BinaryDenseNet37Dilated(input_shape=(224, 224, 3), include_top=False)

# %% [code]
base_model.summary()

# %% [code]
x = base_model.output
x = Reshape((16, 16, 1960))(x)

logits = []
classifier = {}

for i in range(14):
    classifier['fc_'+str(i)] = Conv2D(1, kernel_size=1, strides=1, padding="valid", activation="sigmoid")

for i in range(14):
    x = FPAModule(x.shape[1], x.shape[2], x.shape[3], 1960)(x)
    
#     print(x.shape)
    
    a = GlobalAveragePooling2D()(x)
    b = GlobalMaxPooling2D()(x)
    
    a = Reshape((1, 1, a.shape[1]))(a)
    b = Reshape((1, 1, b.shape[1]))(b)
    
    feat = tf.concat((a,b), axis=3)
    
    feat = BatchNormalization(axis=-1)(feat)
    
    logit = classifier["fc_{}".format(i)](feat)
    
    logit = Flatten()(logit)
    
#     print(logit.shape)
    
    logits.append(logit)

model = Model(inputs=base_model.input, outputs=logits)


model.compile(loss = ['binary_crossentropy']*14,
          optimizer=OPTIMIZER,
          metrics=METRICS
             )

# %% [code]

model.load_weights("full_precision_model.h5")
for i in range(EPOCHS):

    print("EPOCH: {}/{}".format(i+1, EPOCHS))
    print("-"*20)

    history = model.fit(train_sequence,
                        validation_data = validation_sequence,     
                        epochs = 1,
                        shuffle = True,
                        callbacks = [reduce_lr], # add code here
                        verbose = 1
    )

    model.save("full_precision_model.h5")  # save full precision latent weights

    with larq.context.quantized_scope(True):
        model.save("binary_model.h5")  # save binary weights

# %% [code]

import pickle

file = open('dumpfile.txt', 'wb+')
pickle.dump(history.history, file)
file.close()

# acc = history.history['BinaryAccuracy']
# val_acc = history.history['val_BinaryAccuracy']

# loss=history.history['loss']
# val_loss=history.history['val_loss']

# epochs_range = range(EPOCHS)



# # %% [code]
# plt.figure(figsize=(40, 10))
# plt.subplot(1, 2, 1)
# plt.grid()
# plt.plot(epochs_range, acc, label='Training Binary Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Binary Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Binary Accuracy', color='Green')
# plt.savefig('acc_plot.png')

# # %% [code]
# plt.figure(figsize=(40, 10))
# plt.subplot(1, 2, 2)
# plt.grid()
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss', color='red')
# plt.savefig('loss_plot.png')

# plt.show()

# # %% [code]
# larq.models.summary(model, print_fn=None, include_macs=True)

# %% [code]
train_count*batch_size

# %% [code]
dev_count*batch_size

# %% [code]
val_outputs = []
labels = []

for i in tqdm(range(len(validation_sequence))):
    data, label = validation_sequence[i]
    val_outputs.append(model(data))
    labels.append(label)


# %% [code]
# AUC Code

outputs = []
for i in val_outputs:
    outputs.append(np.concatenate(i, axis=1))
    
print(len(outputs), outputs[0].shape)

# %% [code]
all_labels = np.concatenate(labels)
all_outputs = np.concatenate(outputs, axis=0)

# %% [code]
all_labels.shape, all_outputs.shape

# %% [code]
aucs = []

for i in range(14):
    try:
        score = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        print(i, score)
        aucs.append(score)
    except:
        print(i, 0)
        pass

print(np.mean(aucs))

# %% [code]
