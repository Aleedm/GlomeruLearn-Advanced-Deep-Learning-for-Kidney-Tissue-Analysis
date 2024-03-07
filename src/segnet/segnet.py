import tensorflow as tf
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Activation
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

NUM_CLASSES = 2
INPUT_SHAPE = (512, 512, 3)
IMAGE_SIZE = (512, 512)


class SegNet(Model):
    def __init__(self, num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE):
        super().__init__()
        # Load the pre-trained VGG19 model as the base model for SegNet
        vgg19 = tf.keras.applications.vgg19.VGG19(
            include_top=False,   # Exclusion of the last 3 layers
            weights='imagenet',
            input_shape=input_shape,
            pooling='max',
            classes=num_classes,
            classifier_activation='relu'
        )
        # Encoder
        # Block 1
        self.b1 = tf.keras.Sequential([vgg19.get_layer('block1_conv1'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block1_conv2'),
                                      BatchNormalization(),
                                      Activation('relu')])
        self.b1p = MaxPoolingWithArgmax2D(name="layerMP1")
        # Block 2
        self.b2 = tf.keras.Sequential([vgg19.get_layer('block2_conv1'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block2_conv2'),
                                      BatchNormalization(),
                                      Activation('relu')])
        self.b2p = MaxPoolingWithArgmax2D(name="layerMP2")
        # Block 3
        self.b3 = tf.keras.Sequential([vgg19.get_layer('block3_conv1'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block3_conv2'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block3_conv3'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block3_conv4'),
                                      BatchNormalization(),
                                      Activation('relu')])
        self.b3p = MaxPoolingWithArgmax2D(name="layerMP3")
        # Block 4
        self.b4 = tf.keras.Sequential([vgg19.get_layer('block4_conv1'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block4_conv2'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block4_conv3'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block4_conv4'),
                                      BatchNormalization(),
                                      Activation('relu')])
        self.b4p = MaxPoolingWithArgmax2D(name="layerMP4")
        # Block 5
        self.b5 = tf.keras.Sequential([vgg19.get_layer('block5_conv1'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block5_conv2'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block5_conv3'),
                                      BatchNormalization(),
                                      Activation('relu'),
                                      vgg19.get_layer('block5_conv4'),
                                      BatchNormalization(),
                                      Activation('relu')])
        self.b5p = MaxPoolingWithArgmax2D(name="layerMP5")

        # Decoder
        # Block 6
        self.b6p = MaxUnpooling2D()
        self.b6 = tf.keras.Sequential([Conv2D(filters=512, activation='relu', kernel_size=(3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=512, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=512, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=512, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu')])
        # Block 7
        self.b7p = MaxUnpooling2D()
        self.b7 = tf.keras.Sequential([Conv2D(filters=512, activation='relu', kernel_size=(3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=512, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=512, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=256, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu')])
        # Block 8
        self.b8p = MaxUnpooling2D()
        self.b8 = tf.keras.Sequential([Conv2D(filters=256, activation='relu', kernel_size=(3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=256, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=256, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=128, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu')])
        # Block 9
        self.b9p = MaxUnpooling2D()
        self.b9 = tf.keras.Sequential([Conv2D(filters=128, activation='relu', kernel_size=(3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu'),
                                       Conv2D(filters=64, activation='relu', kernel_size=(
                                           3, 3), kernel_initializer='he_normal', padding='same'),
                                       BatchNormalization(),
                                       Activation('relu')])
        # Block 10
        self.b10p = MaxUnpooling2D()
        self.b10 = tf.keras.Sequential([Conv2D(filters=64, activation='relu', kernel_size=(3, 3), kernel_initializer='he_normal', padding='same'),
                                        BatchNormalization(),
                                        Activation('relu'),
                                        Conv2D(filters=64, activation='relu', kernel_size=(
                                            3, 3), kernel_initializer='he_normal', padding='same'),
                                        BatchNormalization(),
                                        Activation('relu')])
        self.o = Conv2D(filters=NUM_CLASSES, kernel_size=(
            1, 1), padding='valid', activation='softmax')

    def call(self, input):
        # Encoder
        # Block 1
        x = self.b1(input)
        x, index1 = self.b1p(x)
        # Block 2
        x = self.b2(x)
        x, index2 = self.b2p(x)
        # Block 3
        x = self.b3(x)
        x, index3 = self.b3p(x)
        # Block 4
        x = self.b4(x)
        x, index4 = self.b4p(x)
        # Block 5
        x = self.b5(x)
        x, index5 = self.b5p(x)
        # Decoder
        # Block 6
        x = self.b6p([x, index5])
        x = self.b6(x)
        # Block 7
        x = self.b7p([x, index4])
        x = self.b7(x)
        # Block 8
        x = self.b8p([x, index3])
        x = self.b8(x)
        # Block 9
        x = self.b9p([x, index2])
        x = self.b9(x)
        # Block 10
        x = self.b10p([x, index1])
        x = self.b10(x)
        o = self.o(x)
        return o
