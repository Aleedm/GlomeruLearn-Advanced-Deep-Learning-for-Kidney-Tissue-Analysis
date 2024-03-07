import numpy as np
import tensorflow as tf
import json
from segnet import SegNet  # Import the U-Net model defined in another module

# Define paths for data files and models
image_train_path = '../../dataset512x512/image_train_balanced.npy'
label_train_path = '../../dataset512x512/label_train_balanced.npy'
image_validation_path = '../../dataset512x512/image_validation_balanced.npy'
label_validation_path = '../../dataset512x512/label_validation_balanced.npy'
image_test_path = '../../dataset512x512/image_test_ss.npy'
label_test_path = '../../dataset512x512/label_test_ss.npy'
version = 1.2
info = "_balanced_dataset_"
path_to_save_model = f'../../saved_model/segnet_V{version}_{info}'
path_to_save_model_weights = f'../../saved_model/segnet_V{version}_{info}_weights.h5'
path_to_save_history = f'../../histories/history_Segnet_V{version}_{info}.json'
path_to_save_test_result = f'../../dataset512x512/test_result_segnet_V{version}_{info}.npy'

# Training parameters
epochs = 50
learning_rate = 0.01
batch_size = 8

# Set up the distribution strategy
strategy = tf.distribute.MirroredStrategy()

# Training and evaluating the SegNet model within the distribution strategy scope
with strategy.scope():
    def train_and_evaluate_segnet(image_train=None, label_train=None, validation_image=None, validation_label=None, dataset_train=None, dataset_validation=None, learning_rate=None):
        # Initialize and compile the model
        model = SegNet()
        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate),
                      metrics=["accuracy", "Precision", "Recall", "FalseNegatives",
                               "FalsePositives", "TrueNegatives", "TruePositives"])
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True, start_from_epoch=15)

        # Train the model
        if image_train is not None and label_train is not None:
            history = model.fit(image_train, label_train, validation_data=(validation_image, validation_label),
                                epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
        elif dataset_train is not None and dataset_validation is not None:
            history = model.fit(dataset_train, validation_data=dataset_validation,
                                epochs=epochs, callbacks=[early_stopping])

        best_accuracy = max(history.history['val_accuracy'])
        return model, best_accuracy, history

    # Define data generators
    def data_generator():
        data = np.load(image_train_path)
        labels = np.load(label_train_path)
        for d, l in zip(data, labels):
            yield d, l

    def validation_data_generator():
        validation_data = np.load(image_validation_path)
        validation_labels = np.load(label_validation_path)
        for d, l in zip(validation_data, validation_labels):
            yield d, l

    # Pre-process the data
    def process_data(image, label):
        return tf.cast(image, tf.float32)/255, tf.one_hot(label, 2, name="label", axis=-1)

    # Load the data generators
    def load_generator():
        output_shapes_train = (tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
                               tf.TensorSpec(shape=(512, 512), dtype=tf.int64))
        output_shapes_validation = (tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
                                    tf.TensorSpec(shape=(512, 512), dtype=tf.int64))

        dataset = tf.data.Dataset.from_generator(
            data_generator, output_signature=output_shapes_train)
        dataset = dataset.batch(batch_size, drop_remainder=True).map(
            process_data, num_parallel_calls=tf.data.AUTOTUNE)

        validation_dataset = tf.data.Dataset.from_generator(
            validation_data_generator, output_signature=output_shapes_validation)
        validation_dataset = validation_dataset.batch(batch_size, drop_remainder=True).map(
            process_data, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset, validation_dataset

    # Functions to test the model
    def test_model(model, image_test):
        return model.predict(image_test)

    def load_test_data():
        image_test = np.load(image_test_path)/255
        return image_test

    # Load data, train, save the model and the results
    dataset, validation_dataset = load_generator()
    model, accuracy, history = train_and_evaluate_segnet(dataset_train=dataset, dataset_validation=validation_dataset,
                                                         learning_rate=learning_rate)
    model.save(path_to_save_model, save_format='tf')
    model.save_weights(path_to_save_model_weights)

    with open(path_to_save_history, "w") as fp:
        json.dump(history.history, fp)

    image_test = load_test_data()
    result = test_model(model, image_test)
    np.save(path_to_save_test_result, result)
