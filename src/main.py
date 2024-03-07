import math
import numpy as np
from sklearn.model_selection import train_test_split
from data_prepr_aug import generate_data_tensor
from Segnet.segnet import SegNet
import tensorflow as tf
from Unet.unet import Unet
import json

#import os
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#    except RuntimeError as e:
#        print(e)

strategy = tf.distribute.MirroredStrategy()

train_data_batch_number = 20

with strategy.scope():

    image_train=np.load('../slides/image_train_augmented.npy')
    #print(f"image_train shape: {image_train.shape}")
    label_train=np.load('../annotations/label_train_augmented.npy')
    #print(f"label_train shape: {label_train.shape}")
    image_validation=np.load('../slides/image_validation.npy')
    label_validation=np.load('../annotations/label_validation.npy')
    image_test=np.load('../slides/image_test.npy')
    label_test=np.load('../annotations/label_test.npy')

    validation_data = generate_data_tensor(image_validation, label_validation)
    test_data = generate_data_tensor(image_test, image_test, train=False)

    # Definisci la funzione per addestrare e valutare un modello con un dato tasso di apprendimento
    def train_and_evaluate_segnet(image_train, label_train, validation_data, learning_rate):
        # Costruisci il modello
        model = SegNet()
        # Compila il modello con la funzione di perdita e l'ottimizzatore appropriati
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy", "Precision", "Recall", "FalseNegatives",
                                                                                                                "FalsePositives", "TrueNegatives", "TruePositives"])
        #print(f"params: {model.count_params()}")
        #print("testing a model")
        # Addestra il modello sul training set
        image_train_size = len(image_train) // train_data_batch_number
        #print(f"image_train_size: {image_train}")
        label_train_size = len(label_train) // train_data_batch_number
        #print(f"label_train_size: {image_train}")
        
        for i in range(train_data_batch_number):
            train_data = [];
            if i == train_data_batch_number-1:
                train_data=generate_data_tensor(image_train=image_train[i*image_train_size:], label_train=label_train[i*label_train_size:])
            else:
                train_data=generate_data_tensor(image_train=image_train[i*image_train_size: (i+1)*image_train_size], label_train=label_train[i*label_train_size:(i+1)*label_train_size])
            
            #print(f"train_data: {train_data}")
            steps_per_epoch = math.ceil(image_train_size / 64)
            #print(f"steps_per_epoch: {steps_per_epoch}")
            model.fit(train_data, epochs=50, steps_per_epoch=steps_per_epoch)
            
    
        # Valuta il modello sul validation set
        evals = model.evaluate(validation_data)
        return model, evals[1]


    def train_and_evaluate_unet(image_train, label_train, validation_data, learning_rate):
        # Costruisci il modello
        model = Unet()
        # Compila il modello con la funzione di perdita e l'ottimizzatore appropriati
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy", "Precision", "Recall", "FalseNegatives",
                                                                                                                "FalsePositives", "TrueNegatives", "TruePositives"])
        print(f"params: {model.count_params()}")
        print("testing a model")
        
        print("testing a model")
        # Addestra il modello sul training set
        image_train_size = len(image_train) // train_data_batch_number
        print(f"image_train_size: {image_train}")
        label_train_size = len(label_train) // train_data_batch_number
        print(f"label_train_size: {image_train}")
        
        for i in range(train_data_batch_number):
            train_data = [];
            if i == train_data_batch_number-1:
                train_data=generate_data_tensor(image_train=image_train[i*image_train_size:], label_train=label_train[i*label_train_size:])
            else:
                train_data=generate_data_tensor(image_train=image_train[i*image_train_size: (i+1)*image_train_size], label_train=label_train[i*label_train_size:(i+1)*label_train_size])
            print(f"train_data: {train_data}")
            steps_per_epoch = math.ceil(image_train_size / 64)
            print(f"steps_per_epoch: {steps_per_epoch}")
            model.fit(train_data, epochs=50, steps_per_epoch=steps_per_epoch)
        # Valuta il modello sul validation set
        evals = model.evaluate(validation_data)
        return model, evals[1]


    best_accuracy = 0.0
    best_learning_rate = None
    best_models = None
    best_history = None

    learning_rates = [0.001, 0.01, 0.1]


    # Valuta ogni tasso di apprendimento e seleziona il migliore
    for learning_rate in learning_rates:
       print("creating a model")
       model, accuracy = train_and_evaluate_segnet(
           image_train, label_train, validation_data, learning_rate)
       print(f"Learning Rate: {learning_rate}, Accuracy: {accuracy}")
       if accuracy > best_accuracy:
           best_accuracy = accuracy
           best_learning_rate = learning_rate
           best_model = model

    best_models.save('../saved_model/segnet')
    with open("../shistory/acc_segnet.txt", "x") as fp:
       json.dump(best_history.history, fp)


#    best_accuracy = 0.0
#    best_learning_rate = None
#    best_models = None
#    best_history = None
#
#    for learning_rate in learning_rates:
#        print("creating a model")
#        model, accuracy = train_and_evaluate_unet(
#            image_train, label_train, validation_data, learning_rate)
#        print(f"Learning Rate: {learning_rate}, Accuracy: {accuracy}")
#        if accuracy > best_accuracy:
#            best_accuracy = accuracy
#            best_learning_rate = learning_rate
#            best_model = model
#
#    best_models.save('../saved_model/unet')
    #with open("../uhistory/history_unet.txt", "x") as fp:
    #    json.dump(best_history.history, fp)

