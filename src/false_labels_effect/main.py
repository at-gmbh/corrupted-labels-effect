# imports
from datetime import datetime
import json
import numpy as np
import os
from pathlib import Path
import sys
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import TensorBoard

# silence tensorflow deprecation warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# add src to sys.path and import local modules
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import callbacks as cbs 
import data_loader as dl
import models as mdls
import util

#------------------------------------------------------------------------------#

# --> TODO: set parameter below <--

# Select model task
#   'Class': main class (4) classification
#   'Subclass': sub class (14) classification
#   'Annotations' : polygon vertices prediction
model_task = 'Class'

# set number of classes in labels
if model_task.lower() == 'class':
    n_classes = 4
elif model_task.lower() == 'subclass':
    n_classes = 14

# set number of images
limit_loaded_images = 300  # use None for "all" images

# set target size of images
resize_to = (244, 244) 

# set list of ratio of false labels in training data between 0 and 1 (low to high)
# 0.05 => 5% of images in training data will be labeled incorrectly
false_ratios = [0.05]

# define data loader parameters
val_split = 0.2
batch_size = 32

# define model processing parameter
n_epochs = 2
multiprocessing = False
n_workers = 1

#------------------------------------------------------------------------------#

# set training and test img png path
train_img_png_path = Path('./data/Images_4c_Poly/Train')
test_img_png_path = Path('./data/Images_4c_Poly/Test')

# set training and test img npy path
train_img_npy_path = Path('./data/Images_4c_Poly/Train_npy')
test_img_npy_path = Path('./data/Images_4c_Poly/Test_npy')

# set label path
train_label_path = Path('./data/Labels_4c_Poly')
test_label_path = Path('./data/Labels_4c_Poly')

#------------------------------------------------------------------------------#

print('Loading labels...')
# load labels
train_labels_dict = util.load_labels(f'{train_label_path}/Train.npy')
test_labels_dict = util.load_labels(f'{test_label_path}/Test.npy')

#------------------------------------------------------------------------------#

print('Reseizing images...')
# load train images, resize and save as npy
if not os.path.exists(f'{train_img_npy_path}'):
    os.mkdir(train_img_npy_path)

    i = 0
    for image_path in train_img_png_path.iterdir():
        i += 1
        if limit_loaded_images is not None and i > limit_loaded_images:
            break

        # Load without resizing so that polygon fits (for now)
        img_id = image_path.name.split(".")[0]
        img = load_img(image_path)

        # Use util resize function resize image and polygon
        # TODO: poly resize currently not saved
        img_res, poly_res = util.resize(
            img, train_labels_dict[img_id], resize_to
        )

        npy_img = img_to_array(img_res)
        np.save(f'{train_img_npy_path}/{img_id}', npy_img)

# load test images, resize and save as npy
if not os.path.exists(f'{test_img_npy_path}'):
    os.mkdir(test_img_npy_path)
    i = 0

    for image_path in test_img_png_path.iterdir():
        i += 1
        if limit_loaded_images is not None and i > limit_loaded_images:
            break

        # Load without resizing so that polygon fits (for now)
        img_id = image_path.name.split(".")[0]
        img = load_img(image_path)

        # Use util resize function resize image and polygon
        # TODO: poly resize currently not saved
        img_res, poly_res = util.resize(
            img, test_labels_dict[img_id], resize_to
        )

        npy_img = img_to_array(img_res)
        np.save(f'{test_img_npy_path}/{img_id}', npy_img)

#------------------------------------------------------------------------------#

print('Filter images...')
# create dict of included train and test images, format for keras data loader
partition = {}
train_img_ids_included = [str(i.name).split(".")[0] for i in train_img_npy_path.iterdir()]
test_img_ids_included = [str(i.name).split(".")[0] for i in test_img_npy_path.iterdir()]

# split for test and train
partition['train'] = [id for id in train_img_ids_included if 'Train' in id]
partition['test'] = [id for id in test_img_ids_included if 'Test' in id]

#------------------------------------------------------------------------------#

print('Filter labels...')
# filter train labels to only include transformed images
train_labels_dict_incl = {}
for (key, value) in train_labels_dict.items():
    if key in partition['train']:
        train_labels_dict_incl[key] = value

# filter test labels to only include transformed images
test_labels_dict_incl = {}
for (key, value) in test_labels_dict.items():
    if key in partition['test']:
        test_labels_dict_incl[key] = value

# generate flattened dict of model task corresponding labels
train_labels_dict_flat = util.select_label(train_labels_dict_incl, model_task)
test_labels_dict_flat = util.select_label(test_labels_dict_incl, model_task)

# encode categorical labels for classification tasks
if model_task in ['Class', 'Subclass']:
    label_mapping, y_train, y_test, = util.encode_labels(train_labels_dict_flat, test_labels_dict_flat)
    
    label_mapping = {int(k):str(v) for k, v in label_mapping.items()}
    print('    Labels:')
    for k, v in label_mapping.items():
        print(f'        {k}: {v}')

#------------------------------------------------------------------------------#

print('Partition labels...')
# split training data into train and validation
partition['train'], y_train, partition['val'], y_val = util.train_val_split(partition['train'], y_train, val_split)
print('    # train imgs:', len(partition['train']), '- # val imgs:', len(partition['val']), '- # test imgs:', len(partition['test']))

#------------------------------------------------------------------------------#

print('Create false labels...')
# create false train labels for all given ratios
util.make_false_labels(train_label_path, y_train, false_ratios, n_classes)

#------------------------------------------------------------------------------#

for ratio in false_ratios:
    # load false train labels
    y_train = util.load_false_labels(train_label_path, ratio)

    # define data loader parameters
    params = {'dim': (resize_to[0],resize_to[1]),
              'batch_size': batch_size,
              'n_classes': n_classes,
              'n_channels': 3,
              'shuffle': True}

    # load data
    training_loader = dl.DataLoader(partition['train'], y_train, **params)
    validation_loader = dl.DataLoader(partition['val'], y_val, **params)
    test_loader = dl.DataLoader(partition['test'], y_test, **params)

    # load models
    basic_cnn = mdls.create_cnn_model(resize_to, n_classes, ratio)
    resnet_cnn = mdls.create_resnet_model(resize_to, n_classes, ratio)
    all_models = [basic_cnn, resnet_cnn] # TODO: set models to be included

    for model in all_models:
        print(model.summary())

        model_start_time = datetime.now().strftime("%Y%m%d-%H%M")

        # initialize logging
        logdir_scalars = f'../logs/scalars/{model._name}/{model_start_time}'
        logdir_label_mapper = f'../logs/label_mapping/{model._name}/{model_start_time}'
        
        os.makedirs(logdir_label_mapper)
        with open(logdir_label_mapper + '/label_mapper.json', 'w+') as f:
            json.dump(label_mapping, f)

        tensorboard_callback = TensorBoard(log_dir=logdir_scalars)
        classReport_callback = cbs.class_report_cb(test_loader, model_start_time)

        file_writer = tf.summary.create_file_writer(logdir_scalars)

        # compile model
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=['accuracy'])

        # train model with tensorboard and classification report logging
        history = model.fit(x = training_loader,
                                epochs = n_epochs,
                                verbose = 1,
                                callbacks=[classReport_callback,
                                           tensorboard_callback],
                                validation_data = validation_loader,
                                use_multiprocessing = multiprocessing,
                                workers = n_workers)
        
        # show test accuracy
        score = model.evaluate(x = test_loader,
                               callbacks=[tensorboard_callback],
                               use_multiprocessing = multiprocessing,
                               workers = n_workers,
                               verbose = 0)

        print(model._name, '- Test accuracy:', score[1],
              f'\n{"=" * 65}\n')

        model.save(f'../logs/models/{model._name}/{model_start_time}')

#------------------------------------------------------------------------------#

# clean up false labels files
for ratio in false_ratios:
    os.remove(f'{train_label_path}/Train_{format(int(ratio*10000),"05d")}r.npy')
