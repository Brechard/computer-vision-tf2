# Object Recognition/Detection

Implementation of models for object detection and object recognition using TensorFlow 2.0.

For object detection, YOLOv3 has been implemented, and for object recognition severl models
are available. Take into consideration that they are intented for recognition of traffic signs
and therefore they models are very small. 


Thanks to: https://github.com/zzh8829/yolov3-tf2 and the references on it.
I've taken several of his functions and used it as a base to check for errors.

# Usage
When training a model, a folder will be created with the date that started and the dataset used to trained.
Inside this folder, there will be again three folders:
- checkpoints: saved checkpoints, final model and a file with the training parameters used.
- figures: accuracy, loss and learning rate through the training.
- logs: created for the log of TensorBoard.
It can be found in models/<model_name>/

## Install the requirements
    pip install -r requirements.txt

## Object Detection
### Create the TFRecords files
Download the dataset wanted and save everything it in the data/external/datasets/<dataset_name> folder.

The supported datasets are: COCO, GTSD, BDD100K and MAPILLARY Traffic Signs. If you want to use a different one,
you have to create a method in the src/data/external_to_raw.py folder. You can find more info in that file.

    cd src/
    python -m data.make_dataset --dataset_name <dataset_name>

### Train YOLOv3 from scratch
Optional parameters (default values inside the parentheses) are:
- epochs (100): number of epochs to train.
- save_freq (5): while training, the model will be saved every 'save_freq' epochs.
- batch_size (32): batch size.
- lr (2e-3): initial learning rate.
- use_cosine_lr (True): flag to use cosine decay scheduler for the learning rate.
- tiny (False): flag to use the tiny version of the model or the full version.
- model_name ('YOLOv3'): If you want to give the model another name. Used for saving the model training history.
- extra (''): any extra information that you want to be saved in the file with the training parameters.

For the parameters that are booleans, if you want them to be True use --parameter if False --noparameter

    python -m models.detection.train_model --dataset_name <dataset_name>

### Fine tune a model with the original weights
Download the weights from the original yolov3 model published in https://pjreddie.com/media/files/yolov3.weights
and copy them to models/YOLOv3/
   
    wget https://pjreddie.com/media/files/yolov3.weights -O models/YOLOv3/yolov3.weights
    wget https://pjreddie.com/media/files/yolov3-tiny.weights -O models/YOLOv3/yolov3-tiny.weights

The same optional parameters as before apply.
There are 4 ways to fine tune the model:
- 'all': All the weights of the model will be updated during the training.
- 'features': Freezes the features extractor (DarkNet) and the rest of the model can be trained.
- 'last_block': Only the last block of convolutional layers are trainable.
- 'last_conv': Only the last convolutional layer is trainable.


    python -m models.detection.train_model --trainable <trainable option> --dataset_name <dataset_name>


### Predict
Optional parameters are:

- weights_path: path to the weights to load, if not added it will load the weights from the original yolo paper.
- dataset_name: name of the dataset to use the labels from. Translating from output of the model to a label.
- output_path: path to save the output image of the prediction
- title: title to add to the output image.
- tiny: by default it uses the full model (use --tiny for the tiny version or --notiny to force the full model).


    python -m models.detection.predict --img_path <img_path>

## Object Detection
Keep in mind that this model was designed to work with the German Traffic Signs Dataset, 
where using images of 50x50 pixels is enough. Therefore the model is pretty small.

### Train the model
To train the model, the images should be in data/external/<dataset_name> and there should be a 'train' and 'test',
inside the train folder for each class there must be again a folder containing the images that belong to that class.
For testing there has to be a Test.csv file in data/external/<dataset_name>/Test.csv with at least the columns 
Path (path of each image, it can be just the name of image, e.g. 000.png) and ClassId (has to correspond to the
name of the folders in data/external/<dataset_name>/train).

Optional parameters (default values inside the parentheses) are:
- epochs (20): number of epochs to train.
- save_freq (5): while training, the model will be saved every 'save_freq' epochs.
- batch_size (256): batch size.
- img_res (50): Image resolution used in the model.
- lr (2e-3): initial learning rate.
- use_cosine_lr (True): flag to use cosine decay scheduler for the learning rate.
- model_name ('Recognizer'): If you want to give the model another name. Used for saving the model training history.
- extra (''): any extra information that you want to be saved in the file with the training parameters.

    python -m models.recognition.train_model --dataset_name <dataset_name>

### Predict
In the directory there must be the weights in a weights.ckpt file and the dictionary (neuron_to_class_dict.json)
to translate from neuron to class. This two files are created when training the model.

    python -m models.recognition.predict --dir_path <dir_path> --img_path <img_path> --dataset_name <dataset_name> --img_res <img_res>

# Project Organization


    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │                     predictions
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
