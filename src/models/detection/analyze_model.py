import os

from constants import PROJECT_PATH
from data.dataset import Dataset
from helpers import natural_keys
from models.detection.predict import detect
from models.detection.yolov3 import YOLOv3

weights = None
checkpoint_dir = '/home/brechard/models/YOLOv3/20191031_163044_COCO/checkpoints/'

dataset_name = checkpoint_dir.split('_')[-1].split('/')[0]
dataset = Dataset(dataset_name)
model = YOLOv3(tiny=True)
model.load_models(dataset=dataset, for_training=False)

# jajas.check_losses(None, "RANDOM INITIALIZATION")
print()
image_path = PROJECT_PATH + 'data/external/datasets/COCO/train/000000000257.jpg'

for checkpoint in sorted(os.listdir(checkpoint_dir), key=natural_keys):
    new_weights = checkpoint.split('.ckpt')[0]
    if new_weights == weights or 'ckpt' not in checkpoint:
        continue
    else:
        weights = new_weights
        print('Use weights', weights)
        title = 'Epoch = ' + weights.split('-')[0].split('_')[-1] + '. Model loss = ' + \
                checkpoint.split('.ckpt')[0].split('-')[-1]
        detect(model,
               dataset_name,
               image_path=image_path,
               weights_path=checkpoint_dir + checkpoint[:checkpoint.find('ckpt') + len('ckpt')],
               title=title)
        # jajas.check_losses(checkpoint_dir + checkpoint[:checkpoint.find('ckpt') + len('ckpt')], title, dataset,
        #                    model)
        print()

# TODO: The analysis was done long ago, it has to be adapted to the recent changes
