# gradCAM
grad CAM and similar visualization techniques for multi-label classification

This implementation is tested on multi-label classification of Pascal VOC images. At present Grad CAM is only available, CAM can be derived from it easily. More techniques beloniging to the same family will be added soon.

To create the visualizations, train the model first
```
python run.py cam_train --net vgg16 --dataset voc_2007_trainval --session 1 --batch_size 1 --total_epoch 40 --cuda --vis-off -ap color_mode=RGB image_range=1 mean="[0.485, 0.456, 0.406]" std="[0.229, 0.224, 0.225]"
```

This will save the model at each epoch. To create the visualizations run
```
python run.py cam_test --net vgg16 --dataset voc_2007_trainval --session 1 --epoch 5 --cuda -ap color_mode=RGB image_range=1 mean="[0.485, 0.456, 0.406]" std="[0.229, 0.224, 0.225]"
```

Here the model saved at epoch 5 is used.
