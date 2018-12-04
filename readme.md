# README
the guild for construct a python project: file structure

### how to saperate the modeule
* the module that mybe replace someday, should be a individule py file
* the config file and flag(commad line arg) just need one of two

### file structure (3 main module)
#### 1. generate dataset to tf.reacord format
 >dataset_gen.py
 >(generator a dataset which prepared for train form raw data)

#### 2. train model
> a) config.py
> b) unet.py
> (the net is replaceable)
> c) data_preprocess.py
> the preprocess is replaceable, alternative. if the process is much complex, then should be as a individual module)
> d) get_batch.py
> (creating batch queue from the dataset, then feed the network)
> e) train_main.py
> (train interface, input batch queue and output a h5 model weight file)
#### 3.  model deploy
> a) deploy.py/predict.py
> (predict after the training)
> b) visualization.py
> (used for visualizing the net['output'])






