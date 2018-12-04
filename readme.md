# README
the guild for construct a python project: file structure

### how to saperate the modeule
* the module that mybe replace someday, should be a individule py file
* the config file and flag(commad line arg) just need one of two

### file structure (5 module)
#### 1.  dataset_gen.py
 >(generator a dataset which prepared for train form raw data)

#### 2.  config.py
#### 3.  unet.py
 >(the net is replaceable)
#### 4. data_preprocess.py
 >the preprocess is replaceable, alternative. if the process is much complex, then should be as a individual module)
#### 5.  get_batch.py
 >(creating batch queue from the dataset, then feed the network)
#### 6.  train_main.py
 >(train interface, input batch queue and output a h5 model weight file)
#### 7.  deploy.py/predict.py
 >(after the training)







