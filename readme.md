# README
the guild for construct a python project: file structure

### how to saperate the modeule
* the module that mybe replace someday, should be a individule py file
* the config file and flag(commad line arg) just need one of two

### file structure (5 module)
* dataset_gen.py 
>
(generator a dataset which prepared for train form raw data)

* config.py
* unet.py
>
(the net is replaceable)
*data_preprocess.py
>
the preprocess is replaceable, alternative. if the process is much complex, then should be as a individual module)
* get_batch.py
>
(creating batch queue from the dataset, then feed the network)
* train_main.py
>
(train interface, input batch queue and output a h5 model weight file)
* deploy.py/predict.py
>
(after the training)







