import utils.util as util

import os
import numpy as np

def get_dictionaries(dataset_path,train_split=0.7):
    file_names = {}

    for i,filename in enumerate(os.listdir(dataset_path)):
        file_names[i+1] = filename

    partition = {'train_ids': [], 'val_ids': []}

    indices = np.arange(1,len(file_names)+1)
    util.shuffleArray(indices,0)

    n_train_images = int(train_split*len(indices))
    partition['train_ids'] = indices[0:n_train_images]
    partition['val_ids'] = indices[n_train_images:len(indices)+1]

    return (partition,file_names)
