import os
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.transform import rotate, resize

def create_folders(version):
    
    CKPT_OUTPUT_PATH = '/leonardo_scratch/fast/INA24_C3B13/GAN_ckpts'+version
    IMG_OUTPUT_PATH = 'GAN_Images'+version
    LOSS_OUTPUT_PATH = 'GAN_Loss'+version

    try:
        os.mkdir(CKPT_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(IMG_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(LOSS_OUTPUT_PATH)
    except FileExistsError:
        pass

    return CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, LOSS_OUTPUT_PATH

def augment_by_rotation(data, angles=[90,180,270]):
    original_data = data.copy()
    for angle in angles:
        original_data['rotation'] = angle
        data = pd.concat([data,original_data])
    return data

def get_unique(data):
    for col in data.columns[1:]:
        print(f'\n"{col}" has {len(data[col].unique())} unique values: {data[col].unique()}')
        
def dynamic_range_opt(array, epsilon=1e-6, mult_factor=1):
    array = (array + epsilon)/epsilon
    a = np.log10(array)
    b = np.log10(1/epsilon)
    return a/b * mult_factor

def load_meta_data(redshift, show=False):
    meta_data = pd.read_csv("mainframe.csv")
    meta_data=meta_data[meta_data['redshift']==redshift]

    meta_data = meta_data[['id','redshift', 'mass', 'simulation', 'snap', 
                           'ax', 'rot']].drop_duplicates()#.sort_values(by=['mass', 'rot']).reset_index(drop=True)
    
    # Showing what all is in my data
    if show:
        get_unique(meta_data)
    
    return meta_data

Dian_sim_fit = lambda m: -3.746 + 0.745*(np.log10(m) - 14.81) # Fit results within 5.00*r_500 for z=0.08
        
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, meta_data, X_col, y_col, batch_size, target_size, rot_col=False, shuffle=True):
        
        self.meta_data = meta_data.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.rot_col = rot_col
        self.n = len(self.meta_data)
        self.data_dir = "/leonardo_scratch/fast/INA24_C3B13/ALL_ROT_npy_version/1024x1024/"

    def on_epoch_end(self):
        if self.shuffle:
            print('Shuffling the data..')
            self.meta_data = self.meta_data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, img_id, mass, target_size):

        file_name = img_id + '.npy'
        file_name = self.data_dir + file_name

        img = np.load(file_name).astype('float32')
        img = tf.image.resize(np.expand_dims(img, axis=-1), target_size).numpy()

        # img /= 10**Dian_sim_fit(10**mass)

        img = dynamic_range_opt(img, mult_factor=2.5)
        
        return img
    
    def __get_output(self, label):
        return label
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_col_batch = batches[self.X_col]
        y_col_batch = batches[self.y_col]

        if self.rot_col:
            rot_col_batch = batches[self.rot_col]
            X_batch = np.asarray([self.__get_input(x, self.target_size, rot) for (x,rot) in zip(X_col_batch,rot_col_batch)])
        else: 
            # X_batch = np.asarray([self.__get_input(x, self.target_size) for x in X_col_batch])
            X_batch = np.asarray([self.__get_input(x, y, self.target_size) for x, y in zip(X_col_batch, y_col_batch)])
        
        y_batch = np.asarray([self.__get_output(y) for y in y_col_batch])
        
        return X_batch, y_batch
    
    def __getitem__(self, index):
        
        # The role of __getitem__ method is to generate one batch of data. 
        
        meta_data_batch = self.meta_data[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(meta_data_batch)
        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
    
