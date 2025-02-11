import sys
import math
import numpy as np
import tensorflow as tf
sys.path.append('../')

from model import *
from data_utils import *
from gan_utils import *
from model_utils import *

def train(G_LR, D_LR, R_LR, EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH, END_SIZE, cbk, pgan, meta_data, loss_out_path):

    generator_optimizer     = tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)
    regressor_optimizer     = tf.keras.optimizers.Adam(learning_rate=R_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)
    
    pgan.compile(d_optimizer=discriminator_optimizer,g_optimizer=generator_optimizer, r_optimizer=regressor_optimizer)

    # Start training the initial generator and discriminator
    START_SIZE = 4
    train_dataset = CustomDataGen(meta_data, X_col='id', y_col='mass', rot_col = False, batch_size = BATCH_SIZE[0], 
                                  target_size=(START_SIZE, START_SIZE), shuffle=True)
    #  4 x 4
    print('SIZE: ', START_SIZE)
    history_init  = pgan.fit(train_dataset, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, callbacks=[cbk], verbose=1)
    np.save(f'{loss_out_path}/history_init.npy', history_init.history)

    n_depth_init = 1

    # Train faded-in/stabilized generators and discriminators
    for n_depth in range(n_depth_init, int(math.log(END_SIZE,2))-1):
        print('------------------------------------------------------------------------ \n')
        print('------------------------------------------------------------------------')

        print('SIZE: ', 4*(2**n_depth))

        pgan.n_depth = n_depth

        steps_per_epoch = STEPS_PER_EPOCH # 110 

        train_dataset = CustomDataGen(meta_data, X_col='id', y_col='mass', rot_col = False, batch_size = BATCH_SIZE[n_depth], 
                                      target_size=(4*(2**n_depth), 4*(2**n_depth)), shuffle=True)

        cbk.set_prefix(prefix=f'{n_depth}_fade_in')
        cbk.set_steps(steps_per_epoch=steps_per_epoch, epochs=EPOCHS)

        # Put fade in generator and discriminator
        print(f'Fading in for {(4*(2**n_depth), 4*(2**n_depth))} image..')
        pgan.fade_in_generator()
        pgan.fade_in_discriminator()
        pgan.fade_in_regressor()

        pgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8),
                     g_optimizer=tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8),
                     r_optimizer=tf.keras.optimizers.Adam(learning_rate=R_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8))
        # Train fade in generator and discriminator
        history_fade_in = pgan.fit(train_dataset, steps_per_epoch = steps_per_epoch, epochs = EPOCHS, callbacks=[cbk], verbose=1) 
        np.save(f'{loss_out_path}/history_fade_in_{n_depth}.npy',history_fade_in.history)

        # Change to stabilized generator and discriminator
        cbk.set_prefix(prefix=f'{n_depth}_stabilize')
        print(f'Stabilizing for {(4*(2**n_depth), 4*(2**n_depth))} image..')
        pgan.stabilize_generator()
        pgan.stabilize_discriminator()
        pgan.stabilize_regressor()

        pgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8),
                     g_optimizer=tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8),
                     r_optimizer=tf.keras.optimizers.Adam(learning_rate=R_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8))

        # Train stabilized generator and discriminator
        history_stabilize = pgan.fit(train_dataset, steps_per_epoch = steps_per_epoch, epochs = EPOCHS, callbacks=[cbk], verbose=1) #train alpha = 1 
        np.save(f'{loss_out_path}/history_stabilize_{n_depth}.npy', history_stabilize.history)

    new_train_dataset = CustomDataGen(meta_data, X_col='id', y_col='mass', rot_col = False, batch_size = BATCH_SIZE[n_depth], 
                                  target_size=(END_SIZE, END_SIZE))
    
    cbk.set_prefix(prefix=f'{n_depth}_final')
    history_final_step = pgan.fit(new_train_dataset, steps_per_epoch = steps_per_epoch, epochs = 2*EPOCHS, callbacks=[cbk], verbose=1) #train alpha = 1 
    np.save(f'{loss_out_path}/history_final_step_{new_train_dataset.mask_par}.npy', history_final_step.history)

    return pgan

def train_fixed(G_LR, D_LR, R_LR, EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH, END_SIZE, cbk, pgan, meta_data, loss_out_path):

    for n_depth in range(1, int(np.log2(END_SIZE/2))):
        pgan.n_depth = n_depth
        pgan.fade_in_generator()
        pgan.fade_in_discriminator()
        pgan.fade_in_regressor()

        pgan.stabilize_generator()
        pgan.stabilize_discriminator()
        pgan.stabilize_regressor()

    generator_optimizer     = tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)
    regressor_optimizer     = tf.keras.optimizers.Adam(learning_rate=R_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)
    
    pgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer, r_optimizer=regressor_optimizer)
    train_dataset = CustomDataGen(meta_data, X_col='id', y_col='mass', rot_col = False, batch_size = BATCH_SIZE[0], 
                                  target_size=(END_SIZE, END_SIZE))
    
    cbk.set_prefix(prefix=f'{n_depth}_final') 
    history_final_step = pgan.fit(train_dataset, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, callbacks=[cbk], verbose=1) #train alpha = 1 
    np.save(f'{loss_out_path}/history_{n_depth}_final.npy', history_final_step.history)

    return pgan