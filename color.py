from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.callbacks import Callback
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
import cv2

run = wandb.init()
config = run.config

config.num_epochs = 1
config.batch_size = 4
config.img_dir = "images"
config.height = 256
config.width = 256

val_dir = 'test'
train_dir = 'train'

# automatically get the data if it doesn't exist
if not os.path.exists("train"):
	print("Downloading flower dataset...")
	subprocess.check_output("curl https://storage.googleapis.com/l2kzone/flowers.tar | tar xz", shell=True)

def my_generator(batch_size, img_dir):
	"""A generator that returns black and white images and color images"""
	image_filenames = glob.glob(img_dir + "/*")
	counter = 0
	while True:
		bw_images = np.zeros((batch_size, config.width, config.height))
		color_images = np.zeros((batch_size, config.width, config.height, 3))
		random.shuffle(image_filenames) 
		if ((counter+1)*batch_size>=len(image_filenames)):
			  counter = 0
		for i in range(batch_size):
			  img = Image.open(image_filenames[counter + i]).resize((config.width, config.height))
			  color_images[i] = np.array(img)
			  bw_images[i] = np.array(img.convert('L'))
		yield (bw_images, color_images)
		counter += batch_size

def original_model():

	model = Sequential()
	model.add(Reshape((config.height,config.width,1), input_shape=(config.height,config.width)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(2,2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
	
	return model

def add_conv_stack(model, num_filters, stack=2):

	for _ in range(stack - 1):
		model.add(Conv2D(num_filters, (3, 3), activation='relu', padding='same'))
		model.add(DownSampling2D(2,2))

	model.add(Conv2D(num_filters, (3, 3), activation='relu', padding='same', strides=(2, 2)))
	# model.add(BatchNormalization())



def my_model():

	print(config.height)
	
	model = Sequential()
	model.add(Reshape((config.height,config.width,1), input_shape=(config.height,config.width)))
	
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))) # 128
	model.add(BatchNormalization())

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))) # 64
	model.add(BatchNormalization())

	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2))) # 32
	model.add(BatchNormalization())

	# 4
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1))) # 32
	model.add(BatchNormalization())

	# 5
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1), dilation_rate=(2,2))) # 32
	model.add(BatchNormalization())	
	
	# 6
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1), dilation_rate=(2,2))) # 32
	model.add(BatchNormalization())	
	
	# 7
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1))) # 32
	model.add(UpSampling2D(2))
	model.add(BatchNormalization())	

	# 8
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1)))
	model.add(UpSampling2D(2)) # 64
	model.add(BatchNormalization())	


	model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2)) # 128
	# model.add(BatchNormalization())	

	






# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	# model.add(MaxPooling2D(2,2))
	# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	# model.add(UpSampling2D((2, 2)))
	# model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))


	# model.add(MaxPooling2D(2,2))

	

	# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	# model.add(UpSampling2D((2, 2)))
	# model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))


	
	# Conv 1
	# add_conv_stack(model, 64)

	# model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

	# Conv 2
	# model.add(UpSampling2D((2, 2)))
	# add_conv_stack(model, 128)

	# model.add(UpSampling2D((2, 2)))
	# add_conv_stack(model, 256, stack=3)

	# add_conv_stack(model, 512, stack=3)

	# # Conv 5 is dilated
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	# model.add(BatchNormalization())

	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	# model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2,2)))
	# model.add(BatchNormalization())

	# add_conv_stack(model, 512, stack=3)

	# add_conv_stack(model, 256, stack=3)
	
	# model.add(UpSampling2D((2, 2)))

	# model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

	return model


model = my_model()
# model = original_model()
model.compile(optimizer='adam', loss='mse')

(val_bw_images, val_color_images) = next(my_generator(145, val_dir))

# model.fit_generator( my_generator(config.batch_size, train_dir),
# 					 steps_per_epoch=20,
# 					 epochs=config.num_epochs, callbacks=[WandbCallback(data_type='image', predictions=16)],
# 					 validation_data=(val_bw_images, val_color_images))

model.fit_generator( my_generator(config.batch_size, train_dir),
					 steps_per_epoch=20,
					 epochs=config.num_epochs,
					 validation_data=(val_bw_images, val_color_images))


