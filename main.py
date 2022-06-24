from utils.crop import crop_image_gui
from utils.hsv_range import adjust_hsv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import json
import scipy
import scipy.ndimage

SEQUENCES_DIR = 'sequences'
RESIZE_FACTOR = 0.5

def get_config(sequence_dir):
	file = os.path.join(sequence_dir, 'config.json')
	with open(file, 'r') as f:
		config = json.load(f)
	return config


def get_diameter(img):

	 # make array for final values
	HSV_LOW = np.array([0, 0, 193])
	HSV_HIGH = np.array([179, 255, 255])

	nb_pixels = []

	for l in range(img.shape[0]):

		line = img[l, :, :]

		line = np.expand_dims(line, axis = 0)
		line = cv2.cvtColor(line, cv2.COLOR_BGR2HSV)
		line = cv2.inRange(line, HSV_LOW, HSV_HIGH)
		line = cv2.erode(line, None, iterations=1)
		line = cv2.dilate(line, None, iterations=1)

		non_masked_pixels = np.count_nonzero(line)
		nb_pixels.append(non_masked_pixels)

	result = max(nb_pixels)
	return result


def get_diameter_sequence(sequence_dir):
	
	diameters = []

	image_list = os.listdir(sequence_dir)

	# remove non image files
	image_list = [x for x in image_list if x.endswith('.tif')]

	image_list = sorted(image_list, reverse=True, key=lambda x:int(x.split('.')[0]))
	# we invert list to get the cells from the smallest to the largest
	image_list = image_list[::-1]

	# DON'T FORGET TO RESIZE IMAGES

	origin_img = cv2.imread(os.path.join(sequence_dir, image_list[-1]))
	origin_img = cv2.resize(origin_img, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

	y_start, y_end, x_start, x_end = crop_image_gui(origin_img)

	for image in image_list:

		img = cv2.imread(os.path.join(sequence_dir, image))
		img = cv2.resize(img, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)
		
		img_crop = img[y_start:y_end, x_start:x_end]
		diameter = get_diameter(img_crop)
		diameters.append(diameter)

	return diameters


def plot_save_values(values, time, sequence_dir, name, unit):
	# plot diameters

	start, step, end = time
	x = np.arange(start, end, step)

	#  plot
	fig, ax = plt.subplots()
	ax.plot(x, values, marker='o', linestyle='-', color='r', )
	ax.set_ylabel(f"{name} ({unit})")
	ax.set_xlabel("time (min)")
	save_plot_path = os.path.join(sequence_dir, f'{name}_plot.png')
	fig.savefig(save_plot_path)

	# plot smooth
	fig, ax = plt.subplots()
	# smooth_values = scipy.signal.savgol_filter(values, window_length=11, polyorder=3)
	smoothed = scipy.ndimage.gaussian_filter(values, 3)
	ax.plot(x, smoothed, marker='o', linestyle='-', color='r', )
	ax.set_ylabel(f"{name} ({unit})")
	ax.set_xlabel("time (min)")
	save_plot_path = os.path.join(sequence_dir, f'{name}_smoothed_plot.png')
	fig.savefig(save_plot_path)


	save_text_path = os.path.join(sequence_dir, f'{name}_text.txt')

	with open(save_text_path, 'w') as f:
		for i, value in zip(range(start, end, step), values):
			f.write(f"{i} {value} \n")
	
	

def save_results(diameters, sequence_dir):


	# 1px ----> unit
	# resize factor * px ----> unit

	config = get_config(sequence_dir)

	diameters = np.array(diameters)

	diameters = diameters / RESIZE_FACTOR

	diameters = diameters * config['pixel_size']['value']

	rayons = diameters / 2

	unit = config['pixel_size']['unit']


	start = config['time']['start']
	step = config['time']['step']
	end = start + step * len(diameters)

	plot_save_values(diameters, (start, step, end), sequence_dir, 'diameter', unit)
	plot_save_values(rayons, (start, step, end), sequence_dir, 'rayon', unit)



def treat_sequences(sequences_dir):

	print("Start...")

	for sequence_dir in os.listdir(sequences_dir):
		
		dir = os.path.join(sequences_dir, sequence_dir)

		if not os.path.isdir(dir) : continue
		if not sequence_dir.startswith('seq') : continue

		print(f"\tTreating {sequence_dir}...",  end="")

		SEQUENCE_DIR = os.path.join(sequences_dir, sequence_dir)

		diameters = get_diameter_sequence(SEQUENCE_DIR)
		save_results(diameters, SEQUENCE_DIR)

		print("Done")




def main():
	treat_sequences(SEQUENCES_DIR)


	


if __name__ == '__main__':
	main()