import numpy as np

# Darknet configuration files
config_file = "./darknet/cfg/yolov4-thesis.cfg"
data_file = "./darknet/thesis.so.data"
weights_file = "./darknet/yolov4-thesis_best.weights"

# Camera parameters
d = np.array([-0.03432, 0.05332, -0.00347, 0.00106, 0.00000, 0.0, 0.0, 0.0]).reshape(1, 8) # distortion coefficients
K = np.array([1189.46, 0.0, 805.49, 0.0, 1191.78, 597.44, 0.0, 0.0, 1.0]).reshape(3, 3) # Camera matrix
K_inv = np.linalg.inv(K)

# Othes
images_file = "./images.txt"
images_folder = "./images/"
images_file_sg = "./images-sg.txt"
pairs_folder = "./pairs/"

# CMDs
cmd_superglue = "./superglue/match_pairs.py --input_pairs " + images_file_sg + " --input_dir " + images_folder + \
    " --output_dir " + pairs_folder + " --match_threshold .2 --resize -1"
cmd_remove =  "rm " + pairs_folder + "*.npz"