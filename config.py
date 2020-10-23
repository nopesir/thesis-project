import numpy as np

# Darknet configuration files
config_file = "./darknet/cfg/yolov4-thesis.cfg"
data_file = "./darknet/thesis.so.data"
weights_file = "./darknet/yolov4-thesis_best.weights"
hasGPU = True

# Camera parameters (Redmi Note 8T)
with np.load('camera.npz') as X:
    K, d, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

print(K)
K_inv = np.linalg.inv(K)

# Othes
images_file = "./images.txt"
images_folder = "./images/"
images_file_sg = "./images-sg.txt"
pairs_folder = "./pairs/"

# CMDs
if not False:
    cmd_superglue = "./superglue/match_pairs.py --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float --input_pairs " + images_file_sg + " --input_dir " + images_folder + \
    " --output_dir " + pairs_folder + " --match_threshold .3 --resize -1 --force_cpu --viz --viz_extension pdf"
else:
    cmd_superglue = "./superglue/match_pairs.py --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float --input_pairs " + images_file_sg + " --input_dir " + images_folder + \
    " --output_dir " + pairs_folder + " --match_threshold .3 --resize -1 --viz --viz_extension pdf"
cmd_remove =  "rm " + pairs_folder + "*"