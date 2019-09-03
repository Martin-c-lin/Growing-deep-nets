# Script for testing multi-particle tracking
import deeptrack
import cv2
import matplotlib.pyplot as plt

from numpy.random import randint, uniform, normal, choice
from math import pi
import numpy as np

def calculate_distances(true_positions,predicted_positions):
    import numpy as np

    nbr_particles = len(true_positions)
    nbr_predictions = len(predicted_positions)

    min_distances = 4000*np.ones(6)#np.zeros(nbr_particles) # problematic if different length when analyzing data later
    closest_predictions = -np.ones(6)#-np.ones(nbr_particles)
    nbr_false_positives = 0

    for i in range(nbr_particles):
        min_distances[i] = 1000000 #np.sqrt((true_positions[i,0] - predicted_positions[0,0])**2 + (true_positions[i,1]-predicted_positions[0,1])**2 )
        for j in range(nbr_predictions):
            new_dist = np.sqrt((true_positions[i,0] - predicted_positions[j,0])**2 + (true_positions[i,1]-predicted_positions[j,1])**2 )
            if new_dist<min_distances[i]:
                min_distances[i] = new_dist
                closest_predictions[i] = j

    #print(closest_predictions)
    closest_predictions = closest_predictions.astype(int)
    for i in range(nbr_predictions):
        if i not in closest_predictions:
            #print('False positive detected',i,' not in true values',nbr_predictions)
            nbr_false_positives += 1
    return min_distances,closest_predictions.astype(int),nbr_false_positives
def find_r_threshold(predicted_positions,indices):
    r = 0
    for idx in indices:
        if predicted_positions[idx,2]>r:
            r = predicted_positions[idx,2]
    return r
def extract_accurate_predictions_from_enhanced(prediction_wrt_frame,indices):
    import numpy as np
    tmp = []
    for i in range(len(indices)):
        tmp.append(prediction_wrt_frame[indices[i,0],indices[i,1],:])
    return np.asarray(tmp)
def distance2D(a,b):
    from numpy import sqrt
    return sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def find_best_predictions(predictions,min_dist=8):
    import numpy as np
    best_predictions = []
    while len(predictions)>0:
        # assign new particle
        best_predictions.append(predictions[0]) # move first element to ...
        predictions = predictions[1:] # remove first element
        # Check which other predictions belong to the same particle
        i=0
        while i <len(predictions):
            if distance2D(predictions[i],best_predictions[-1])<min_dist: # belong to same particle
                tmp = predictions[i]#.pop(i)
                predictions = np.delete(predictions,i,0)
                i-=1
                if tmp[2]<best_predictions[-1][2]:
                    best_predictions[-1] = tmp
            i += 1
    return np.asarray(best_predictions)
def reshape_predictions(prediction_wrt_frame,image_parameters):
    import numpy as np

    tmp = np.reshape(prediction_wrt_frame,(len(prediction_wrt_frame)**2,3))
    predictions = np.zeros((len(tmp),3))
    predictions[:,1] = tmp[:,0]
    predictions[:,0] = tmp[:,1]
    predictions[:,2] = tmp[:,2]

    nbr_particles = len(image_parameters['Particle Center X List'])
    true_positions = np.zeros((nbr_particles,2))
    true_positions[:,0] = image_parameters['Particle Center Y List']+image_parameters['Image Half-Size']
    true_positions[:,1] = image_parameters['Particle Center X List']+image_parameters['Image Half-Size']
    return predictions,true_positions
def extract_low_r_predictions(predicitons,nbr_to_extract=18): # 12
    tmp = predictions[predictions[:, 2].argsort()]
    return tmp[:nbr_to_extract,:]
particle_number = 6
image_half_size = 125
first_particle_range = round(image_half_size*0.8)
other_particle_range = round(image_half_size*0.8)
particle_distance = 20 # 30?

def get_image_parameters_optimized():
    (particles_center_x, particles_center_y) = deeptrack.particle_positions(particle_number, first_particle_range, other_particle_range, particle_distance)
    image_parameters = {}

    image_parameters['Particle Center X List'] = particles_center_x
    image_parameters['Particle Center Y List'] = particles_center_y
    image_parameters['Particle Radius List'] = uniform(2, 3, particle_number)
    image_parameters['Particle Bessel Orders List'] = [[1, ],
                                                       [1, ],
                                                       [1, ],
                                                       [1, ],
                                                       [1, ],
                                                       [1, ]]
    image_parameters['Particle Intensities List'] = [[uniform(0.4, 0.7, 1), ], # make more consistent
                                                     [uniform(0.4, 0.7, 1), ],
                                                     [uniform(0.4, 0.7, 1), ],
                                                     [uniform(0.4, 0.7, 1), ],
                                                     [uniform(0.4, 0.7, 1), ],
                                                     [uniform(0.4, 0.7, 1)] ]
    image_parameters['Image Half-Size'] = image_half_size
    image_parameters['Image Background Level'] = 0.2#uniform(.2, .4)
    image_parameters['Signal to Noise Ratio'] = 5#10#20#uniform(2, 8)
    image_parameters['Gradient Intensity'] = 0#uniform(0, 0.2)
    image_parameters['Gradient Direction'] = uniform(-pi, pi)
    image_parameters['Ellipsoid Orientation'] = uniform(-pi, pi, particle_number)
    image_parameters['Ellipticity'] = 1

    return image_parameters



image_parameters_function = lambda : get_image_parameters_optimized()

### Define image generator
image_generator = lambda : deeptrack.get_image_generator(image_parameters_function)

### Show some examples of generated images
#path = 'vesicle_models/LBL_vesicles_larger_r_rangerun_0_network_no' # WORKS!!!
path = 'vesicle_models/default_training_run_1_network_no' # ok, 8.7as threshold for large, run 0!?!?!
#path = 'vesicle_models/LBL_vesiclesrun_2_network_no'
#path = 'multi_particle/large_dist_vesicle_training_run0_network_no'
saved_network_file_name = path+'0.h5'#'vesicle_models/LBL_vesicles_larger_r_rangerun_0_network_no0.h5'
network_small = deeptrack.load(saved_network_file_name)
saved_network_file_name = path+'4.h5' # 4.h5
network_large = deeptrack.load(saved_network_file_name)

enhanced_thres = 7.1
small_thres_enhanced = 7.5 # use smaller for actual accurate predictionsmall_thres
small_thres = 7#7.3
large_thres = 6.7

# OK for default_training_run_0_network_no
# enhanced_thres = 8.2#7.95
# small_thres_enhanced = 7.8 # use smaller for actual accurate predictionsmall_thres
# small_thres = 7.5
# large_thres = 8.2
number_of_images_to_process = 1000
box_scanning_step = 10#10 ok for default training 15 # too large with 15 ? ...
# 15 -> 14x14
# 10 -> 21x21
false_positive_counts_large = 0
false_positive_counts_small = 0
false_positive_counts_enhanced = 0

distances_large = []
distances_enhanced = []

distances_small = []
images = []
image_parameters_list = []
for image_number, image, image_parameters in image_generator():
    if image_number>=number_of_images_to_process:
        break
    images.append(image)
    image_parameters_list.append(image_parameters)
    print('New image generated no',image_number)

for i in range(number_of_images_to_process):
    print(i)
    image = images[i]
    image_parameters = image_parameters_list[i]
    #deeptrack.plot_sample_image(image, image_parameters)

    #plt.imshow(image,cmap='gray')

    # MAKE large and small predictions
    prediction_wrt_box,prediction_wrt_frame_small,boxes = deeptrack.track_frame(network_small,image,box_half_size=25,box_scanning_step=box_scanning_step)
    prediction_wrt_box,prediction_wrt_frame_large,boxes = deeptrack.track_frame(network_large,image,box_half_size=25,box_scanning_step=box_scanning_step)
    print(boxes.shape)

    predictions,true_positions = reshape_predictions(prediction_wrt_frame_large,image_parameters)
    # predictions = extract_low_r_predictions(predictions)
    # min_distances_large,closest_predictions = calculate_distances(true_positions,predictions)
    # r_threshold = find_r_threshold(predictions,closest_predictions)
    to_plot_large = predictions[predictions[:,2]<large_thres]
    to_plot_large = find_best_predictions(to_plot_large)
    min_distances_large,closest_predictions,nbr_false_positives_large = calculate_distances(true_positions,to_plot_large)
    false_positive_counts_large += nbr_false_positives_large
    # Plot predictions
    #plt.plot(to_plot_large[:,0],to_plot_large[:,1],'*r',markersize=10,label='large')
    #print('large network r values: ',predictions[:,2])  # For checking suitable r threshold
    distances_large.append(min_distances_large)


    predictions,true_positions = reshape_predictions(prediction_wrt_frame_small,image_parameters) # Small
    to_plot_small = predictions[predictions[:,2]<small_thres]
    to_plot_small = find_best_predictions(to_plot_small)
    #plt.plot(to_plot_small[:,0],to_plot_small[:,1],'xb',label='small')
    min_distances_small,closest_predictions,nbr_false_positives_small = calculate_distances(true_positions,to_plot_small)
    #predictions = extract_low_r_predictions(predictions)
    #min_distances_small,closest_predictions = calculate_distances(true_positions,predictions)
    r_threshold = find_r_threshold(predictions,closest_predictions)
    distances_small.append(min_distances_small)
    false_positive_counts_small += nbr_false_positives_small

    #print('small network r values: ',predictions[:,2]) # For checking suitable r threshold

    r_thres = predictions[-1,2] # find suitable Threshold
    prediction_wrt_box,prediction_wrt_frame_enhanced,boxes,accuracte_indices = deeptrack.track_frame_multi_network(
        network_small,
        network_large,
        image,
        box_half_size=25,
        box_scanning_step=box_scanning_step,
        intermid_idx=2,
        particle_radial_distance_threshold=small_thres_enhanced,#difficult to choose-> won't work right of the batch
        # 7 PRETTY GOOD FOR default_training_run_0_network_no
        )

    predictions = extract_accurate_predictions_from_enhanced(prediction_wrt_frame_enhanced,accuracte_indices)

    # min_distances_enhanced,closest_predictions = calculate_distances(true_positions,predictions)
    # r_threshold = find_r_threshold(predictions,closest_predictions)
    #for idx in closest_predictions:
    #print('enhanced: ',predictions[:12,2])

    to_plot_enhanced = predictions[predictions[:,2]<large_thres]
    to_plot_enhanced = find_best_predictions(to_plot_enhanced)

    # Shuffle indices to be same as plot
    tmp = np.zeros(to_plot_enhanced.shape)
    tmp[:,0] = to_plot_enhanced[:,1]
    tmp[:,1] = to_plot_enhanced[:,0]
    tmp[:,2] = to_plot_enhanced[:,2]

    min_distances_enhanced,closest_predictions,nbr_false_positives_enhanced = calculate_distances(true_positions,tmp)
    distances_enhanced.append(min_distances_enhanced)
    false_positive_counts_enhanced += nbr_false_positives_enhanced
