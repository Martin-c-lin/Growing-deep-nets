#import keras
from keras import  Input,models,layers
from keras.models import Sequential, Model, load_model
import numpy as np


from numpy.random import randint, uniform, normal, choice
from math import pi
import evaluate_deeptrack_performance as edp
import deeptrack

### load settings for previous image generator.
translation_distance = 1
particle_center_x_list = lambda : normal(0, translation_distance, translation_distance)
particle_center_y_list = lambda : normal(0, translation_distance, translation_distance)

#fbf_model_path = "C:/Users/Simulator/Desktop/Martin Selin/DeepTrack 1.0/FBF_modular new/models/high_noise_normal/"
#fbf_model_path = "C:/Users/Simulator/Desktop/Martin Selin/DeepTrack 1.0/FBF_modular new/models03_14/4-layers_long_traing1/"#high_noise_modular2/"
fbf_model_path = "models_default_training_modular/"

nbr_layers = 5
nbr_features = [16,32,64,32,32]
step = 16
SNT_levels = [5,10,20,30,50,100]
total_nbr_nodes = 0
for n in nbr_features:
    total_nbr_nodes+=n
fbf_model_results = np.zeros( (3,len(SNT_levels),round(total_nbr_nodes/step) ) )

nbr_images_to_evaluate = 1000
np.save("nbr_images_evaluated_on",nbr_images_to_evaluate)
np.save("SNT_levels",SNT_levels)
idx = 0
for i in range(nbr_layers):
    for j in range(round(nbr_features[i]/step)):
        nodes = (j+1)*step
        #single_output_full_model_1T_S16L1_16F
        #model_name = fbf_model_path+"single_output_full_model_1T_S16L"+str(i+1)+"_"+str(nodes)+"F.h5"
        model_name = fbf_model_path+"run1_modular_default_deeptrack_training_S16L"+str(i+1)+"_"+str(nodes)+"F.h5"
        #model_name =fbf_model_path+"layer_no"+str(i+1)+"top_size"+str(j+1)+".h5"

        model = load_model(model_name)
        fbf_model_results[0,:,idx],fbf_model_results[1,:,idx],fbf_model_results[2,:,idx]= edp.evaluate_noise_levels(
            model,
            SNT_levels,
            nbr_images_to_evaluate=nbr_images_to_evaluate,
            particle_center_x_list = particle_center_x_list ,
            particle_center_y_list = particle_center_y_list,
            )
        model.summary()
        print(fbf_model_results[:,0,idx])
        idx+=1
        del(model)
np.save("results/run1_modular"+str(step)+"S_default_training",fbf_model_results)
