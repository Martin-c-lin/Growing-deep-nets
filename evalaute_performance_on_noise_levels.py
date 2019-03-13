#import keras
from keras import  Input,models,layers
from keras.models import Sequential, Model, load_model
import numpy as np


from numpy.random import randint, uniform, normal, choice
from math import pi
import evaluate_deeptrack_performance as edp
import deeptrack

### load settings for previous image generator.
translation_distance = 5
particle_center_x_list = lambda : normal(0, translation_distance, translation_distance)
particle_center_y_list = lambda : normal(0, translation_distance, translation_distance)
particle_radius_list=lambda : uniform(3, 5, 1)
#fbf_model_path = "C:/Users/Simulator/Desktop/Martin Selin/DeepTrack 1.0/FBF_modular new/models/high_noise_normal/"
fbf_model_path = "C:/Users/Simulator/Desktop/Martin Selin/DeepTrack 1.0/FBF_modular new/models/"#high_noise_modular2/"

nbr_layers = 1
nbr_features = 17
#SNT_levels = [5,10,20,30,50,100]
SNT_levels = [5,10,20,30,50,100]
fbf_model_results = np.zeros((3,len(SNT_levels),nbr_layers,nbr_features))

nbr_images_to_evaluate = 1000
np.save("nbr_images_evaluated_on",nbr_images_to_evaluate)
np.save("SNT_levels",SNT_levels)

for i in range(nbr_layers):
    for j in range(nbr_features):
        model_name = fbf_model_path+"avg_testingL"+str(i+1)+"_"+str(j+1)+"F.h5"
        #model_name =fbf_model_path+"layer_no"+str(i+1)+"top_size"+str(j+1)+".h5"
        model = load_model(model_name)
        fbf_model_results[0,:,i,j],fbf_model_results[1,:,i,j],fbf_model_results[2,:,i,j]= edp.evaluate_noise_levels(
            model,
            SNT_levels,
            nbr_images_to_evaluate=nbr_images_to_evaluate,
            particle_center_x_list = particle_center_x_list ,
            particle_center_y_list = particle_center_y_list,
            #particle_radius_list=particle_radius_list
            )
        model.summary()
        print(fbf_model_results[:,0,i,j])
        del(model)
np.save("results/avg_modular",fbf_model_results)
