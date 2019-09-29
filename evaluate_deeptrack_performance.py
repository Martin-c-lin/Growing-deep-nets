import deeptrack
import numpy as np
### Define image properties
from numpy.random import randint, uniform, normal, choice
from math import pi
def get_new_parameters_function(sn_level,image_parameters_function):
    image_parameters = image_parameters_function()
    image_parameters['Signal to Noise Ratio'] = sn_level
    return image_parameters
def evaluate_noise_levels(
    network,
    SNT_levels,
    nbr_images_to_evaluate=1000,
    particle_center_x_list = lambda : normal(0, 1, 1),
    particle_center_y_list=lambda : normal(0, 1, 1),
    particle_radius_list=lambda : uniform(1.5, 3, 1),
    particle_bessel_orders_list=lambda : [[randint(1, 3),], ],
    particle_intensities_list=lambda : [[choice([-1, 1]) * uniform(.2, .6, 1), ], ],
    image_half_size=lambda : 25,
    image_background_level=lambda : uniform(.2, .8),
    signal_to_noise_ratio=lambda : uniform(10, 100),
    gradient_intensity=lambda : uniform(0, 1),
    gradient_direction=lambda : uniform(-pi, pi)):
    """
    Function for evaluating performance of network for different noise levels
    Inputs: network to be evaluated, noise levels to evaluate on and parameters of the image generator(other than noise level)
        Number of images to evaluate on
    Outputs: Loss,mae and mse for the various noise levels
    """

    image_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]

    losses = []
    MSEs = []
    MAEs = []

    for SNT_level in SNT_levels:

        ### Create an image generator
        image_parameters_function = lambda : deeptrack.get_image_parameters(
           particle_center_x_list=particle_center_x_list,
           particle_center_y_list=particle_center_y_list,
           particle_radius_list=particle_radius_list,
           particle_bessel_orders_list=particle_bessel_orders_list,
           particle_intensities_list=particle_intensities_list,
           image_half_size=image_half_size,
           image_background_level=image_background_level,
           signal_to_noise_ratio= lambda: SNT_level,
           gradient_intensity=gradient_intensity,
           gradient_direction= gradient_direction)
        image_generator = lambda : deeptrack.get_image_generator(image_parameters_function)

        ### Genereate images for evaluation
        images, targets = deeptrack.get_images_and_targets(image_generator,nbr_images_to_evaluate,image_shape=image_shape)
        loss,mse,mae = network.evaluate(images,targets)

        ### Fix units and append to final results
        mse *= image_half_size()**2 #what happens if it is not a function? Silly error?
        mae *= image_half_size()
        losses.append(loss)
        MSEs.append(mse)
        MAEs.append(mae)
    return losses,MSEs,MAEs
def predictions_on_noise_levels(
    network,
    SNT_levels,
    nbr_images_to_evaluate=1000,
    particle_center_x_list = lambda : normal(0, 1, 1),
    particle_center_y_list=lambda : normal(0, 1, 1),
    particle_radius_list=lambda : uniform(1.5, 3, 1),
    particle_bessel_orders_list=lambda : [[randint(1, 3),], ],
    particle_intensities_list=lambda : [[choice([-1, 1]) * uniform(.2, .6, 1), ], ],
    image_half_size=lambda : 25,
    image_background_level=lambda : uniform(.2, .8),
    signal_to_noise_ratio=lambda : uniform(10, 100),
    gradient_intensity=lambda : uniform(0, 1),
    gradient_direction=lambda : uniform(-pi, pi)):
    """
    Function for making predictions on different noise levels
    Inputs: network to be evaluated, noise levels to evaluate on and parameters of the image generator(other than noise level)
        Number of images to evaluate on
    Outputs: predicted positions and target positions.
    """

    image_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]

    all_predictions = []
    all_targets = []

    for SNT_level in SNT_levels:

        ### Create an image generator
        image_parameters_function = lambda : deeptrack.get_image_parameters(
           particle_center_x_list=particle_center_x_list,
           particle_center_y_list=particle_center_y_list,
           particle_radius_list=particle_radius_list,
           particle_bessel_orders_list=particle_bessel_orders_list,
           particle_intensities_list=particle_intensities_list,
           image_half_size=image_half_size,
           image_background_level=image_background_level,
           signal_to_noise_ratio= lambda: SNT_level,
           gradient_intensity=gradient_intensity,
           gradient_direction= gradient_direction)
        image_generator = lambda : deeptrack.get_image_generator(image_parameters_function)

        ### Genereate images for evaluation
        images, targets = deeptrack.get_images_and_targets(image_generator,nbr_images_to_evaluate,image_shape=image_shape)
        all_targets.append(targets)
        predictions = network.predict(images)
        all_predictions.append(predictions)

    return all_predictions,all_targets

def separate_ensemble_predictions(
    networks,
    SNT_levels,
    nbr_images_to_evaluate=1000,
    translation_distance=1,
    particle_radius_list=lambda : uniform(1.5, 3, 1),
    particle_bessel_orders_list=lambda : [[randint(1, 3),], ],
    particle_intensities_list=lambda : [[choice([-1, 1]) * uniform(.2, .6, 1), ], ],
    image_half_size=lambda : 25,
    image_background_level=lambda : uniform(.2, .8),
    signal_to_noise_ratio=lambda : uniform(10, 100),
    gradient_intensity=lambda : uniform(0, 1),
    gradient_direction=lambda : uniform(-pi, pi)):
    """
    Function for making predictions by multiple networks on the same images(generated)
    """
    import numpy as np
    all_targets,all_images = generate_images_and_targets_from_noise_levels(
        networks[0],
        SNT_levels,
        nbr_images_to_evaluate=nbr_images_to_evaluate,
        translation_distance=translation_distance,
        particle_radius_list=particle_radius_list,
        particle_bessel_orders_list=particle_bessel_orders_list,
        particle_intensities_list=particle_intensities_list,
        image_half_size=image_half_size,
        image_background_level=image_background_level,
        signal_to_noise_ratio=signal_to_noise_ratio,
        gradient_intensity=gradient_intensity,
        gradient_direction=gradient_direction)
    all_predictions = np.zeros([len(SNT_levels),len(networks),nbr_images_to_evaluate,3])

    for image_idx in range(len(all_images)):
        image_set = all_images[image_idx]
        for network_idx in range(len(networks)):

            prediction = networks[network_idx].predict(image_set)
            all_predictions[image_idx,network_idx,:,:] = prediction
    return all_targets,all_predictions
def generate_images_and_targets_from_noise_levels(
    network,
    SNT_levels,
    nbr_images_to_evaluate=1000,
    translation_distance = 1,
    particle_radius_list=lambda : uniform(1.5, 3, 1),
    particle_bessel_orders_list=lambda : [[randint(1, 3),], ],
    particle_intensities_list=lambda : [[choice([-1, 1]) * uniform(.2, .6, 1), ], ],
    image_half_size=lambda : 25,
    image_background_level=lambda : uniform(.2, .8),
    signal_to_noise_ratio=lambda : uniform(10, 100),
    gradient_intensity=lambda : uniform(0, 1),
    gradient_direction=lambda : uniform(-pi, pi)):
    """
    Generates images of different noise levels and corresponding targets
    """
    translation_distance = 1
    particle_center_x_list = lambda : normal(0, translation_distance, translation_distance)
    particle_center_y_list = lambda : normal(0, translation_distance, translation_distance)
    image_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]

    #all_predictions = []
    all_targets = []
    all_images = []

    for SNT_level in SNT_levels:

        ### Create an image generator
        image_parameters_function = lambda : deeptrack.get_image_parameters(
           particle_center_x_list=particle_center_x_list,
           particle_center_y_list=particle_center_y_list,
           particle_radius_list=particle_radius_list,
           particle_bessel_orders_list=particle_bessel_orders_list,
           particle_intensities_list=particle_intensities_list,
           image_half_size=image_half_size,
           image_background_level=image_background_level,
           signal_to_noise_ratio= lambda: SNT_level,
           gradient_intensity=gradient_intensity,
           gradient_direction= gradient_direction)
        image_generator = lambda : deeptrack.get_image_generator(image_parameters_function)

        ### Genereate images for evaluation
        images, targets = deeptrack.get_images_and_targets(image_generator,nbr_images_to_evaluate,image_shape=image_shape)
        all_targets.append(targets)
        all_images.append(images)

    return all_targets,all_images

def evaluate_gradient_levels(
    network,
    gradient_levels,
    nbr_images_to_evaluate=1000,
    particle_center_x_list = lambda : normal(0, 1, 1),
    particle_center_y_list=lambda : normal(0, 1, 1),
    particle_radius_list=lambda : uniform(1.5, 3, 1),
    particle_bessel_orders_list=lambda : [[randint(1, 3),], ],
    particle_intensities_list=lambda : [[choice([-1, 1]) * uniform(.2, .6, 1), ], ],
    image_half_size=lambda : 25,
    image_background_level=lambda : uniform(.2, .8),
    signal_to_noise_ratio=lambda : uniform(10, 100),
    gradient_intensity=lambda : uniform(0, 1),
    gradient_direction=lambda : uniform(-pi, pi)):
    """
    Function for evaluating performance of network for different noise levels
    Inputs: network to be evaluated, noise levels to evaluate on and parameters of the image generator(other than noise level)
        Number of images to evaluate on
    Outputs: Loss,mae and mse for the various noise levels
    """

    image_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]

    losses = []
    MSEs = []
    MAEs = []

    for gradient_level in gradient_levels:

        ### Create an image generator
        image_parameters_function = lambda : deeptrack.get_image_parameters(
           particle_center_x_list=particle_center_x_list,
           particle_center_y_list=particle_center_y_list,
           particle_radius_list=particle_radius_list,
           particle_bessel_orders_list=particle_bessel_orders_list,
           particle_intensities_list=particle_intensities_list,
           image_half_size=image_half_size,
           image_background_level=image_background_level,
           signal_to_noise_ratio= signal_to_noise_ratio,
           gradient_intensity=lambda:gradient_level,
           gradient_direction= gradient_direction)
        image_generator = lambda : deeptrack.get_image_generator(image_parameters_function)

        ### Genereate images for evaluation
        images, targets = deeptrack.get_images_and_targets(image_generator,nbr_images_to_evaluate,image_shape=image_shape)
        loss,mse,mae= network.evaluate(images,targets)

        ### Fix units and append to final results
        mse *= image_half_size()**2 #what happens if it is not a function? Silly error?
        mae *= image_half_size()
        losses.append(loss)
        MSEs.append(mse)
        MAEs.append(mae)
    return losses,MSEs,MAEs
