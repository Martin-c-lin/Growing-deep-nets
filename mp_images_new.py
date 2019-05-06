def get_image_generator_movies():
    from numpy.random import randint, uniform, normal, choice
    from math import pi
    import deeptrack
    image_parameters_function = lambda : deeptrack.get_image_parameters(
        particle_center_x_list=lambda : [normal(0 ,2, 1), ],
        particle_center_y_list=lambda : [normal(0 ,2, 1), ],
        particle_radius_list=lambda : uniform(2, 3, 1),
        particle_bessel_orders_list=lambda : [[1, 2], ],
        particle_intensities_list=lambda : [[uniform(.7, .9, 1), -uniform(.2, .3, 1)], ],
        image_half_size=lambda : 25,
        image_background_level=lambda : uniform(.2, .5),
        signal_to_noise_ratio=lambda : uniform(5, 100),
        gradient_intensity=lambda : uniform(0, .8),
        gradient_direction=lambda : uniform(-pi, pi))

    ### Define image generator
    image_generator = lambda : deeptrack.get_image_generator(image_parameters_function)
    return image_generator

def get_default_image_generator_deeptrack(translation_distance = 5,SN_limits=[10,100],radius_limits=[1.5,3]):
    """
    Returns a default image genereator for deeptrack. Allows for some parameter tweaking.
    """
    import deeptrack
    from numpy.random import randint, uniform, normal, choice
    from math import pi

    image_parameters_function = lambda : deeptrack.get_image_parameters(
       particle_center_x_list=lambda : normal(0, translation_distance, translation_distance),
       particle_center_y_list=lambda : normal(0, translation_distance, translation_distance),
       particle_radius_list=lambda : uniform(radius_limits[0],radius_limits[1], 1),
       particle_bessel_orders_list=lambda : [[randint(1, 3),], ],
       particle_intensities_list=lambda : [[choice([-1, 1]) * uniform(.2, .6, 1), ], ],
       image_half_size=lambda : 25,
       image_background_level=lambda : uniform(.2, .8),
       signal_to_noise_ratio=lambda : uniform(SN_limits[0], SN_limits[1]),
       gradient_intensity=lambda : uniform(0, 1),
       gradient_direction=lambda : uniform(-pi, pi),
       ellipsoidal_orientation=lambda : uniform(-pi,pi,1 ),
       ellipticity=lambda : 1)
    ### Define image generator
    image_generator = lambda : deeptrack.get_image_generator(image_parameters_function)
    return image_generator


def get_image_parameters_optimized_vesicles():
    ### Define image properties
    #%matplotlib inline
    from numpy.random import randint, uniform, normal, choice
    from math import pi
    import deeptrack

    particle_number = 4
    first_particle_range = 10
    other_particle_range = 25
    particle_distance = 10

    (particles_center_x, particles_center_y) = deeptrack.particle_positions(particle_number, first_particle_range, other_particle_range, particle_distance)
    image_parameters = {}

    image_parameters['Particle Center X List'] = particles_center_x
    image_parameters['Particle Center Y List'] = particles_center_y
    image_parameters['Particle Radius List'] = uniform(2, 3, particle_number)
    image_parameters['Particle Bessel Orders List'] = [[1, ],
                                                       [1, ],
                                                       [1, ],
                                                       [1, ]]
    image_parameters['Particle Intensities List'] = [[uniform(0.4, 0.7, 1), ],
                                                     [uniform(0.4, 0.7, 1), ],
                                                     [uniform(0.4, 0.7, 1), ],
                                                     [uniform(0.4, 0.7, 1), ]]
    image_parameters['Image Half-Size'] = 25
    image_parameters['Image Background Level'] = uniform(.2, .4)
    image_parameters['Signal to Noise Ratio'] = uniform(2, 8)
    image_parameters['Gradient Intensity'] = uniform(0, 0.2)
    image_parameters['Gradient Direction'] = uniform(-pi, pi)
    image_parameters['Ellipsoid Orientation'] = uniform(-pi, pi, particle_number)
    image_parameters['Ellipticity'] = 1

    return lambda:image_parameters
def get_vesicle_image_generator():
    import deeptrack

    image_generator = lambda : deeptrack.get_image_generator(get_image_parameters_optimized_vesicles())
    return image_generator

def f(x,SN_limits=[10,100],translation_distance=5,radius_limits=[1.5,3],parameter_function=0):
    import numpy as np

    if parameter_function==0:
        image_generator = get_default_image_generator_deeptrack(translation_distance=translation_distance,SN_limits=SN_limits,radius_limits=radius_limits) # change for various parameters
    elif parameter_function==1:
        image_generator = get_vesicle_image_generator()
    elif parameter_function==2:
        image_generator = get_image_generator_movies()
    targets = np.zeros((x,3))

    images = np.zeros((x,51,51))
    half_image_size = 25#(image_shape[0] - 1) / 2 # Ugly I know

    for image_number, image, image_parameters in image_generator():
        if image_number>=x:
            break
        images[image_number] = image
        # Do same preprocessing as in deeptrack
        particle_center_x = image_parameters['Particle Center X List'][0]
        particle_center_y = image_parameters['Particle Center Y List'][0]
        targets[image_number] = [particle_center_x / half_image_size,
                                 particle_center_y / half_image_size,
                                 (particle_center_x**2 + particle_center_y**2)**.5 / half_image_size]

    return images,targets
def get_images_mp(sample_size_total,SN_limits=[10,100],translation_distance=1,radius_limits=[1.5,3],parameter_function=0):
    """
    Function for creating images in a parallized way. Currently only works for specific image generators...
    Must be protected by a if __name__ == '__main__': or similar to work properly!
    """


    if __name__ == '__main__' or __name__=='mp_images_new':
        import numpy as np

        from multiprocessing import Pool,cpu_count
        from functools import partial

        nbr_images = []
        nbr_workers = cpu_count()
        sample_size = sample_size_total//nbr_workers

        # Avoid sending the extra paramters multiple times by creating partial

        tmp = partial(f,SN_limits=SN_limits,translation_distance=translation_distance,radius_limits=radius_limits,parameter_function=parameter_function)

        # Do some load balancing

        for i in range(nbr_workers):
            nbr_images.append(sample_size)
            if(i<sample_size_total%nbr_workers):
                nbr_images[-1]+=1

        # Use a pool to efficiently generate lots of images

        with Pool(processes=cpu_count()) as pool:
            tmp = pool.map(tmp,nbr_images)
            pool.close()
            pool.join()

        # Reformat the results from the pool map to be the right shape

        final_images = np.zeros((sample_size_total,51,51,1))
        final_targets = np.zeros((sample_size_total,3))
        img_idx = 0
        for worker_no in range(nbr_workers):
            for image_no in range(len(tmp[worker_no][0][:])):

                final_images[img_idx] = np.reshape(tmp[worker_no][0][image_no],(51,51,1))
                final_targets[img_idx] = tmp[worker_no][1][image_no]
                img_idx+=1
        return final_images,final_targets
