from multiprocessing import Pool,cpu_count

td = 5 # translation_distance
def f(x):
    import numpy as np
    import sys
    sys.path.insert(0,"C:/Users/Simulator/Desktop/Martin Selin/DeepTrack 1.0/") #
    import feature_by_feature as fbf
    image_generator = fbf.get_default_image_generator_deeptrack(td) # change for various parameters
    #print(x)
    targets = np.zeros((x,3))
    images = np.zeros((x,51,51))
    half_image_size = 25#(image_shape[0] - 1) / 2 # Ugly I know

    for image_number, image, image_parameters in image_generator():
        if image_number>=x:
            break
        images[image_number] = image

        particle_center_x = image_parameters['Particle Center X List'][0]
        particle_center_y = image_parameters['Particle Center Y List'][0]
        targets[image_number] = [particle_center_x / half_image_size,
                                 particle_center_y / half_image_size,
                                 (particle_center_x**2 + particle_center_y**2)**.5 / half_image_size]
    #print(targets.shape)

    return images,targets
def get_images_mp(sample_size_total):
    """
    Function for creating images in a parallized way. Currently only works for specific image generators...
    """
    from multiprocessing import Pool,cpu_count

    if __name__ == '__main__' or __name__=='mp_images_new':

        import numpy as np
        import time

        start = time.clock()

        nbr_images = []
        nbr_workers = cpu_count()
        sample_size = sample_size_total//nbr_workers
        for i in range(nbr_workers):
            nbr_images.append(sample_size)
            if(i<sample_size_total%nbr_workers):
                nbr_images[-1]+=1
        with Pool(processes=cpu_count()) as pool:

            tmp = pool.map(f,nbr_images ) # (1000,1000,1000,1000)

        final_images = np.zeros((sample_size_total,51,51,1))
        final_targets = np.zeros((sample_size_total,3))
        img_idx = 0
        for worker_no in range(nbr_workers):
            #print(len(tmp[worker_no][0][:]))
            for image_no in range(len(tmp[worker_no][0][:])):

                final_images[img_idx] = np.reshape(tmp[worker_no][0][image_no],(51,51,1))
                final_targets[img_idx] = tmp[worker_no][1][image_no]
                img_idx+=1
        #print(' time ', time.clock()-start,'s')
        return final_images,final_targets
        # Verify that it works
