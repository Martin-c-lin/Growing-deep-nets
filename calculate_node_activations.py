if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn') # requires restart of kernel when run in Jupyterlab
    import number_of_node_activations as nbr
    import mp_images_new as mp
    from keras.models import load_model
    import numpy as np

    nbr_images = 10000 # 10 000 should be default
    images,targets = mp.get_images_mp(nbr_images)
    layer_name = 'Dense_1_2'
    layer_indices = [1,3,5,8,9]
    tmp = []
    run_numbers = [0,1,2,3,4,6,7,8,9,10]
    for n in run_numbers:
        for i in range(8):
            nbr_nodes = (i+1)*16
            model = load_model('normal_comp/normal_deeptrack_run'+str(n)+'base'+str(nbr_nodes)+'network_no0.h5')
            model.summary()
            nonzero_model_fractions = []
            for layer_index in layer_indices:
                nonzero_fractions,activation_binary_list,predictions = nbr.evaluate_layer_activations(model,images,int(layer_index))
                nonzero_model_fractions.append(np.mean(nonzero_fractions))
                print(layer_index,nonzero_fractions)

            print(nonzero_model_fractions)
            tmp.append(nonzero_model_fractions)
    np.save('normal_models'+str(len(run_numbers))+'_activation_fractions',tmp)
