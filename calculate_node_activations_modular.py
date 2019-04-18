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
    layer_names = ['Conv_1_1','Conv_2_2','Conv_3_3','Dense_4_1','Dense_5_2',
        'Conv_6_1','Conv_6_2','Conv_6_3','Dense_6_1','Dense_6_2',
        'Conv_7_1','Conv_7_2','Conv_7_3','Dense_7_1','Dense_7_2',
        'Conv_8_1','Conv_8_2','Conv_8_3','Dense_8_1','Dense_8_2',
        'Conv_9_1','Conv_9_2','Conv_9_3','Dense_9_1','Dense_9_2',
        'Conv_10_1','Conv_10_2','Conv_10_3','Dense_10_1','Dense_10_2',
        'Conv_11_1','Conv_11_2','Conv_11_3','Dense_11_1','Dense_11_2',
        'Conv_12_1','Conv_12_2','Conv_12_3','Dense_12_1','Dense_12_2',
        'Conv_13_1','Conv_13_2','Conv_13_3','Dense_13_1','Dense_13_2',]
    tmp = []
    model = load_model('LBL_breadth_modular/run4_32topnetwork_no12.h5')
    model.summary()
    for layer_name in layer_names:
        layer_index = nbr.get_layer_idx(model,layer_name)
        #nonzero_model_fractions = []
        nonzero_fractions,activation_binary_list,predictions = nbr.evaluate_layer_activations(model,images,int(layer_index))
        #nonzero_model_fractions.append(np.mean(nonzero_fractions))
        print(layer_name,layer_index,np.mean(nonzero_fractions))

        tmp.append(np.mean(nonzero_fractions))
    np.save('LBL_model4'+str(0)+'_activation_fractions',tmp)
