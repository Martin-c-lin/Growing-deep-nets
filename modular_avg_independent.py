def independent_avg_modular_deeptrack_L1(# bad name
    layer_size,
    train_generator,
    input_shape=(51,51,1),
    output_shape=3,
    nbr_nodes_added=1,
    sample_sizes=(8, 32, 128, 512, 1024),
    iteration_numbers=(401, 301, 201, 101, 51),
    verbose=0.01,
    save_networks=False,
    mp_training=False, # use multiprocessing training
    translation_distance=5, # parameters for multiprocessing training
    SN_limits=[10,100], # parameters for multiprocessing training
    model_path="", # determines the type of layer used for combining the
     # predictions. Addition and average only options at the moment
    ):
    """
    First layer in the modular averaging architecture where each layer is
    trained independetly of previous ones.
    Inputs:
        layer_size - size of first layer in model
        train_generator - generator for images
        output_shape - number of outputs from network, 3 for normal deeptrack
        nbr_nodes_added - number of nodes to add at a time
        sample_size - same as deeptrack
        iteration_numbers - same as deeptrack
        save_networks - if networks are to be saved automatically once training is finished
        mp_training - use multiprocessing to speed up training. Not available for custom
        image generators as supplied by train_generator, uses FBFs own image generator.
        translation_distance - parameter for mp_training image generator, determines
        the area the particle is allowed to appear in
        SN_limits -
    Outputs:
    TODO - Implement weight decay in later layers and lowering learning rate
    """
    import deeptrack
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    from feature_by_feature import freeze_all_layers
    #import deeptrackelli_mod_mproc as multiprocess_training # Needed for fast parallelized image generator

    input_tensor = Input(input_shape)
    conv_list = [] # List of convolutional neruons in L1, needed for subsequent layers
    flattened_list = [] # List of flattened layers
    output_list = [] # List of the output layers

    # Loop thorugh the neoruns and add them one by one
    for i in range(round(layer_size/nbr_nodes_added)):
        next_node = layers.Conv2D(nbr_nodes_added,(3,3),activation='relu')(input_tensor)
        next_node = layers.MaxPooling2D((2,2))(next_node)
        if(i==0):
            # i = 0 special case. No addition needed
            next_flattened = layers.Flatten()(next_node)
            next_output = layers.Dense(3)(next_flattened)
            final_output = next_output
            output_list.append(next_output)
            flattened_list.append(next_flattened)
        else:
            # Construct the next output node
            next_flattened = layers.Flatten()(next_node)
            flattened_list.append(next_flattened)
            next_output = layers.Concatenate(axis=-1)(flattened_list)
            next_output = layers.Dense(3)(next_output)
            output_list.append(next_output)
            # Construct and compile network
        network = models.Model(input_tensor,next_output)
        network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
        network.summary()

        # Train and freeze layers in network
        if mp_training:
            deeptrack.train_deep_learning_network_mp(
                network,
                sample_sizes = sample_sizes,
                iteration_numbers = iteration_numbers,
                verbose=verbose,
                SN_limits=SN_limits,
                translation_distance=translation_distance,
                )
        else:
            deeptrack.train_deep_learning_network(
                network,
                train_generator,
                sample_sizes = sample_sizes,
                iteration_numbers = iteration_numbers,
                verbose=verbose)

        freeze_all_layers(network)
        conv_list.append(next_node)
        if(save_networks):
            network.save(model_path+"L1_"+str((i+1)*nbr_nodes_added)+"F.h5")
    # Create final output using all the output layers and averaging them
    avg_out = layers.average(output_list)
    network = models.Model(input_tensor,avg_out)
    network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
    print('final network architecture')
    network.summary()
    if(save_networks):
        network.save(model_path+"final_L"+str(1)+"_F.h5")
    return network,conv_list,output_list,flattened_list,input_tensor
def independent_avg_modular_deeptrack_new_layer(
    layer_size, # Total number of nodes in layer
    train_generator,
    conv_list,
    output_list,
    input_tensor,
    layer_no=2,
    nbr_nodes_added=1,
    sample_sizes=(8, 32, 128, 512, 1024),
    iteration_numbers=(401, 301, 201, 101, 51),
    verbose=0.01,
    mp_training = False, # use multiprocessing training?
    translation_distance=5, # parameters for multiprocessing training
    SN_limits=[10,100], # parameters for multiprocessing training
    save_networks=False,
    model_path="",
    layer_type='conv'):
    """
    Adds a new layer to the modular averaging architecture where each output is
    trained independently
    """
    import deeptrack
    from keras import  models,layers
    from keras.models import Model
    from feature_by_feature import freeze_all_layers
    new_layer_node_list = [] # Convolutions in the new layer
    new_layer_flattened_list = [] # flattened output from new layer

    # Create input tenosr for new node
    if len(conv_list)>1:
        new_layer_input = layers.Concatenate()(conv_list)
    else:
        new_layer_input = conv_list[0]

    # If next layer is dense then we (probably) need to flatten previous input
    if layer_type=='dense':
        import keras.backend as K
        # Check dimension of previous output to see if Flatten layer is neede.
        prev_out_size = K.shape(new_layer_input).shape
        if(prev_out_size[0]>2):
            new_layer_input = layers.Flatten()(new_layer_input)

    # Add all the new nodes and train network in between
    for i in range(round(layer_size/nbr_nodes_added)):
        if layer_type=='dense':
            next_node = layers.Dense(nbr_nodes_added,activation='relu')(new_layer_input)
            next_flattened = next_node
        else:
            next_node = layers.Conv2D(nbr_nodes_added,(3,3),activation='relu')(new_layer_input)
            next_node = layers.MaxPooling2D((2,2))(next_node)
            next_flattened = layers.Flatten()(next_node)
        new_layer_flattened_list.append(next_flattened)
        if(i==0):
            # i = 0 special case. No concatenation needed
            next_output = layers.Dense(3)(next_flattened) # Different for i==0
        else:
            next_output = layers.Concatenate(axis=-1)(new_layer_flattened_list)
            next_output = layers.Dense(3)(next_output)

        # Construct and compile network
        network = models.Model(input_tensor,next_output)
        network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
        network.summary()
        output_list.append(next_output)

        # Train and freeze layers in network

        if mp_training:
            deeptrack.train_deep_learning_network_mp(
                network,
                sample_sizes = sample_sizes,
                iteration_numbers = iteration_numbers,
                verbose=verbose,
                SN_limits=SN_limits,
                translation_distance=translation_distance,
                )
        else:
            deeptrack.train_deep_learning_network(
                network,
                train_generator,
                sample_sizes = sample_sizes,
                iteration_numbers = iteration_numbers,
                verbose=verbose)

        freeze_all_layers(network)
        new_layer_node_list.append(next_node)
        if(save_networks):
            network.save(model_path+"L"+str(layer_no)+"_"+str((i+1)*nbr_nodes_added)+"F.h5")
    avg_out = layers.average(output_list)
    network = models.Model(input_tensor,avg_out)
    network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
    print('final network architecture')
    network.summary()
    if(save_networks):
        network.save(model_path+"final_L"+str(layer_no)+"_F.h5")
    # IF dense statement needed
    return network,new_layer_node_list,output_list,new_layer_flattened_list
def build_full_independent_modular_avg_network(
    layer_sizes,
    train_generator,
    layer_types,
    input_shape=(51,51,1),
    output_shape=3,
    nbr_nodes_added=1,
    sample_sizes=(8, 32, 128, 512, 1024),
    iteration_numbers=(401, 301, 201, 101, 51),
    verbose=0.01,
    save_networks=True,
    mp_training = False, # use multiprocessing training
    translation_distance=5, # parameters for multiprocessing training
    SN_limits=[10,100], # parameters for multiprocessing training
    model_path="",
    ):
    """
    Inputs:
        layer_sizes - nbr of nodes in each layer
        layer_types - types of layers to use #TODO improve this one
    Outputs:
        network - A finihsed modular network which is trained.
    """
    if len(layer_sizes)<1:
        print("Too few layer sizes given")
        return 0
    network,node_list,output_list,flattened_list,input_tensor=independent_avg_modular_deeptrack_L1(
        layer_size=layer_sizes[0],
        train_generator=train_generator,
        input_shape=input_shape,
        output_shape=output_shape,
        nbr_nodes_added=nbr_nodes_added,
        sample_sizes=sample_sizes,
        iteration_numbers=iteration_numbers,
        verbose=verbose,
        save_networks=save_networks,
        mp_training=mp_training, # use multiprocessing training
        translation_distance=translation_distance, # parameters for multiprocessing training
        SN_limits=SN_limits, # parameters for multiprocessing training
        model_path=model_path,
    )

    for i in range(len(layer_sizes)-1):
        idx = i+1
        network,node_list,output_list,new_layer_flattened_list = independent_avg_modular_deeptrack_new_layer(
            layer_size=layer_sizes[idx], # Total number of nodes in layer
            train_generator=train_generator,
            conv_list=node_list,
            output_list=output_list,
            input_tensor=input_tensor,
            layer_no=idx+1,
            nbr_nodes_added=nbr_nodes_added,
            sample_sizes=sample_sizes,
            iteration_numbers=iteration_numbers,
            verbose=verbose,
            mp_training = mp_training, # use multiprocessing training?
            translation_distance=translation_distance, # parameters for multiprocessing training
            SN_limits=SN_limits, # parameters for multiprocessing training
            save_networks=save_networks,
            model_path=model_path,
            layer_type=layer_types[idx])
    return network
