def modular_breadth_network_start(conv_layers_sizes,dense_layers_sizes,input_shape=(51,51,1)):
    """
    Function which creates the firs part of a breadthwise grown network
    """
    import deeptrack
    from keras import layers,models,Input

    input_tensor = Input(input_shape)

    conv_layers = []
    dense_layers = []

    # Add the convolutional layers
    if(len(conv_layers_sizes)>0):
        for i in range(len(conv_layers_sizes)):
            if i==0:# connect to input tensor
                new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu')(input_tensor)
            else:# connect to previous layer
                new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu')(new_conv_layer)
            new_conv_layer = layers.MaxPooling2D((2,2))(new_conv_layer)# add pooling
            conv_layers.append(new_conv_layer)
        conv_output = layers.Flatten()(new_conv_layer)
    else:
        # No convolutional layers in model
        conv_output = layers.Flatten()(input_tensor)

    # Add the dense layers
    for i in range(len(dense_layers_sizes)):
        if i==0:
            new_dense_layer = layers.Dense(dense_layers_sizes[i],activation='relu')(conv_output)
        else:
            new_dense_layer = layers.Dense(dense_layers_sizes[i],activation='relu')(new_dense_layer)
        dense_layers.append(new_dense_layer)

    # Add output layer
    if len(dense_layers_sizes)>0:
        final_output = layers.Dense(3)(new_dense_layer)
    else:
        final_output = layers.Dense(3)(conv_output)

    # Combine into single model
    network = models.Model(input_tensor,final_output)

    return network,input_tensor,conv_layers,dense_layers,final_output
def concatenate_layer_lists(list_A,list_B):
    """
    Concatenates the layers in two lists and puts the result in a new list.
    If one list is longer than the other than the extra elements of that list are
    transfered directly. If either contain None elements these are skipped.
    """
    from keras.layers import Concatenate
    concat_output_list = []
    # Deal with empty layers in beginning?
    for n in range(max(len(list_A),len(list_B))):
        if n<len(list_A) and n<len(list_B):
            if(list_A[n]!=None and list_B[n]!=None):
                tmp_concat = Concatenate(axis=-1)([list_A[n],list_B[n]])
            elif list_A[n]==None:
                tmp_concat = list_B[n]
            else:
                tmp_concat = list_A[n]
            concat_output_list.append(tmp_concat)
        elif n>=len(list_A):
            if list_B[n] ==None:
                return concat_output_list
            concat_output_list.append(list_B[n])
        else:
            concat_output_list.append(list_A[n])
            if list_A[n] ==None:
                return concat_output_list
    return concat_output_list
def residual_connection(input_tensor,output_dims):
    """
    Reshapes input tensor to be of ouput_dims dimensions but with more layers.
    Ex transforms 51x51x1 image to 24x24x5 and pads the extra with zeros.
    """
    from keras import layers
    input_size = 1
    input_dims = input_tensor.shape[1:3]
    for input_dim in input_dims:
        input_size *= input_dim

    output_size = 1
    for output_dim in output_dims:
        output_size *= output_dim
    nbr_layers_ouput = int(int(input_size)/int(output_size))+1 # nbr layers we will need
    padding_size = int(nbr_layers_ouput*output_size - input_size)

    flattened = layers.Reshape((input_dims[0]*input_dims[1],1))(input_tensor)
    padded = layers.ZeroPadding1D((0,padding_size))(flattened)

    output_shape = (int(output_dims[0]),int(output_dims[1]),nbr_layers_ouput)
    tmp = layers.Reshape(output_shape)(padded)
    return tmp
def modular_breadth_network_growth(input_tensor,old_conv_layers,old_dense_layers,conv_layers_sizes,dense_layers_sizes,residual_connections=False):
    """
    Function for growing a modular bredth network by adding another network next
    to it.

    """
    import deeptrack
    from keras import layers,models,Input

    conv_layers = []
    dense_layers = []
    first_nonzero_idx = 0
    # Add the convolutional layers
    if(len(conv_layers_sizes)>0):
        for i in range(len(conv_layers_sizes)):
            if(conv_layers_sizes[i]<=0 and first_nonzero_idx==i):
                first_nonzero_idx+=1
                conv_layers.append(None)
            else:
                if i==0:# connect to input tensor
                        new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu')(input_tensor)

                elif i==first_nonzero_idx: # Connect only to old layers and perhaps a residual connection

                    if residual_connections:
                        # Create residual connection to input tensor
                        output_dims = old_conv_layers[i-1].shape[1:3]
                        print(old_conv_layers[i-1].shape)
                        res_connection = residual_connection(input_tensor,output_dims)
                        new_input = layers.Concatenate(axis=-1)([res_connection,old_conv_layers[i-1]])
                        new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu')(new_input)
                    else:
                        new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu')(old_conv_layers[i-1])

                else:# connect to previous layer and old layer
                    if(i<=len(old_conv_layers)): # Connect to previous network
                        new_conv_layer = layers.Concatenate(axis=-1)([new_conv_layer,old_conv_layers[i-1]])
                    new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu')(new_conv_layer)

                new_conv_layer = layers.MaxPooling2D((2,2))(new_conv_layer)# add pooling
                conv_layers.append(new_conv_layer)
        conv_output = layers.Flatten()(new_conv_layer)
    else:
        # No convolutional layers in model
        conv_output = layers.Flatten()(input_tensor)

    # Need to transfer stuff into the new lists... so model can grow beyond two additions
    conv_output_list = concatenate_layer_lists(conv_layers,old_conv_layers)
    # Assumes previous conv_output not trivially connected to the input_tensor

    # Can become odd here if the two networks being combined have different numbers of convolutional layers
    # which is why we do like this
    if(len(old_conv_layers)>=len(conv_layers)):
        tmp_flatten = layers.Flatten()(old_conv_layers[-1])
        conv_output = layers.Concatenate(axis=-1)([conv_output,tmp_flatten])
    # Add the dense layers
    for i in range(len(dense_layers_sizes)):
        if i==0:
            new_dense_layer = layers.Dense(dense_layers_sizes[i],activation='relu')(conv_output)
        else:
            if(i<=len(old_dense_layers)): # Connect to previous network
                new_dense_layer = layers.Concatenate(axis=-1)([new_dense_layer,old_dense_layers[i-1]])
            new_dense_layer = layers.Dense(dense_layers_sizes[i],activation='relu')(new_dense_layer)
        dense_layers.append(new_dense_layer)

    dense_output_list = concatenate_layer_lists(dense_layers,old_dense_layers)

    # Previosly not added as intended! only connected to second to last output layer
    # Add output layer
    if len(dense_layers_sizes)>0:
        final_output = layers.Dense(3)(dense_output_list[-1])
    else:
        final_output = layers.Dense(3)(conv_output)

    # Combine into single model
    network = models.Model(input_tensor,final_output)
    return network,conv_output_list,dense_output_list,final_output
def freeze_all_layers(model):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
def build_modular_breadth_model(
        conv_layers_sizes,
        dense_layers_sizes,
        sample_sizes=[8,32,128,512,1024],
        iteration_numbers=[4000,3000,2000,1000,500],
        SN_limits=[10,100],
        translation_distance=1,
        verbose=0.01,
        model_path="",
        save_networks=True,
        residual_connections=False
        ):
    """
    residual_connections - Used to connect the first non-zero convolutional
        layer directly also to input for expansion layers.
    """
    import deeptrack
    import evaluate_deeptrack_performance as edp
    import numpy as np
    # Create first network
    network,input_tensor,conv_layers_list,dense_layers_list,final_output = modular_breadth_network_start(conv_layers_sizes[0],dense_layers_sizes[0])

    # compile and verify appearence
    network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
    network.summary()
    # Use default parameters for evaluation
    SNR_evaluation_levels = [5,10,20,30,50,100]
    nbr_images_to_evaluate = 1000
    # Train and freeze layers in network
    deeptrack.train_deep_learning_network_mp(
        network,
        sample_sizes = sample_sizes,
        iteration_numbers = iteration_numbers,
        verbose=verbose,
        SN_limits=SN_limits,
        translation_distance=translation_distance,
        )
    freeze_all_layers(network)
    if(save_networks):
        network.save(model_path+"network_no"+str(0)+".h5")
    for i in range(len(conv_layers_sizes)-1):
        model_idx = i+1
        # Grow network
        network,conv_layers_list,dense_layers_list,final_output = modular_breadth_network_growth(
            input_tensor,
            conv_layers_list,
            dense_layers_list,
            conv_layers_sizes[model_idx],
            dense_layers_sizes[model_idx],
            residual_connections=residual_connections)

        network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
        network.summary()

        # Train and freeze layers in network
        deeptrack.train_deep_learning_network_mp(
            network,
            sample_sizes = sample_sizes,
            iteration_numbers = iteration_numbers,
            verbose=verbose,
            SN_limits=SN_limits,
            translation_distance=translation_distance,
            )
        freeze_all_layers(network)
        # Save network
        if save_networks:
            # don't work with the residual connections
            network.save(model_path+"network_no"+str(model_idx)+".h5")
        # Evaluate performance on the fly if residual_connections are used
        if residual_connections:
            res = edp.evaluate_noise_levels(
                network,
                SNR_evaluation_levels,
                nbr_images_to_evaluate=nbr_images_to_evaluate,
                )
            np.save(model_path+"model_no"+str(model_idx),res)
    return network
