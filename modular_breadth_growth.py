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
    transfered directly.
    """
    from keras.layers import Concatenate
    concat_output_list = []
    for n in range(max(len(list_A),len(list_B))):
        if n<len(list_A) and n<len(list_B):
            tmp_concat = Concatenate(axis=-1)([list_A[n],list_B[n]])
            concat_output_list.append(tmp_concat)
        elif n>=len(list_A):
            concat_output_list.append(list_B[n])
        else:
            concat_output_list.append(list_A[n])
    return concat_output_list
def modular_breadth_network_growth(input_tensor,old_conv_layers,old_dense_layers,final_output,conv_layers_sizes,dense_layers_sizes):
    """

    """
    import deeptrack
    from keras import layers,models,Input

    conv_layers = []
    dense_layers = []

    # Add the convolutional layers
    if(len(conv_layers_sizes)>0):
        for i in range(len(conv_layers_sizes)):
            if i==0:# connect to input tensor
                new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu')(input_tensor)

            else:# connect to previous layer
                if(i<=len(old_conv_layers)): # Connect to previous network
                    print(i)
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

    # Add output layer
    if len(dense_layers_sizes)>0:
        final_output = layers.Dense(3)(new_dense_layer)
    else:
        final_output = layers.Dense(3)(conv_output)

    # Combine into single model
    network = models.Model(input_tensor,final_output)
    return network,input_tensor,conv_output_list,dense_output_list,final_output
def freeze_all_layers(model):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
