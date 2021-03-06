def resize_images(images,new_size=24):
    nbr_images = len(images)
    resized = np.zeros((nbr_images,new_size,new_size,4))
    for i in range(new_size):
        for j in range(new_size):
            resized[:,i,j,0] = images[:,2+2*i,1+2*j] # might need to switch down
            # start index if padding usd in convolutions
            resized[:,i,j,1] = images[:,1+2*i,2+2*j]
            resized[:,i,j,2] = images[:,1+2*i,1+2*j]
            resized[:,i,j,3] = images[:,2+2*i,2+2*j]
    return resized
def reformat_images_res_net(images,new_sizes):
    """
    Function which reformats images to make them in a suitable resnet format,
    i.e variation of downsampling
    """
    import numpy as np
    image_length = len(images)
    print(new_sizes)
    res = [images]
    prev_images = images
    for i in range(len(new_sizes)):
        size = new_sizes[i]
        nbr_channels = np.power(4,i+1)
        prev_nbr_channels = np.power(4,i)
        final_images = np.zeros((image_length,size,size,nbr_channels))
        print(nbr_channels)
        for i in range(prev_nbr_channels):
            final_images[:,:,:,i*4:i*4+4] = resize_images(prev_images[:,:,:,i],size)
        prev_images = final_images
        res.append(final_images)
    return res
def modular_breadth_network_start(conv_layers_sizes,dense_layers_sizes,input_shape=(51,51,1),extra_pooling=0):
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
            layer_name = 'Conv_1_'+str(i+1)
            if i==0:# connect to input tensor
                new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu',name=layer_name)(input_tensor)
            else:# connect to previous layer
                new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu',name=layer_name)(new_conv_layer)
            new_conv_layer = layers.MaxPooling2D((2,2))(new_conv_layer)# add pooling
            conv_layers.append(new_conv_layer)
        if extra_pooling==1:
            # Add extra pooling to decrease number of parameters
            tmp = layers.MaxPooling2D((2,2))(new_conv_layer)
            conv_output = layers.Flatten()(tmp)
        elif extra_pooling==2:
            tmp = layers.Conv2D(int(conv_layers_sizes[-1]/2),(3,3),activation='relu',name='auxiliary_classifier_conv')(new_conv_layer)
            tmp = layers.MaxPooling2D((2,2),name='auxiliary_classifier_pooling')(tmp)
            conv_output = layers.Flatten()(tmp)
        else:
            conv_output = layers.Flatten()(new_conv_layer)

    else:
        # No convolutional layers in model
        conv_output = layers.Flatten()(input_tensor)
        print('Connected to input first model')

    # Add the dense layers
    for i in range(len(dense_layers_sizes)):
        layer_name = 'Dense_1_'+str(i+1)
        if i==0:
            new_dense_layer = layers.Dense(dense_layers_sizes[i],activation='relu',name=layer_name)(conv_output)
        else:
            new_dense_layer = layers.Dense(dense_layers_sizes[i],activation='relu',name=layer_name)(new_dense_layer)
        dense_layers.append(new_dense_layer)

    # Add output layer
    if len(dense_layers_sizes)>0:
        final_output = layers.Dense(3,name='Output_1')(new_dense_layer)
    else:
        final_output = layers.Dense(3,name='Output_1')(conv_output)

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
    #import keras.backend as K #for gather
    input_size = 1
    input_dims = input_tensor.shape[1:3]
    for input_dim in input_dims:
        input_size *= int(input_dim)

    output_size = 1
    for output_dim in output_dims:
        output_size *= int(output_dim)
    nbr_layers_ouput = int(int(input_size)/int(output_size))+1 # nbr layers we will need
    padding_size = int(nbr_layers_ouput*output_size - input_size)

    flattened = layers.Reshape((input_dims[0]*input_dims[1],1))(input_tensor)
    # use gather to make it nicer?
    padded = layers.ZeroPadding1D((0,padding_size))(flattened)

    output_shape = (int(output_dims[0]),int(output_dims[1]),nbr_layers_ouput)
    tmp = layers.Reshape(output_shape)(padded)
    return tmp
def get_new_res_weights(old_weights3x3,old_weights2x2):
    """
    """
    old_weights3x3[0][:] = 0
    old_weights3x3[1][0] = 0
    for i in range(len(old_weights3x3[0][0,0,:,0])):
        old_weights3x3[0][1,1,i,i] = 1 # not correct

    old_weights2x2[1][:] = 0
    old_weights2x2[0][:] = 0
    #for j in range(len(old_weights2x2[0][0,0,:,0])):
    nbr_inputs = len(old_weights2x2[0][0,0,:,0])
    for i in range(int(len(old_weights2x2[0][0,0,0,:])/4)):
            old_weights2x2[0][0,0,i,4*i+0]=1
            old_weights2x2[0][1,0,i,4*i+1]=1
            old_weights2x2[0][0,1,i,4*i+2]=1
            old_weights2x2[0][1,1,i,4*i+3]=1
    return old_weights3x3,old_weights2x2
def new_residual_connection(input_tensor,nbr_steps):
    """
    Surperior implementation for creating a new residual conneciton to input layer. Only for convolutioal
    layers. Skips "nbr_steps" layers.
    """
    from keras import models,layers
    import numpy as np

    res_layers = []
    last = input_tensor

    # Loop through all the layers and create residual connections,

    for i in range(nbr_steps):

        nbr_nodes = np.power(4,i) # nbr nodes to add in res connections

        first = layers.Conv2D(nbr_nodes,(3,3),name='residual_'+str(i))(last) # Naming needs fixing
        last = layers.Conv2D(4*nbr_nodes,(2,2),strides=2,name='residual_second_'+str(i))(first)# Naming needs fixing

        # Create model to easily acces weights
        tmp_model = models.Model(input_tensor,last)
        weights3x3 = tmp_model.layers[-2].get_weights()
        weights2x2 = tmp_model.layers[-1].get_weights()

        # get new weights for the residual connection
        weights3x3,weights2x2 = get_new_res_weights(weights3x3,weights2x2)
        tmp_model.layers[-2].set_weights(weights3x3)
        tmp_model.layers[-1].set_weights(weights2x2)

        # Make the residual connection non-trainable
        tmp_model.layers[-2].trainable = False
        tmp_model.layers[-1].trainable = False
        res_layers.append(tmp_model.layers[-1])

    return last
def extend_residual_connection(input_tensor,last_residual_connection,last_res_grade,nbr_steps_to_extend):
    """
    Function for extending an already existing residual connection.

    """
    from keras import models,layers
    import numpy as np
    res_layers = []
    last = last_residual_connection

    # Loop through all the layers and create residual connections,

    for i in range(nbr_steps_to_extend):

        nbr_nodes = np.power(4,i+last_res_grade) # nbr nodes to add in res connections

        first = layers.Conv2D(nbr_nodes,(3,3),name='residual_'+str(i)+'_'+str(last_res_grade))(last) # Naming needs fixing
        last = layers.Conv2D(4*nbr_nodes,(2,2),strides=2,name='residual_second_'+str(i)+'_'+str(last_res_grade))(first)# Naming needs fixing

        # Create model to easily acces weights
        tmp_model = models.Model(input_tensor,last)
        weights3x3 = tmp_model.layers[-2].get_weights()
        weights2x2 = tmp_model.layers[-1].get_weights()

        # get new weights for the residual connection
        weights3x3,weights2x2 = get_new_res_weights(weights3x3,weights2x2)
        tmp_model.layers[-2].set_weights(weights3x3)
        tmp_model.layers[-1].set_weights(weights2x2)

        # Make the residual connection non-trainable
        tmp_model.layers[-2].trainable = False
        tmp_model.layers[-1].trainable = False
        res_layers.append(tmp_model.layers[-1])

    return last
def modular_breadth_network_growth(input_tensor,old_conv_layers,old_dense_layers,conv_layers_sizes,dense_layers_sizes,residual_connections=False,nbr_residual_connections=0,last_residual_connection=None,model_idx=2):
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
            layer_name = 'Conv_'+str(model_idx)+'_'+str(i+1)
            if(conv_layers_sizes[i]<=0 and first_nonzero_idx==i):
                first_nonzero_idx+=1
                conv_layers.append(None)
            else:
                if i==0:# connect to input tensor
                        new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu',name=layer_name)(input_tensor)

                elif i==first_nonzero_idx: # Connect only to old layers and perhaps a residual connection

                    if residual_connections:
                        # Create residual connection to input tensor
                        output_dims = old_conv_layers[i-1].shape[1:3]

                        if nbr_residual_connections==0:
                            res_connection = new_residual_connection(input_tensor,i)
                        else:
                            res_connection = extend_residual_connection(input_tensor,last_residual_connection,last_res_grade=nbr_residual_connections,nbr_steps_to_extend=1)

                        nbr_residual_connections +=1
                        last_residual_connection = res_connection

                        new_input = layers.Concatenate(axis=-1)([res_connection,old_conv_layers[i-1]]) #need to fix so that we do not create new residual connections all the time
                        new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu',name=layer_name)(new_input)
                    else:
                        new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu',name=layer_name)(old_conv_layers[i-1])

                else:# connect to previous layer and old layer
                    if(i<=len(old_conv_layers)): # Connect to previous network
                        new_conv_layer = layers.Concatenate(axis=-1)([new_conv_layer,old_conv_layers[i-1]])
                    new_conv_layer = layers.Conv2D(conv_layers_sizes[i],(3,3),activation='relu',name=layer_name)(new_conv_layer)

                new_conv_layer = layers.MaxPooling2D((2,2))(new_conv_layer)# add pooling
                conv_layers.append(new_conv_layer)

        conv_output = layers.Flatten()(new_conv_layer) # gives an error if no conv layers are added
    else:
        if len(old_conv_layers)>0:
            conv_output = layers.Flatten()(old_conv_layers[-1])
        else:
            # No convolutional layers in model
            conv_output = layers.Flatten()(input_tensor)
            print('Connected to input')
    # Need to transfer stuff into the new lists... so model can grow beyond two additions
    conv_output_list = concatenate_layer_lists(conv_layers,old_conv_layers)
    # Assumes previous conv_output not trivially connected to the input_tensor

    # Can become odd here if the two networks being combined have different numbers of convolutional layers
    # which is why we do like this
    if(len(old_conv_layers)>=len(conv_layers) and len(conv_layers)>0): #Not correct when using residual connections
        tmp_flatten = layers.Flatten()(old_conv_layers[-1])
        conv_output = layers.Concatenate(axis=-1)([conv_output,tmp_flatten])
    # Add the dense layers
    first_nonzero_idx = 0

    for i in range(len(dense_layers_sizes)):
        layer_name = 'Dense_'+str(model_idx)+'_'+str(i+1)
        if dense_layers_sizes[i]<=0:
                dense_layers.append(None)
                first_nonzero_idx += 1
        else:
            if i==0:
                new_dense_layer = layers.Dense(dense_layers_sizes[i],activation='relu',name=layer_name)(conv_output)
            else:
                if(i<=len(old_dense_layers)): # Connect to previous network
                    if i==first_nonzero_idx:
                        if len(conv_layers_sizes)>=1:
                            new_dense_layer = layers.Concatenate(axis=-1)([conv_output,old_dense_layers[i-1]])
                        else:
                            new_dense_layer = old_dense_layers[i-1]
                    else:
                        new_dense_layer = layers.Concatenate(axis=-1)([new_dense_layer,old_dense_layers[i-1]])
                new_dense_layer = layers.Dense(dense_layers_sizes[i],activation='relu',name=layer_name)(new_dense_layer)
            dense_layers.append(new_dense_layer)

    dense_output_list = concatenate_layer_lists(dense_layers,old_dense_layers)

    # Previosly not added as intended! only connected to second to last output layer
    # Add output layer
    if len(dense_layers_sizes)>0:
        final_output = layers.Dense(3,name='Output_'+str(model_idx))(dense_output_list[-1])
    else:
        final_output = layers.Dense(3,name='Output_'+str(model_idx))(conv_output)

    # Combine into single model
    network = models.Model(input_tensor,final_output)
    return network,conv_output_list,dense_output_list,final_output,last_residual_connection,nbr_residual_connections
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
        residual_connections=False,
        train_network=True,
        parameter_function=0,
        extra_pooling=0,
        ):
    """
    residual_connections - Used to connect the first non-zero convolutional
        layer directly also to input for expansion layers.
    """
    import deeptrack
    import evaluate_deeptrack_performance as edp
    import numpy as np

    output_layers = [] # list of all output layers in model, used for training
    # all layers at once
    # Create first network
    network,input_tensor,conv_layers_list,dense_layers_list,final_output = modular_breadth_network_start(conv_layers_sizes[0],dense_layers_sizes[0],extra_pooling=extra_pooling)
    output_layers.append(final_output)
    # compile and verify appearence
    nbr_images_to_evaluate = 1000
    nbr_residual_connections = 0
    last_residual_connection = None
    if train_network:
        network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
        network.summary()
        # Use default parameters for evaluation
        # Train and freeze layers in network

        deeptrack.train_deep_learning_network_mp(
            network,
            sample_sizes = sample_sizes,
            iteration_numbers = iteration_numbers,
            verbose=verbose,
            SN_limits=SN_limits,
            translation_distance=translation_distance,
            parameter_function=parameter_function,
            )
        freeze_all_layers(network)
        if(save_networks):
            network.save(model_path+"network_no"+str(0)+".h5")
    for i in range(len(conv_layers_sizes)-1):
        model_idx = i+1
        # Grow network
        network,conv_layers_list,dense_layers_list,final_output,last_residual_connection,nbr_residual_connections = modular_breadth_network_growth(
            input_tensor,
            conv_layers_list,
            dense_layers_list,
            conv_layers_sizes[model_idx],
            dense_layers_sizes[model_idx],
            residual_connections=residual_connections,
            nbr_residual_connections=nbr_residual_connections,
            last_residual_connection=last_residual_connection,
            model_idx=model_idx+1
            )
        output_layers.append(final_output)

        if train_network:

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
                parameter_function=parameter_function,
                )
            freeze_all_layers(network) # migth complain about this, no worries
            network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

            # Save network
            if save_networks:
                network.save(model_path+"network_no"+str(model_idx)+".h5")
    if train_network:
        return network
    return network,output_layers,input_tensor

def simultaneus_training_network(conv_layers_sizes,
        dense_layers_sizes,
        sample_sizes=[8,32,128,512,1024],
        iteration_numbers=[4000,3000,2000,1000,500],
        SN_limits=[10,100],
        translation_distance=1,
        verbose=0.01,
        model_path="",
        save_networks=True,
        residual_connections=False):
    """
    Function for building a multi-output network and then train it all outputs
    at once.
    """
    from keras.models import Model
    import deeptrack

    network,output_layers,input_tensor = build_modular_breadth_model(
            conv_layers_sizes=conv_layers_sizes,
            dense_layers_sizes=dense_layers_sizes,
            sample_sizes=sample_sizes,
            iteration_numbers=iteration_numbers,
            SN_limits=SN_limits,
            translation_distance=translation_distance,
            verbose=verbose,
            model_path=model_path,
            save_networks=save_networks,
            residual_connections=residual_connections,
            train_network=False)
    single_network = Model(input_tensor,output_layers)
    losses = []
    # network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
    for i in range(len(output_layers)):
        losses.append('mse')
    single_network.summary()
    single_network.compile(optimizer='rmsprop', loss=losses, metrics=['mse', 'mae'])
    deeptrack.train_deep_learning_network_mp(
        single_network,
        sample_sizes = sample_sizes,
        iteration_numbers = iteration_numbers,
        verbose=verbose,
        SN_limits=SN_limits,
        translation_distance=translation_distance,
        nbr_outputs=len(output_layers)
        )
    if save_networks:
        single_network.save(model_path+'simultaneus_training_network.h5')
    return single_network
def ensamble_network_LBL(
        conv_layers_sizes,# 1d array
        dense_layers_sizes,# 1d array
        ensemble_size=2,
        sample_sizes=[8,32,128,512,1024],
        iteration_numbers=[4000,3000,2000,1000,500],
        SN_limits=[10,100],
        translation_distance=1,
        verbose=0.01,
        model_path="",
        save_networks=True,
        input_shape=(51,51,1),
        parameter_function=0):
    ### # TODO: ADD TRAINING LBL style
    # Save networks
    import deeptrack
    from keras import layers,models,Input
    import numpy as np


    input_tensor = Input(input_shape)

    pooled_layers = []
    old_pooled_layers = []

    # Add the convolutional layers
    if(len(conv_layers_sizes)<1):
        print("Error no convolutional layers")
        return 0
    for layer_no in range(len(conv_layers_sizes)):# range(1):
        old_pooled_layers = pooled_layers
        pooled_layers = []
        for i in range(ensemble_size):
            conv_name = "conv_layer_"+str(layer_no)+'_'+str(i)
            pooling_name = "pooling_layer_"+str(layer_no)+'_'+str(i)
            if layer_no==0:
                new_conv_layer = layers.Conv2D(conv_layers_sizes[layer_no],(3,3),
                    activation='relu',name=conv_name)(input_tensor)
            else:
                new_conv_layer = layers.Conv2D(conv_layers_sizes[layer_no],(3,3),
                    activation='relu',name=conv_name)(old_pooled_layers[i])

            new_pooling_layer = layers.MaxPooling2D((2,2),name=pooling_name)(new_conv_layer)
            pooled_layers.append(new_pooling_layer)

        # Add dense top, train the model and freeze all the layers. Then save if need be
        network,output_layers = Add_ensemble_output(input_tensor=input_tensor,
            top_layers=pooled_layers,ensemble_size=ensemble_size,convolutions=True)
        network.compile(optimizer='rmsprop',loss='mse', metrics=['mse', 'mae'])
        network.summary()
        deeptrack.train_deep_learning_network_mp(network,
            translation_distance=translation_distance,SN_limits = SN_limits,
            sample_sizes = sample_sizes,iteration_numbers = iteration_numbers,
            verbose=verbose,nbr_outputs=ensemble_size,parameter_function=parameter_function)
        freeze_all_layers(network)
        if(save_networks):
            network.save(model_path+"network_no"+str(layer_no)+".h5")

    # Adding flatten layers
    flattened_layers = []
    for i in range(ensemble_size):
            flattened_name = "flatten_layer_"+str(i)
            new_flattened = layers.Flatten(name=flattened_name)(pooled_layers[i])
            flattened_layers.append(new_flattened)

    # Add dense layers
    dense_layers = []
    old_dense_layers = []
    for layer_no in range(len(dense_layers_sizes)):
        old_dense_layers = dense_layers
        dense_layers = []
        for i in range(ensemble_size):
            dense_name = "dense_layer_"+str(layer_no)+'_'+str(i)

            if(layer_no==0):
                new_dense_layer = layers.Dense(dense_layers_sizes[layer_no],
                activation='relu',name=dense_name)(flattened_layers[i])
            else:
                new_dense_layer = layers.Dense(dense_layers_sizes[layer_no],
                activation='relu',name=dense_name)(old_dense_layers[i])
            dense_layers.append(new_dense_layer)

        network,output_layers = Add_ensemble_output(input_tensor=input_tensor,
            top_layers=dense_layers,ensemble_size=ensemble_size,convolutions=False)
        network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
        network.summary()
        deeptrack.train_deep_learning_network_mp(network,
            translation_distance=translation_distance,SN_limits = SN_limits,
            sample_sizes = sample_sizes,iteration_numbers = iteration_numbers,
            verbose=verbose,nbr_outputs=ensemble_size,parameter_function=parameter_function)
        freeze_all_layers(network)
        if(save_networks):
            network.save(model_path+"network_no"+str(len(conv_layers_sizes)+layer_no)+".h5")

    return network
def Add_ensemble_output(input_tensor,top_layers,ensemble_size,convolutions):
        # Function which adds an output on top of a ensemble network
        from keras import layers,models
        output_layers = []

        # Add flattening to convolutional layers
        if convolutions:
            new_top_layers = []
            for i in range(ensemble_size):
                flattened_name = "flatten_layer_"+str(i)
                new_flattened = layers.Flatten(name=flattened_name)(top_layers[i])
                new_top_layers.append(new_flattened)
            top_layers = new_top_layers

        # Add outputs
        for i in range(ensemble_size):
                output_name = "Output_"+str(i+1)
                new_dense_out = layers.Dense(3,name=output_name)(top_layers[i])
                output_layers.append(new_dense_out)
        # Create network
        network = models.Model(input_tensor,output_layers)
        return network,output_layers
