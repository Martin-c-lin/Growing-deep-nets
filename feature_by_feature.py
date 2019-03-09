# import keras
# from keras import  Input,models,layers
# from keras.models import Model

def print_results_to_file(network,history,file_name,model_name):
    """
    Prints history and model setup

    Inputs:
        Network, network history and network/model name
    Ouputs:
        file with model summary and the data from the history
    """
    import numpy as np
    file_trad = open(file_name, "w")

    # Print to file
    orig_std_out = sys.stdout
    sys.stdout = file_trad
    print(network.summary())
    sys.stdout = orig_std_out
    file_trad.write(model_name+'acc = ')
    file_trad.write(str(history.history['acc']))
    file_trad.write(';\n')
    file_trad.write(model_name+'val_acc = ')
    file_trad.write(str(history.history['val_acc']))
    file_trad.write(';\n')
    file_trad.write(model_name+'loss = ')
    file_trad.write(str(history.history['loss']))
    file_trad.write(';\n')
    file_trad.write(model_name+'val_loss = ')
    file_trad.write(str(history.history['val_loss']))
    file_trad.write(';\n')
    file_trad.close()

    # Save to numpy array
    file_length = len(history.history['acc'])
    np_results = np.zeros((4,file_length))
    np_results[0,:] = history.history['acc']
    np_results[1,:] = history.history['val_acc']
    np_results[2,:] = history.history['loss']
    np_results[3,:] = history.history['val_loss']
    np_filname = file_name+"np_res"
    np.save(np_filname,np_results)
def fbf_create_model_conv(input_shape=(51,51,1),number_of_outputs=1,output_activation = None):
    """
    Funciton for creating the first seed for a fbf trained conv-model
    Inputs:


    Outputs:

    """
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    input_tensor = Input(shape=input_shape)
    conv_list=[]
    conv_feature = layers.Conv2D(1, (3,3),activation = 'relu',input_shape=(150,150,3) )(input_tensor)
    conv_list.append(conv_feature)
    intermediate_conv = layers.MaxPooling2D((2,2))(conv_feature) # needed later
    intermediate_layer = layers.Flatten()(intermediate_conv) # Remove intermediate_layer in favour of only
    if(output_activation==None):
        output_tensor = layers.Dense(number_of_outputs)(intermediate_layer)
    else:
        output_tensor = layers.Dense(number_of_outputs,activation=output_activation)(intermediate_layer)

    #Put model together
    model = Model(input_tensor,output_tensor)
    return model,conv_list,input_tensor# returns also intermediate models
def fbf_create_model_conv_new_initalizer_test(initializer,input_shape=(51,51,1),number_of_outputs=1,output_activation = None):
    """
    Funciton for creating the first seed for a fbf trained conv-model
    Inputs:


    Outputs:

    """
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    input_tensor = Input(shape=input_shape)
    conv_list=[]

    conv_feature = layers.Conv2D(1, (3,3),activation = 'relu',input_shape=(150,150,3),kernel_initializer=initializer )(input_tensor)
    conv_list.append(conv_feature)
    intermediate_conv = layers.MaxPooling2D((2,2))(conv_feature) # needed later
    intermediate_layer = layers.Flatten()(intermediate_conv) # Remove intermediate_layer in favour of only
    if(output_activation==None):
        output_tensor = layers.Dense(number_of_outputs)(intermediate_layer)
    else:
        output_tensor = layers.Dense(number_of_outputs,activation=output_activation)(intermediate_layer)

    #Put model together
    model = Model(input_tensor,output_tensor)
    return model,conv_list,input_tensor# returns also intermediate models
def add_new_conv_feature(conv_list,input_layer,previous_layer,number_of_outputs=1,output_activation=None):
    """
    Function for adding a feature to a fbf model
    Inputs:
        conv_list - list of convolutional features in the current layer
        input_tensor - tensor for input to the current layer # take care here to make it so that it works also for layers other than the first
        number_of_outputs - Dimension of output
        output_activation - Activation function to use for the final dense layer
    Outputs:
        model - model with one more feature in the current layer(not compiled)
        con_list - list of the convolutional layer-features in the model
    """
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    ### Create new single feature layer and add it to the list

    new_conv_feature = layers.Conv2D(1, (3,3),activation = 'relu' )(previous_layer)
    conv_list.append(new_conv_feature)

    ### Construct model

    intermediate_layers = layers.concatenate(conv_list,axis=-1)
    intermediate_layers = layers.MaxPooling2D((2,2))(intermediate_layers)
    intermediate_layers = layers.Flatten()(intermediate_layers)

    ### Check output type
    if(output_activation==None):
        output_layer = layers.Dense(number_of_outputs)(intermediate_layers)
    else:
        output_layer = layers.Dense(number_of_outputs,activation=output_activation)(intermediate_layers)
    model = Model(input_layer,output_layer)
    return model,conv_list
def add_new_conv_layer(conv_list,input_layer,previous_layer,number_of_outputs=1,output_activation=None):
    """
    Function for creating a new layer in a fbf model
    Inputs:
        conv_list - list of convolutional features in the current layer
        input_tensor - tensor for input to the current layer # take care here to make it so that it works also for layers other than the first
        number_of_outputs - Dimension of output
        output_activation - Activation function to use for the final dense layer
    Outputs:
        model - Model with a single feature in the new layer
        new_conv_list - List with the features in the current top convolutional
            layer
        old_layers - Previous convolutional layers
    """
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    old_layers = layers.concatenate(conv_list,axis=-1)
    old_layers = layers.MaxPooling2D((2,2))(old_layers)
    new_feature = layers.Conv2D(1, (3,3),activation = 'relu' )(old_layers)
    new_conv_list = []
    new_conv_list.append(new_feature)
    intermediate_conv = layers.MaxPooling2D((2,2))(new_feature) # needed later
    intermediate_layer = layers.Flatten()(intermediate_conv) # Remove intermediate_layer in favour of only
    if(output_activation==None):
        output_layer = layers.Dense(number_of_outputs)(intermediate_layer)
    else:
        output_layer = layers.Dense(number_of_outputs,activation=output_activation)(intermediate_layer)
    model = Model(input_layer,output_layer)

    return model,new_conv_list,old_layers
def freeze_all_layers(model):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
def fbf_sum_deeptrack(
    max_number_of_features,
    train_generator,
    input_shape=(51,51,1),
    number_of_outputs=3,
    sample_sizes=(8, 32, 128, 512, 1024),
    iteration_numbers=(401, 301, 201, 101, 51),
    verbose=0.01,
    model_path=""):
    import deeptrack
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    ############################################################################
    # Funciton for creating the first fbf trained layer. Uses sum rather
    # than concatenation to combine models
    # Inputs:
    #
    # Outputs:
    #
    ############################################################################
    input_tensor = Input(shape=input_shape)

    # Loop over features and gradually grow network
    output_list=[]
    for i in range(max_number_of_features):
        # Create and add new conv feature
        conv_feature = layers.Conv2D(1, (3,3),activation = 'relu' )(input_tensor)
        intermediate_conv = layers.MaxPooling2D((2,2))(conv_feature) # needed later
        #CHECK if the weights are correctly transfered
        intermediate_layer = layers.Flatten()(intermediate_conv) # Remove intermediate_layer in favour of only

        output_single = layers.Dense(number_of_outputs)(intermediate_layer)#Output activation probably very relevant
        # Just adding things up will ruin classification, mus be function with mean 0

        output_list.append(output_single)
        if(i==0):
            combined_output = output_single
        else:
            combined_output = layers.add(output_list)#layers.concatenate([conv_features,conv_feature])
        #Put model together
        model = Model(input_tensor,combined_output)
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

        model.summary()

        deeptrack.train_deep_learning_network(
            model,
            train_generator,
            sample_sizes = sample_sizes,
            iteration_numbers = iteration_numbers,
            verbose=verbose)
        freeze_all_layers(model)
        model.save(model_path+"fbf_sum_DT_L1_"+str(i+1)+"F.h5")
        # Log both validation and test accuracy. But use only validation accuracy for the model to optimize its shape on
    return model,intermediate_conv,input_tensor# returns also intermediate models
def FBF_sum_modular_deeptrack(
    layer_size,
    train_generator,
    input_shape=(51,51,1),
    output_shape=3,
    sample_sizes=(8, 32, 128, 512, 1024),
    iteration_numbers=(401, 301, 201, 101, 51),
    verbose=0.01,
    save_networks=True,
    # TODO add mp functionality option
    model_path=""):
    """
    Function implementing the more advanced modular FBF which has denser connctions.

    Inputs:

    Outputs:

    TODO - Implement weight decay in later layers and lowering learning rate
    """
    import deeptrack
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    #import deeptrackelli_mod_mproc as multiprocess_training # Needed for fast parallelized image generator

    input_tensor = Input(input_shape)
    conv_list = [] # List of convolutional neruons in L1, needed for subsequent layers
    flattened_list = [] # List of flattened layers
    output_list = [] # List of the output layers

    # Loop thorugh the neoruns and add them one by one
    for i in range(layer_size):
        next_neuron = layers.Conv2D(1,(3,3),activation='relu')(input_tensor)
        next_neuron = layers.MaxPooling2D((2,2))(next_neuron)
        if(i==0):
            # i = 0 special case. No addition needed
            next_flattened = layers.Flatten()(next_neuron)
            next_output = layers.Dense(3)(next_flattened)
            final_outptut = next_output
            output_list.append(next_output)
            flattened_list.append(next_flattened)
        else:
            # Construct the next output neuron
            next_flattened = layers.Flatten()(next_neuron)
            flattened_list.append(next_flattened)
            next_output = layers.Concatenate(axis=-1)(flattened_list)
            next_output = layers.Dense(3)(next_output)
            output_list.append(next_output)
            final_outptut = layers.add(output_list)
            # Construct and compile network
        network = models.Model(input_tensor,final_outptut)
        network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
        network.summary()

        # Train and freeze layers in network
        deeptrack.train_deep_learning_network(
            network,
            train_generator,
            sample_sizes = sample_sizes,
            iteration_numbers = iteration_numbers,
            verbose=verbose)

        freeze_all_layers(network)
        conv_list.append(next_neuron)
        if(save_networks):
            network.save(model_path+"L1_"+str(i+1)+"F.h5")


    return network,conv_list,output_list,flattened_list,input_tensor
def FBF_sum_modular_deeptrack_layer2(
    layer_size,
    train_generator,
    conv_list,
    output_list,
    #flattened_list, # not used
    input_tensor,
    sample_sizes=(8, 32, 128, 512, 1024),
    iteration_numbers=(401, 301, 201, 101, 51),
    verbose=0.01,
    save_networks=True,
    model_path=""):
    """
    Function for adding a new layer in a modular fbf model, note that it adds a
    convolutional layer with pooling.
    """
    import deeptrack
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    #import deeptrackelli_mod_mproc as multiprocess_training
    # Needed for fast parallelized image generator but don't yet support windows

    new_layer_conv_list = [] # Convolutions in the new layer
    new_layer_flattened_list = [] # flattened output from new layer

    if len(conv_list)>1:
        new_layer_input = layers.Concatenate()(conv_list)
    else:
        new_layer_input = conv_list[0]
    for i in range(layer_size):
        next_neuron = layers.Conv2D(1,(3,3),activation='relu')(new_layer_input)
        next_neuron = layers.MaxPooling2D((2,2))(next_neuron)
        next_flattened = layers.Flatten()(next_neuron)
        new_layer_flattened_list.append(next_flattened)
        if(i==0):
            # i = 0 special case. No addition needed
            next_output = layers.Dense(3)(next_flattened) # Different for i==0
        else:
            next_output = layers.Concatenate(axis=-1)(new_layer_flattened_list)
            next_output = layers.Dense(3)(next_output)
        output_list.append(next_output)
        total_output = layers.add(output_list)
        # Construct and compile network
        network = models.Model(input_tensor,total_output)
        network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
        network.summary()

        # Train and freeze layers in network
        deeptrack.train_deep_learning_network(
            network,
            train_generator,
            sample_sizes = sample_sizes,
            iteration_numbers = iteration_numbers,
            verbose=verbose)

        freeze_all_layers(network)
        new_layer_conv_list.append(next_neuron)
        #new_layer_flattened_list.append(next_flattened)
        if(save_networks):
            network.save(model_path+"L2_"+str(i+1)+"F.h5")
    return network,new_layer_conv_list,output_list,new_layer_flattened_list,input_tensor
def fbf_modular_expand_layer(
    expansion_size,
    train_generator,
    conv_list,
    output_list,
    flattened_list,
    input_tensor,
    sample_sizes=(8, 32, 128, 512, 1024),
    iteration_numbers=(401, 301, 201, 101, 51),
    verbose=0.01,
    save_networks=True,
    model_path=""):
    """
    Function for expanding preexisting layer of an fbf_modular model.
    Inputs:
        expansion_size - number of nodes which layer should be expanded with
        train_generator - image generator for the new nodes to be trained on
    Outputs:

    """
    import deeptrack
    import keras
    from keras import  Input,models,layers
    from keras.models import Model
    #import deeptrackelli_mod_mproc as multiprocess_training # Needed for fast parallelized image generator

    for i in range(expansion_size):
        next_neuron = layers.Conv2D(1,(3,3),activation='relu')(input_tensor)
        next_neuron = layers.MaxPooling2D((2,2))(next_neuron)

        # Construct the next output neuron
        next_flattened = layers.Flatten()(next_neuron)
        flattened_list.append(next_flattened)
        next_output = layers.Concatenate(axis=-1)(flattened_list)
        next_output = layers.Dense(3)(next_output)
        output_list.append(next_output)
        final_outptut = layers.add(output_list)
        # Construct and compile network
        network = models.Model(input_tensor,final_outptut)
        network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
        network.summary()

        # Train and freeze layers in network
        deeptrack.train_deep_learning_network(
            network,
            train_generator,
            sample_sizes = sample_sizes,
            iteration_numbers = iteration_numbers,
            verbose=verbose)

        freeze_all_layers(network)
        conv_list.append(next_neuron)
        if(save_networks):
            network.save(model_path+"L1_"+str(i+1)+"F.h5")

    return network,conv_list,output_list,flattened_list,input_tensor
def get_modular_terms_outputs(input_tensor,output_node_list,images):
    """
    Function which calculates the outputs from the various nodes(or layers) in output
    node list on a specific image and returns them in a list.
    Inputs:
        input_tensor -input tensor to model as gotten from the fbf_modular_sum...
        output_node_list - list of nodes which produce an output
        images - images to make predictions on
    Outputs:
        predictions - predictions made by the different nodes
    """
    from keras.models import Model
    from keras.layers import concatenate
    import numpy as np
    nbr_output_nodes = len(output_node_list)
    if len(output_node_list)>1:
        output = concatenate(output_node_list,axis=1)
    else:
        output = output_node_list[0]
    network = Model(input_tensor,output)
    predictions = network.predict(images)
    predictions = np.reshape(predictions,(len(images),nbr_output_nodes,3))
    return predictions
# def get_modular_terms_outputs_from_network(network,images):
#     """
#
#     """
#     #layer.input
#     from keras.models import Model
#     new_output = network.layers[-1].input
#     layer.get_input_at(node_index)
#     return 0
def get_default_image_generator_deeptrack(translation_distance = 5,SN_limits=[10,100]):
    import deeptrack

    ### Define image properties
    from numpy.random import randint, uniform, normal, choice
    from math import pi

    image_parameters_function = lambda : deeptrack.get_image_parameters(
       particle_center_x_list=lambda : normal(0, translation_distance, translation_distance),# normal(0, 1, 1), # increase these to get a more vivid approximation?
       particle_center_y_list=lambda : normal(0, translation_distance, translation_distance),# normal(0, 1, 1),
       particle_radius_list=lambda : uniform(1.5, 3, 1),
       particle_bessel_orders_list=lambda : [[randint(1, 3),], ],
       particle_intensities_list=lambda : [[choice([-1, 1]) * uniform(.2, .6, 1), ], ],
       image_half_size=lambda : 25,
       image_background_level=lambda : uniform(.2, .8),
       signal_to_noise_ratio=lambda : uniform(SN_limits[0], SN_limits[1]),
       gradient_intensity=lambda : uniform(0, 1),
       gradient_direction=lambda : uniform(-pi, pi))

    ### Define image generator
    image_generator = lambda : deeptrack.get_image_generator(image_parameters_function)

    return image_generator
def modular_fbf_weight_initializer(size_inputs,size_outputs,neuron_number):
    """
    Customized weight initializer for fbf modular. Initializer is a modification
    of Glorot uniform
    Inputs:
        size_inputs - number of inputs to neuron
        size_outputs - number of outputs of neuron
        neuron number - the index of the current neuron
    Outputs:
        initializer
    """
    from keras.initializers import RandomUniform
    import math
    limit = math.sqrt(6/(size_inputs+size_outputs))
    limit /= neuron_number
    return RandomUniform(minval=-limit,maxval=limit)
