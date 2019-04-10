def get_inputs_to_last_layer(model):
    """
    Takes a network and returns a network which outputs the inputs to the last
    layer of that network. Useful for monitoring modular networks.
    """
    from keras.models import Model
    intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[-1].input)
    return intermediate_layer_model
def model_outputs_from_layer(model,layer_idx):
    """
    Takes a network and returns a network which outputs the outputs to layer idx
    layer of that network. Useful for monitoring modular networks.
    """
    from keras.models import Model
    intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[layer_idx].output)
    return intermediate_layer_model
def get_predictions_from_layer(model,layer_idx,data):
    """
    Returns predictions made by layer layer_idx on data
    """
    new_model = model_outputs_from_layer(model,layer_idx)
    predictions = new_model.predict(data)
    return predictions
def nbr_nonzero_dense_activations_single(prediction):
    """
    Calculates the fraction of activations which are nonzero. Also returns a
    array whith elements being 1 if corresponding node was larger than 0 and
    """
    import numpy as np

    results = np.zeros((len(prediction),1))# predictions often 2d
    fraction_nonzero = 0

    for i in range(len(prediction)):
        if prediction[i]>0:
            results[i] = 1
            fraction_nonzero += 1

    fraction_nonzero /= len(prediction)

    return fraction_nonzero,results
def nbr_nonzero_conv_activations_single(predictions):
    """
    Check how many of the nodes activations where non-zero
    """
    import numpy as np

    nbr_nodes = len(predictions[0,0,:])
    results = np.zeros((nbr_nodes,1))
    fraction_nonzero = 0
    for i in range(nbr_nodes):
        #print(np.max(predictions[:,:,i]),nbr_nodes)
        if np.max(predictions[:,:,i])>0:
            fraction_nonzero += 1
            results[i] = 1
    fraction_nonzero /= nbr_nodes
    return fraction_nonzero,results
def nbr_nonzero_activations_relu(predictions,type=0):
    """

    """
    import numpy as np
    if len(predictions.shape)>2:
        type = 1
    nbr_predictions = len(predictions)
    nonzero_fractions = np.zeros((nbr_predictions,1))
    activation_list = [] # List of the activations for the different inputs
    for i in range(nbr_predictions):
        if type==0:
            nonzero_fractions[i],activation = nbr_nonzero_dense_activations_single(predictions[i])
        else:
            nonzero_fractions[i],activation = nbr_nonzero_conv_activations_single(predictions[i])
        activation_list.append(activation)
    return nonzero_fractions,activation_list
def evaluate_layer_activations(model,data,layer_idx): # Make it possible to speciy layer by name as well as index
    """

    """
    predictions = get_predictions_from_layer(model,layer_idx,data)
    #print(predictions.shape,len(predictions.shape))
    return nbr_nonzero_activations_relu(predictions)
