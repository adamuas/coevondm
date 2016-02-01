

import numpy;
import math;

"""

Output Functions


"""


def inner_prod(inputs,weights, **kwargs ):
    """
    Performs inner product input combination
    :param inputs:
    :param weights:
    :param kwargs:
    """

    output =  numpy.sum(numpy.float32(weights) * numpy.float32(inputs));


    return output;

def euclidean_dist(inputs, weights, **kwargs):
    """
    Performs euclidean distance input combination
    :param inputs:
    :param weights:
    :param kwargs:
    """

    exp = 2;
    output = numpy.sum(
        numpy.power((numpy.float32(weights) - numpy.float32(inputs)),2)
    );


    return output;

def manhattan_dist(inputs, weights,**kwargs):
    """
     Performs manhattan distance input combination
    :param inputs:
    :param weights:
    :param kwargs:
    """

    output = numpy.sum(numpy.float32(weights) - numpy.float32(inputs));

    return output;




def max_dist(inputs, weights, **kwargs):
    """
     Performs max dist input combination
    :param inputs:
    :param weights:
    :param kwargs:
    """
    try:
        output = (numpy.float32(weights) - numpy.float32(inputs));
        output = numpy.min(output);
    except ValueError:
        raise Exception("Some issue while applying max distance");
        #Could happen if the inputs and weights are just 1D arrays
        if len(inputs) == 0 and len(weights) == 0:
            output = 0.0;

    return output;



def higher_order_prod(inputs, weights, **kwargs):
    """
     Performs higher order product input combination
    """
    if 'prodConstant' in kwargs.keys():
        prodConstant = kwargs['prodConstant'];
    else:
        # print ("ProdConstant Not passed: using 1 as prodConstant");
        #use product constant as 1
        prodConstant = 1;

    AP = (numpy.float32(weights) * numpy.float32(inputs)) * prodConstant;

    output = 1;

    try:
        #do product
        for ap in AP:
            if numpy.nonzero(ap):
                output *= ap;

    except TypeError:
        #AP is a float and not an ndarray
        raise Exception("Some issue while applying HO prod");
        output = AP;

    return output;




def higher_order_sub(inputs, weights, **kwargs):
    """
     Performs higher order sub input combination
    """

    #first connection
    central_feature = 0;

    AP = (numpy.float32(weights) * numpy.float32(inputs));
    try:
        #get central feature
        central_feature_out = AP[central_feature];

        output = 0;
        #do subtractive

        for i,ap in enumerate(AP):
            if numpy.nonzero(ap) and i != central_feature:
                diff = (central_feature_out - ap);
                output += diff;
    except IndexError:
        raise Exception("Some issue while applying HO Sub");
        #handle when AP is only a float
        output = AP;


    return output;

def higher_order_sub_adaptive(inputs, weights, **kwargs):
    """
     Performs higher order sub input combination - However, the central feature is selected adaptively
    """

    if 'central_feature' in kwargs.keys():
        connActive = kwargs['central_feature'];
    else:
        #first connection (AS in higher order subtractive)
        central_feature = 1;

    AP = (numpy.float32(weights) * numpy.float32(inputs));

    output = 1;
    central_feature_out = AP[central_feature];

    #do product
    for i,ap in enumerate(AP):
        if numpy.nonzero(ap) and i != central_feature:
            output += (central_feature_out - ap);

    return output;


def std_dev(inputs, weights, **kwargs):
    """
     Performs standard dev input combination
    """

    output = numpy.std(numpy.float32(weights) * numpy.float32(inputs));

    return output;


def mean(inputs, weights, **kwargs):
    """
     Performs mean  input combination
     :param inputs
     :param weights
     :param kwargs - Additional parameters

    """

    output = numpy.mean(numpy.float32(weights) * numpy.float32(inputs));

    return output;


def min(inputs, weights, **kwargs):
    """
     Performs min input combination
    """
    try:
        output = (numpy.float32(weights) * numpy.float32(inputs));
        output = numpy.min(output);
    except ValueError:
        #Could happen if the inputs and weights are just 0D arrays
        raise Exception("Some issue while applying min");
        output = 0.0;

    return output;


def max(inputs, weights, **kwargs):
    """
     Performs max  input combination
    """

    try:
        output = (numpy.float32(weights) * numpy.float32(inputs));
        output = numpy.max(output);
    except ValueError:
        #Could happen if the inputs and weights are just 1D arrays
        raise Exception("Some issue while applying max");
        output = 0.0;

    return output;

"""
Store Indices of the functions
"""

fn_indices = {
    '1': inner_prod,
    '2': euclidean_dist,
    '3': higher_order_prod,
    '4': higher_order_sub,
    '5': std_dev,
    '6': min,
    '7': max,
    '8': manhattan_dist,
    '9': max_dist,
    '10':higher_order_sub_adaptive, #An experimental function that aims to make the transfer function more robust
    '11':mean
};


def getFunctionIndex(Fn):
    """
    Returns the index of the function None other wise
    :param Fn - Function
    """
    indx = None;
    for fni,fn in fn_indices.iteritems():

        if id(fn) == id(Fn):
            indx = int(fni);
            return indx;

NUM_FUNC_CLASS = 4;

def getFunctionsClass(fn_class):

    """
    Returns a set of transfer functions for the desired class
    :param class of the functions ('radial-basis','projection', 'statistical', 'higher-order')
    """

    fn_classes = {
        'radial-basis': [euclidean_dist, manhattan_dist, max_dist],
        'projection': [inner_prod],
        'statistical': [min,max,std_dev, mean],
        'higher-order': [higher_order_prod, higher_order_sub],

    };

    return fn_classes[fn_class];