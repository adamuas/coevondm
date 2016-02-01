import numpy;
import math;
"""

Activation Functions

"""


#TODO: Consider neural computational path fn visualisation
#TODO: Grammatic Programming to mutate the transfer functions

PRECISIONS = 5;

def identity_fn(x, *args):
    #returns normalised identity of the given function
    # smooth_param = 10;
    # dSquared = numpy.float32(x) * numpy.float32(x);
    # out = numpy.float32(x)/((numpy.sqrt(dSquared + smooth_param)));
    # if math.isnan(out):
    #     out = 0.0;
    #
    # if out == float('inf') or out == float('-inf'):
    #     out = 1.0;

    return x;


def sigmoid_fn( x ,*args):
    steepness = args[0];
    c = args[1];
    #sigmoid activation function
    out = (round(c,PRECISIONS))/(1 + (numpy.exp(round(-x,PRECISIONS) * numpy.float32(steepness)) ));
    try:
        if math.isnan(out):
            raise Exception("NaN - sigmoid fn")
            out = 0.0;

        if out == float('inf') or out == float('-inf'):
            raise Exception("Inf - sigmoid fn")
            out = 1.0;

    except TypeError:
        #can occur if the input is empty
        raise Exception("Possible empty input for sigmoid fn")
        out = 0.0;


    return out;


def gaussian_fn( x, *args):
    #gaussian activation function
    #divide by zero fix
    mu = args[0]; #mean
    var = args[0]; #var

    if var == 0.0: #avoid divide by zero
        var = 0.1;
    out = numpy.exp(
                -(numpy.power(numpy.float32(x-mu),2))/numpy.power(numpy.float32(var),2)
        );


    return out;


def thin_plate_spline_fn( x, *args):
    #thin plate spline
    width =numpy.float32( args[0]);
    exp = 2;
    out = numpy.power(x*width,exp) * numpy.log(x*width);

    if math.isnan(out):
        raise Exception("Not a number - thin plate spline fn")
        out = 0.0;

    if out == float('inf') or out == float('-inf'):
        raise Exception("Inf- thin plate spline fn")
        out = 1.0;

    return out;


def tanh_fn( x, *args):
    steepness = args[0];
    #tanh activation function
    return numpy.tanh(numpy.float32(x) * numpy.float32(steepness));

def arc_tanh_fn( x, *args):
    steepness = args[0];
    #arc tanh output function
    return math.atan(x * numpy.float32(steepness));

def gaussian_ii_fn( x, *args):

    width = numpy.float32(args[0]);
    cut_off = numpy.float32(args[1]);

    #gaussian II activation
    val = gaussian_fn(x,width);
    #cut-off at threshold
    if val >= cut_off:
        val = 1;
    #return value
    return val;

"""
Store Indices of the functions
"""

fn_indices = {
    '1': identity_fn,
    '2': sigmoid_fn,
    '3': gaussian_fn,
    '4': tanh_fn,
    '5': gaussian_ii_fn,
    '6': thin_plate_spline_fn,
    '7': arc_tanh_fn
};

def getFunctionIndex(Fn):
    """
    Returns the index of the function None other wise
    :param Fn - Function
    """
    indx = -1;
    for fni,fn in fn_indices.iteritems():

        if id(fn) == id(Fn):
            indx = int(fni);
            return indx;
    print "Function not found";
    return indx;


""" Number of function classes available """
NUM_FUNC_CLASS = 3;

def getFunctionsClass(fn_class):

    """
    Returns a set of output functions for the desired class
    :param class of the functions ('radial-basis','projection', 'statistical', 'higher-order')
    """

    fn_classes = {
        'radial-basis': [gaussian_fn, gaussian_ii_fn],
        'projection': [identity_fn, sigmoid_fn,tanh_fn],
        'statistical': [],
        'higher-order': [identity_fn, sigmoid_fn, tanh_fn], #use identity, sigmod and tan-fn

    };

    return fn_classes[fn_class];
