# Artificial, generated  examples.
import random;
import math;
import numpy;

def Majority(k, n):
    """Return a DataSet with n k-bit examples of the majority problem:
    k random bits followed by a 1 if more than half the bits are 1, else 0."""
    
    
    inputs = [];
    outputs = [];
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        inputs.append(bits);
        out = sum(bits) > k/2;
        if out == False:
            out = 0.0;
        if out == True:
            out = 0.9;
        outputs.append([out]);
    
    return dict(NAME='majority', IN=inputs, OUT = outputs)

def Parity(k, n, name="N-Bit Parity"):
    """Return a DataSet with n k-bit examples of the parity problem:
    k random bits followed by a 1 if an odd number of bits are 1, else 0."""
    
    inputs = [];
    outputs = [];
    for i in range(n):
        bits = [random.choice([0, 1]) for i in range(k)]
        inputs.append(bits);
        out = (sum(bits) % 2);
        if out == False:
            out = 0.0;
        if out == True:
            out = 0.9;
        outputs.append([out]);
       
    return dict(NAME=name, IN= inputs, OUT = outputs)

def Xor(n):
    """Return a DataSet with n examples of 2-input xor."""
    return Parity(2, n, name="XOR")

def ContinuousXor(n):
    "2 inputs are chosen uniformly form (0.0 .. 2.0]; output is xor of ints."
    examples = []
    for i in range(n):
        x, y = [random.uniform(0.0, 2.0) for i in '12']
        examples.append([x, y, int(x) != int(y)])
    return dict(NAME='continuous xor', examples=examples)


def girosiFunction(n):
    """2 inputs are chosen uniformly from [0.0 ... 1.0]"""
    
    name = 'GirosiFunction';
    inputs = [];
    outputs = [];
    
    for i in range(n):
         
        #pick random coordinates   
        x = random.uniform(0.0, 1.0);
        y = random.uniform(0.0, 1.0);
        #append inputs
        inputs.append([x,y]);
        
        #calculate the girosi function for such coordinates
        girosi_out = math.sin(2*math.pi * x) + 4* (math.pow(y - 0.5,2));
        
        outputs.append([girosi_out]);
       
    return dict(NAME=name, IN= inputs, OUT = outputs)

def gaborFunction(n):
    """2 inputs are chosen uniformly from [-1 ... 1.0]"""
    
    name = 'GaborFunction';
    inputs = [];
    outputs = [];
    
    for i in range(n):
         
        #pick random coordinates   
        x = random.uniform(-1.0, 1.0);
        y = random.uniform(-1.0, 1.0);
        #append inputs
        inputs.append([x,y]);
        
        #calculate the girosi function for such coordinates
        gabor_out = math.exp(-1 * numpy.abs(math.pow(x,2))) * math.cos(0.75 * math.pi * (x+y))
        
        outputs.append([gabor_out]);
       
    return dict(NAME=name, IN= inputs, OUT = outputs)

def sugenoFunction(n,start,end):
    """2 inputs are chosen uniformly from [-1 ... 1.0]"""
    
    name = 'SugenoFunction';
    inputs = [];
    outputs = [];
    
    for i in range(n):
         
        #pick random coordinates   
        x = random.uniform(start, end);
        y = random.uniform(start, end);
        z = random.uniform(start,end)
        #append inputs
        inputs.append([x,y,z]);
        
        #calculate the girosi function for such coordinates
        fxyz = (1 + math.pow(x,0.5) + math.pow(y,-1) + math.pow(z,-1.5));
        gabor_out = math.pow(fxyz,2);
        
        outputs.append([gabor_out]);
       
    return dict(NAME=name, IN= inputs, OUT = outputs)