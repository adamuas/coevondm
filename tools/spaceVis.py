from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import Image;
import ntpath;
from matplotlib import cm

""" This class helps to visualise 3D input problem spaces"""



def visSpace(filename,detail):
    """ """
    #get the filename
    head, dataname = ntpath.split(filename);

    #read the image
    I = Image.open(filename);
    I = I.convert('RGB')
   

    #figure
    fig = plt.figure(filename);
    
    #add subplot
    ax = fig.add_subplot(111,projection='3d');
    

    
    X = [];
    Y = [];
    Z = [];

    w,h= I.size;
    
    
    widthStepSize = int(w/float(detail));
    heightStepSize = int(h/float(detail));
    
    if widthStepSize == 0:
        widthStepSize = 1;
    if heightStepSize == 0:
        heightStepSize = 1;
    
    
    #start from random position
    for y in xrange(1,w,widthStepSize):
        
        for x in xrange(1,h,heightStepSize):
            
            i1 = x ;
            i2 = y ;
             
            
             
            #get coordinates
            i1 = i1/float(w);
            i2 = i2/float(h);
            
            X.append(i1);
            Y.append(i2);
        
           
            #get output of the pixel coordinates
            r,g,b = I.getpixel((x,y));
    
            out = ((max(r, g, b) + min(r, g, b)) / 2.0)/255.0;
            Z.append(out);
            
            
            
            
    #X,Y = np.meshgrid(X,Y);
    #draw wire frames
    #ax.plot_surface(X,Y,Z,rstride=5, cstride=5,cmap =cm.jet);
    ax.plot_wireframe(X,Y,Z,rstride=5, cstride=5);
    
    #show
    plt.show();
    
    






