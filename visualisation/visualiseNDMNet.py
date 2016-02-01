#import ndm path
import os, sys, inspect
sys.path.insert(0,"../");
sys.path.insert(0,"../core");
sys.path.insert(0,"../tools");

import math
import numpy;

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt, QRect, QRectF
from PyQt4.QtGui import QPen, QBrush, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QPixmap, QPainter, QApplication
from NDMNode import Node;
import Edge;

from ndmModel import ndmModel;
import constants;
import time;
import output_fn;
import activation_fn;
import commons;

class GraphWidget(QtGui.QGraphicsView):
    def __init__(self, model, title = None):
        super(GraphWidget, self).__init__()

        self.timerId = 0
        
        scene = QGraphicsScene(self)
        scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)
        scene.setSceneRect(-200, -200, 640, 400)
        self.gscene = scene;
        self.setScene(scene)
        self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QtGui.QGraphicsView.BoundingRectViewportUpdate)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtGui.QGraphicsView.AnchorViewCenter)
        self.model = model;
        self.whitebrush = QBrush(Qt.white);
        self.redbrush = QBrush(Qt.red)
        self.drawNetwork(self.gscene, self.model);
        
        #setttings of the window
        self.scale(1.2,1.2);
        self.setMinimumSize(640, 480);
        
        
        if title == None:
            self.setWindowTitle("Network Visualisation");
        else:
            self.setWindowTitle("Network Visualisation -" + title);
        
    
        
    def drawNetwork(self,scene, model):
        """ Draws the visualisation of the network"""
        nodes = [];
        
        #create input nodes
        for n in xrange(model.numI):

            #get output function
            #node representation
            INPUT_NODE = 1;
            IDENTITY = 1;
            n_rep = Node(self,nodeType=INPUT_NODE,node_fn = IDENTITY);
            #add to nodes
            nodes.append(n_rep);
            
            #add to scene
            scene.addItem(n_rep);

        #create hidden nodes
        for n in xrange(model.numH):

            #get the node
            node = model.hiddenNodes[n];

            out_fn = node.output_fn;
            out_fni = output_fn.getFunctionIndex(out_fn);

            #node representation
            HIDDEN_NODE = 2;
            n_rep = Node(self,nodeType=HIDDEN_NODE,node_fn=out_fni);
            #add to nodes
            nodes.append(n_rep);

            #add to scene
            scene.addItem(n_rep);

        #create output nodes
        for n in xrange(model.numO):

            #get the node
            node = model.outputNodes[n];

            out_fn = node.output_fn;
            out_fni = output_fn.getFunctionIndex(out_fn);

            #node representation
            OUTPUT_NODE = 3;
            n_rep = Node(self,nodeType=OUTPUT_NODE,node_fn=out_fni);
            #add to nodes
            nodes.append(n_rep);

            #add to scene
            scene.addItem(n_rep);
                
            
        #create connections (input to hidden units)
        for ni in xrange(model.numI):
            nodei = nodes[ni];

            for nj in xrange(model.numH):
                nodej = nodes[model.numI + nj];
                weightFn = model.hiddenNodes[nj].activation_fn;
                wfni = activation_fn.getFunctionIndex(weightFn);

                #Get connection details
                active = model.connActive_IH[ni][nj];
                connWeight = model.connWeight_IH[ni][nj];

                if active:
                    conn = Edge.Edge(nodei,nodej,wfni,connWeight);
                    scene.addItem(conn);

        #create connections (hidden to hidden units)
        for ni in xrange(model.numH):
            nodei = nodes[model.numI+ni];

            for nj in xrange(model.numH):
                nodej = nodes[model.numI + nj];
                weightFn = model.hiddenNodes[nj].activation_fn;
                wfni = activation_fn.getFunctionIndex(weightFn);

                #Get connection info.
                active = model.connActive_HH[ni][nj];
                connWeight = model.connWeight_HH[ni][nj];

                #connect if connection is active
                if active:
                    conn = Edge.Edge(nodei,nodej,wfni,connWeight);
                    scene.addItem(conn);


        #create connections (hidden to hidden units)
        for ni in xrange(model.numH):
            nodei = nodes[model.numI+ni];

            for nj in xrange(model.numO):
                nodej = nodes[model.numI + model.numH + nj];
                weightFn = model.hiddenNodes[nj].activation_fn;
                wfni = activation_fn.getFunctionIndex(weightFn);

                #get connection info
                active = model.connActive_HO[ni][nj];
                connWeight = model.connWeight_HO[ni][nj];

                if active:
                    conn = Edge.Edge(nodei,nodej,wfni,connWeight);
                    scene.addItem(conn);


                    
        #set node positions randomly
        energy = 50;
        posX = 0.0;
        posY = 0.0;
        
        for n in nodes:
            posX = numpy.random.rand() * 50;
            posY = numpy.random.rand() * 50;
            
            n.setPos(posX,posY);
               
    def itemMoved(self):
        if not self.timerId:
            self.timerId = self.startTimer(250 / 25);
            
            
    def timerEvent(self, event):
        nodes = [item for item in self.scene().items() if isinstance(item, Node)]

        for node in nodes:
            node.calculateForces()

        itemsMoved = False
        for node in nodes:
            if node.advance():
                itemsMoved = True

        if not itemsMoved:
            self.killTimer(self.timerId)
            self.timerId = 0
        
    def wheelEvent(self, event):
        self.scaleView(math.pow(2.0, -event.delta() / 240.0))

   

    def scaleView(self, scaleFactor):
        factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()

        if factor < 0.07 or factor > 100:
            return

        self.scale(scaleFactor, scaleFactor)
        
    def drawBackground(self, painter, rect):
        
        #define scene rect
        sceneRect = self.sceneRect()
        # Text.
        textRect = QtCore.QRectF(sceneRect.left() + 4, sceneRect.top() + 4,
                sceneRect.width() - 4, sceneRect.height() - 4)
        message = "(i) Click and drag the nodes around, and zoom with the " \
                "mouse wheel or the '+' and '-' keys"


        font = painter.font()
        font.setBold(False)
        font.setPointSize(7)
        painter.setFont(font)
        painter.setPen(QtCore.Qt.lightGray)
        painter.drawText(textRect.translated(2, 2), message)
        painter.setPen(QtCore.Qt.black)
        painter.drawText(textRect, message)
        
        
    def saveImage(self,paintedScene):
        
        w = paintedScene.width();
        h = paintedScene.height();
            
        paintedScene.setBackgroundBrush(self.whitebrush);
        view = QGraphicsView();
        view.setScene(paintedScene)
      
    
    
        outputimg = QPixmap(w, h)
        painter = QPainter(outputimg)
        targetrect = QRectF(0, 0, w, h)
        sourcerect = QRect(0, 0, w, h)
        view.render(painter, targetrect, sourcerect)
        filename = "NET_VIS_"+ commons.getStrTimeStamp()+".jpg";
        outputimg.save(filename, "JPG")
    
        painter.end()

        
if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv);
    QtCore.qsrand(QtCore.QTime(0,0,0).secsTo(QtCore.QTime.currentTime()));

    widget = GraphWidget();
    widget.show();
    

    sys.exit(app.exec_());

