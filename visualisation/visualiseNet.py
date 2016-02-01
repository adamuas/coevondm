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
from Node import Node;
from Edge import Edge;
from ndm import ndm;
from node import *;
import constants;
import time;

import commons;

class GraphWidget(QtGui.QGraphicsView):
    def __init__(self, net, title = None):
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
        self.net = net;
        self.whitebrush = QBrush(Qt.white);
        self.redbrush = QBrush(Qt.red)
        self.drawNetwork(self.gscene, self.net);
        
        #setttings of the window
        self.scale(1.2,1.2);
        self.setMinimumSize(640, 480);
        
        
        if title == None:
            self.setWindowTitle("Network Visualisation");
        else:
            self.setWindowTitle("Network Visualisation -" + title);
        
    
        
    def drawNetwork(self,scene, net):
        """ Draws the visualisation of the network"""
        nodes = [];
        
        #add nodes
        for n in net.nodes:
            
            #if the node is an instance of inactive node, skip it.
            if isinstance(n,inactive_node) is True:
                #print "inactive node";
                continue;
         
            else:
                #print "active node";
                #node representation
                n_rep = Node(self,n.node_id, n.map_id, n.type, n.nodeFn);
                #add to nodes
                nodes.append(n_rep);
            
                #add to scene
                scene.addItem(n_rep);
                
            
        #create connections
        for i in xrange(len(nodes)):
            i_id = nodes[i].node_id;
            i_mapId = nodes[i].map_id;
            
            
            for j in xrange(len(nodes)):
                j_id = nodes[j].node_id;
                j_mapId = nodes[j].map_id;
                
                #debug
                #print "i",i_mapId,"j",j_mapId ,"=", net.conn_active[i_mapId][j_mapId];
                
                if net.conn_active[i_mapId][j_mapId] == constants.ACTIVE:
                    #get weight function
                    weightFn = net.nodes[j_id].weightFn;
                    
                    #get connection weight
                    weight = net.connMatrix[i_mapId][j_mapId];
                    conn = Edge(nodes[i],nodes[j], weightFn, weight);
                    #add to scene
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

