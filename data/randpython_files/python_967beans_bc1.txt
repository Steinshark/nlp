"""
Created on Sun Oct 12 19:49:18 2014
@author: Elena Shaw & Brandon Chiou
"""

import numpy
import pygame, sys, random, time
from pygame.locals import *
Clock = pygame.time.Clock()
color = pygame.Color(0,160,255)
height = 560
width = 560

#screen = pygame.display.set_mode((560,560))

class View:
    """Initializes and creates game"""
    def __init__(self, model):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.model = model
        pygame.display.set_caption('Jack and the Bean Stalk')
        background = pygame.Surface(self.screen.get_size())
        self.background = background.convert()
        #draw model.vines screen.blit(model.vines)
        #This class (View class) will print self.text/the ACTUAL vines

    def viewprint(self):
        #print self.model.vines.text
        self.screen.fill(color)
        pygame.draw.rect(self.screen, pygame.Color(255,0,0), #Jack
                         (self.model.jack.x, self.model.jack.y, 10, 10))
        for vine in self.model.vines:
            pygame.draw.rect(self.screen, pygame.Color(0,255,0), #vines
                         (vine.x, vine.y, vine.height, vine.width))
        pygame.draw
        pygame.display.update()
#####################################################################        
class Vines: 
    #Abstracting for now --- INSERT VINE CODE HERE
    def __init__(self,x,y):
        self.text = '""fucking vines"-that can be interpreted two ways"-Pratool Gadtaula'
        self.width = 10                                     ###
        self.height = 10                                    ###
        self.x=x
        self.y=y
        #vineback = pygame.Surface(self.width, self.height)  ###
        #pygame.draw.rect(vineback, pygame.Color(0,255,0))   ###
#####################################################################    
class Model:
    def __init__(self):
        #Keeps track of vine locations 
        self.vine_locations=[(10,10),(10,20),(10,30),(20,10),(20,20),(20,30),(10,40),(10,50),(10,60),(20,40),(20,50),(20,60)]
        #Creates empty list called vines        
        self.vines=[]
        for location in self.vine_locations:
            #Creates instance in list of vines object
            self.vines.append(Vines(location[0],location[1]))
        self.jack = Jack()

#####################################################################                
class Controller:
    def __init__(self, model):
        self.model = model    
    def pressbutton(self):
        startbutton = pygame.image.load('start.jpeg').convert()
        youwin = pygame.image.load('youwin.jpeg').convert()
        
        
    def handle_key_event(self, event):        
        if event.type == KEYUP:
            if event.key == pygame.K_UP:
                if (self.model.jack.y > 0 and ((self.model.jack.x, self.model.jack.y-10) not in (self.model.vine_locations))): self.model.jack.y -= 10
            if event.key == pygame.K_DOWN:
                if (self.model.jack.y < height - 10 and ((self.model.jack.x, self.model.jack.y+10) not in (self.model.vine_locations))): self.model.jack.y += 10
            if event.key == pygame.K_LEFT:
                if (self.model.jack.x > 0  and ((self.model.jack.x-10, self.model.jack.y) not in (self.model.vine_locations))): self.model.jack.x -= 10
            if event.key == pygame.K_RIGHT:
                if (self.model.jack.x < width - 10 and ((self.model.jack.x+10, self.model.jack.y) not in (self.model.vine_locations))): self.model.jack.x += 10
            
#####################################################################
class Jack:
    def __init__(self):
        self.x = height - 10
        self.y = width - 10

#####################################################################               
if __name__ == "__main__":
    gameModel = Model()
    gameView = View(gameModel)
    gameCont = Controller(gameModel)
    running=True
    
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT: 
                running = False
            else:
                gameCont.handle_key_event(event)
        gameView.viewprint()
        time.sleep(0.001)
    pygame.quit()