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

    def viewJack(self):
        #print self.model.vines.text
        self.screen.fill(color)
        pygame.draw.rect(self.screen, pygame.Color(255,0,0),#Jack
                         (self.model.jack.x, self.model.jack.y, 10, 10))
        pygame.draw
        pygame.display.update()
    
    def viewVines(self):
        for vine in self.model.vines:
            pygame.draw.rect(self.screen, pygame.Color(0,255,0), #vines
                     (vine.x, vine.y, vine.height, vine.width))
        pygame.draw
        pygame.display.update()
#####################################################################        
class Vines: 
    #Abstracting for now --- INSERT VINE CODE HERE
#    def __init__(self,x,y):
#        self.text = '""fucking vines"-that can be interpreted two ways"-Pratool Gadtaula'
#        self.width = 10                                     ###
#        self.height = 10                                    ###
#        self.x=x
#        self.y=y
#        #vineback = pygame.Surface(self.width, self.height)  ###
#        #pygame.draw.rect(vineback, pygame.Color(0,255,0))   ###

#####ES#####
    def __init__(self, vcolor = (0,150,100)):
        self.color = vcolor
        self.width = WIDTH
        self.height = HEIGHT
    
    def increasepix(self, obj):
        """Increase thickness of the vine"""
        if obj < 0:
            newobj = obj-4
            return int(newobj)
        if obj >= 0:
            newobj = obj+4
            return int(newobj)

    def drawSolu(self):
        """Draw the correct solution path outputted above from 'build_maze'"""                    
        vinemaze = MakeMaze()
        solulist = vinemaze.build_maze(self.width-4, 0, (8,24))
        
        #draw starting point square at bottom right corner
        start = pygame.Rect(self.width, self.height, -4, -4)
        pygame.draw.rect(gameView.background, self.color, start)
        dimenlist = []
        
        #determine the rectangle's starting coordinate, width, and height
        for i in range(1, len(solulist)):
            cur_coord = solulist[i]
            prev_coord = solulist[i-1]
            RectW = prev_coord[0] - cur_coord[0]
            RectH = prev_coord[1] - cur_coord[1]

            #keep track of coordinates where non-overlapping vines occur            
            if RectH > 0:
                dimenlist.append(prev_coord)
            
            #Increase thickness of vines, depending on if they lie vertical or horizontal            
            if abs(RectW) < abs(RectH):
                newRectW = self.increasepix(RectW)
                vine = pygame.Rect(cur_coord[0], cur_coord[1], newRectW, RectH)              
            else:
                newRectH = self.increasepix(RectH)
                vine = pygame.Rect(cur_coord[0], cur_coord[1], RectW, newRectH)
            pygame.draw.rect(gameView.background, self.color, vine)
        
        #fill in the missing corners due to non-overlapping vines
        for i in range(1, len(dimenlist)):
            prevW = dimenlist[i][0]
            prevH = dimenlist[i][1]
            if prevW > 0:
                fillcoord = (prevW, prevH)
                fill = pygame.Rect(fillcoord[0], fillcoord[1], 4, 4)
                pygame.draw.rect(gameView.background, self.color, fill)

        gameView.screen.blit(gameView.background,(0,0))
        pygame.display.update()
        
    def drawDead(self):
        """Draw vines to the list of deadend nodes assigned under 'dead_ends'"""
        vinemaze = MakeMaze()
        for x in xrange(10):  
            mazelist = vinemaze.dead_ends()
            #determine the rectangle's starting coordinate, width, and height
            for j in range(1, len(mazelist)):
                coord = mazelist[j]
                precoord = mazelist[j-1]
                RectW2 = precoord[0] - coord[0]
                RectH2 = precoord[1] - coord[1]
                
                 #Increase thickness of vines, depending on if they lie vertical or horizontal 
                if abs(RectW2) < abs(RectH2):
                    newRectW2 = self.increasepix(RectW2)
                    vine2 = pygame.Rect(coord[0], coord[1], newRectW2, RectH2)
                    pygame.draw.rect(gameView.background, self.color, vine2)
                else:
                    newRectH2 = self.increasepix(RectH2)
                    vine2 = pygame.Rect(coord[0], coord[1], RectW2, newRectH2)
                    pygame.draw.rect(gameView.background, self.color, vine2)
            
            gameView.screen.blit(gameView.background,(0,0))  
                
        pygame.display.update()

#####ES##############################################################
class MakeMaze:
    """calculates a maze path using the node points at which
        the maze will change directions"""
    def __init__(self):
        self.state = 'up'
        self.width = WIDTH
        self.height = HEIGHT
    
    def turn_R(self, pos, step):
        return (pos[0] + step, pos[1])

    def turn_L(self, pos, step):
        return (pos[0] - step, pos[1])
        
    def go_up(self, pos, step):
        return (pos[0], pos[1] - step)
    
    def build_maze(self, start, end, steprange): 
        """Builds exactly one randomized path leading from the starting 
        point (x coordinate) to the end point (y coordinate),
        and outpits a list of nodes based on the stepincreases of
        (min steprange, max steprange)"""
    
        self.directions = [self.turn_R, self.turn_L, self.turn_L, self.go_up]
        currentpos = (start, self.height-4)
        list_of_points = [currentpos]
    
        if currentpos[1] == 0 and currentpos[0] == 0:
            return list_of_points
        #Before the end point is reached, move in a random direction
        while currentpos[1] > end or currentpos[0] > end:
            f = random.choice(self.directions)
            currentpos = f(currentpos, random.randint(steprange[0], steprange[1]))
            
            #Check if the vine is out of bounds
            if currentpos[0] < 0:
                currentpos = (0, currentpos[1])
            if currentpos[0] > self.width:
                currentpos = (self.width-4, currentpos[1])
            if currentpos[1] < 0:
                currentpos = (currentpos[0], 0)
            #Add the point to the list of points the vine goes through
            list_of_points.append(currentpos)
        return list_of_points

    def dead_ends(self):
        """Creates random additional deadend vines that are not solutions"""
        rn_W = random.randint(int(self.width*.1), 560) #start 10% up from bottom of screen
        rn_H = random.randint(0, 560)
        return self.build_maze(rn_W, rn_H, (8, 28))
#####ES##############################################################    
class Model:
    def __init__(self):
        #Keeps track of vine locations 
        #self.vine_locations=[(10,10),(10,20),(10,30),(20,10),(20,20),(20,30),(10,40),(10,50),(10,60),(20,40),(20,50),(20,60)]
        self.vine_locations= MakeMaze.build_maze
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
######ES#####
def main():
    """generate the random maze and draw it to the GUI background"""
    maze = Vines()
    maze.drawSolu()
    maze.drawDead()
######ES#####

if __name__ == "__main__":
    gameModel = Model()
    gameView = View(gameModel)
    gameCont = Controller(gameModel)
    main()
    running=True
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT: 
                running = False
            else:
                gameCont.handle_key_event(event)
        gameView.viewJack()
        time.sleep(0.001)
    pygame.quit()
#rect = pygame.Rect(rectangle)
#image = pygame.Surface(rect.size).convert()