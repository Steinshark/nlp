import pygame
import random
from colors import *
import directions
import coords
import collectables
import images
import terrain
import numpy
import os.path

class Map():
    """Contains array of Cells and properties representing the map as a whole"""
    CELLDAMAGEDROUGHNESS = 40
    CELLBURNINGCOST = 200
    DIRTYCACHE = False

    def __init__(self, mapdict):
        """Load the map from image files"""
        self.mapdef = mapdict
        self.startpos = tuple(mapdict['startpos'])
        try:
            self.gemgodefs = mapdict['gemgos']
        except KeyError:
            self.gemgodefs = {}
        self.origcoins = 0
        self.burningtiles = set()
        self.fusetiles = set()
        self.damagedtiles = {}
        self.crcount = 0

        terrainfilepath = os.path.join('map', mapdict['dir'], mapdict['terrainfile'])
        itemfilepath = os.path.join('map', mapdict['dir'], mapdict['itemfile'])
        for filepath in terrainfilepath, itemfilepath:
            if not os.path.isfile(filepath):
                raise Exception("%s is not a file" %filepath)

        groundimage = pygame.image.load(terrainfilepath).convert()
        groundarray = pygame.surfarray.pixels2d(groundimage)
        wronggroundcolours = numpy.setdiff1d(groundarray, terrain.colorlist(groundimage))
        if wronggroundcolours.size:
            print wronggroundcolours
            raise Exception("Unexpected value in %s" %terrainfilepath)
        collectablesimage = pygame.image.load(itemfilepath).convert()
        collectablesarray = pygame.surfarray.pixels2d(collectablesimage)
        collectablescolorsflat = pygame.surfarray.map_array(collectablesimage, numpy.array(collectables.mapcolor.keys()))
        wrongcollectablecolors = numpy.setdiff1d(collectablesarray, collectablescolorsflat)
        if wrongcollectablecolors.size:
            print wrongcollectablecolors
            raise Exception("Unexpected value in %s" %itemfilepath)
        self.size = groundimage.get_rect().size

        nbrcount = numpy.zeros(self.size, dtype=numpy.uint)
        for i in [(1,1,1), (1,0,2), (-1,1,4), (-1,0,8)]:
            nbrcount += (groundarray == numpy.roll(groundarray,  i[0], axis=i[1])) * i[2]

        randomgrid = numpy.random.randint(256, size=self.size)
        self.cellarray = numpy.empty(self.size, dtype=terrain.celldtype)
        for color_type in terrain.color_typeindex(groundimage):
            istype = groundarray == color_type[0]
            self.cellarray[istype] = terrain.typeindextocell[color_type[1]]
            for level in ['groundimage', 'topimage']:
                images = terrain.typeindextocell[color_type[1]][level]
                if not images:
                    continue
                nbrimagelists = filter(lambda a: isinstance(a, list), images)
                if nbrimagelists:
                    # Non-directional sprites are ignored if one or more directional sets provided.
                    zimages = zip(*nbrimagelists)
                    for i in range(16):
                        ind = (nbrcount == i) & istype
                        self.cellarray[level][ind] = numpy.choose(randomgrid[ind], zimages[i], mode='wrap')
                else:
                    self.cellarray[level][istype] = numpy.choose(randomgrid[istype], images, mode='wrap')

        for color_collectable in collectables.mapcolor.iteritems():
            color = pygame.surfarray.map_array(collectablesimage, numpy.array([color_collectable[0]]))
            self.cellarray['collectableitem'][collectablesarray == color] = color_collectable[1]

        self.origcoins = (self.cellarray['collectableitem'] == collectables.COIN).sum()

    def __getitem__(self, coord):
        """Get map item with [], wrapping"""
        return self.cellarray[coord[0]%self.size[0]][coord[1]%self.size[1]]

    def __setitem__(self, coord, value):
        """Set map item with [], wrapping"""
        self.cellarray[coord[0]%self.size[0]][coord[1]%self.size[1]] = value

    def sprites(self, coord):
        sprites = []
        def addsprite(image, layer):
            sprites.append((image,
                            (coord[0]*images.TILESIZE + (images.TILESIZE-image.get_width())/2,
                             coord[1]*images.TILESIZE + (images.TILESIZE-image.get_height())/2),
                            layer))
        cell = self[coord]
        if not cell['explored']:
            addsprite(images.Unknown, -20)
            return sprites

        offsetsprite = cell['groundimage']
        if offsetsprite:
            addsprite(offsetsprite, cell['layeroffset']-10)
        offsetsprite = cell['topimage']
        if offsetsprite:
            addsprite(offsetsprite, cell['layeroffset']+10)

        if coord in self.damagedtiles:
            addsprite(self.damagedtiles[coords.mod(coord, self.size)], -3)
        if coord in self.fusetiles:
            for direction in directions.CARDINALS:
                nbrcoord = coords.modsum(coord, direction, self.size)
                if nbrcoord in self.fusetiles or self[nbrcoord]['collectableitem'] == collectables.DYNAMITE:
                    addsprite(images.Fuse[direction], -2)
        if cell['collectableitem'] != 0:
            addsprite(images.Collectables[cell['collectableitem']], -1)
        if coord in self.burningtiles:
            addsprite(random.choice(images.Burning), -1)
        return sprites

    def placefuse(self, coord):
        self.fusetiles.add(coords.mod(coord, self.size))

    def ignitefuse(self, coord):
        coord = coords.mod(coord, self.size)
        if self[coord]['collectableitem'] == collectables.DYNAMITE:
            self.detonate(coord)
        if not coord in self.fusetiles:
            return False
        openlist = set()
        openlist.add(coord)
        while len(openlist) > 0:
            curpos = openlist.pop()
            if curpos in self.fusetiles:
                self.fusetiles.remove(curpos)
            for nbrpos in coords.neighbours(curpos):
                if self[nbrpos]['collectableitem'] == collectables.DYNAMITE:
                    self.detonate(nbrpos)
                nbrpos = coords.mod(nbrpos, self.size)
                if nbrpos in self.fusetiles:
                    openlist.add(nbrpos)

    def destroy(self, coord):
        """Change cell attributes to reflect destruction"""
        cell = self[coord]
        if not cell['destructable']:
            return False
        self.damagedtiles[coords.mod(coord, self.size)] = random.choice(images.Damaged)
        cell['hasroof'] = False
        cell['name'] = "shattered debris"
        cell['collectableitem'] = 0
        cell['top'] = False
        cell['fireignitechance'] = 0
        cell['fireoutchance'] = 1
        cell['transparent'] = True
        cell['solid'] = False
        cell['roughness'] = max(100, cell['roughness'] + Map.CELLDAMAGEDROUGHNESS)
        cell['topimage'] = 0
        return True

    def ignite(self, coord, multiplier=1, forceignite=False):
        """Start a fire at coord, with chance cell.firestartchance * multiplier"""
        coord = coords.mod(coord, self.size)
        cell = self[coord]
        if coord in self.fusetiles:
            self.ignitefuse(coord)
        if cell['collectableitem'] == collectables.DYNAMITE:
            self.detonate(coord)
        if forceignite or random.random() < cell['fireignitechance'] * multiplier:
            if cell['fireignitechance'] > 0:
                self.destroy(coord)
            self.burningtiles.add(coord)
            return True
        return False

    def detonate(self, coord):
        """Set off an explosion at coord"""
        def blam(epicentre):
            self[epicentre]['collectableitem'] = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    curpos = coords.sum(epicentre, (dx, dy))
                    if not self.ignite(curpos, multiplier=3):
                        self.destroy(curpos)
        if not self[coord]['destructable']:
            return False
        blam(coord)
        return True

    def update(self):
        """Spread fire, potentially other continuous map processes"""
        for tile in self.burningtiles.copy():
            cell = self[tile]
            for nbrpos in coords.neighbours(tile):
                self.ignite(nbrpos)
            if random.random() < cell['fireoutchance']:
                self.burningtiles.remove(tile)
