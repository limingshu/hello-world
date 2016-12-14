import pickle
import random
import math
import numpy as np
import neuralNetwork
import motorType
from math import exp
from random import seed
from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
PI = math.pi
Reward = [[1,5,0,0,0],[0,5,15,0,0],[1,5,0,-1,-1],[1,5,0,0,0],[2,5,0,0,-1]]
a = 0.8
d = 0.8
def distance(point1,point2):
    return math.sqrt((point1[0] - point2[0])*(point1[0] - point2[0]) + (point1[1] - point2[1])*(point1[1] - point2[1]))
def calAngle(position):
    ang1 = np.arctan2(*position[::-1])
    ang2 = np.arctan2(*(1,0)[::-1])
    diff = ang1 - ang2
    return diff

def calPIRange(direction):
    while(1):
        if direction > PI:
            direction - 2*PI
        if direction < -1*PI:
            direction + 2*PI
        if direction > -1*PI and direction < PI:
            return direction
def almostEqual(direction1,direction2):
    if (direction1 < (direction2 + PI/3)) and (direction1 > (direction2 - PI/3)):
        return True
    return False
def calVector(angle):
    return (math.cos(angle),math.sin(angle))
def normalizeVector(vector):
    if vector[0] == 0 and vector[1] ==0:
        return vector
    else:
        vLength = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
        return (vector[0]/vLength, vector[1]/vLength)
def calPosition(current, speed, direction):
    return ((current[0] + int(round(speed * math.cos(direction)))), (current[1] + int(round(speed * math.sin(direction)))))      

directionOrder = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
directionAngleOrder = [-1*3*PI/4,PI,3*PI/4,-1*PI/2,1*PI/2,-1*PI/4,0,PI/4]

class FoodGradient:
    def __init__(self):
        self.gradient = 0
        self.direction = 0

class Pheronmone:
    def __init__(self, pheronmone):
        self.pheronmone = pheronmone
        # pheromone[0] group info
        # pheronmone[1] whether find food
        # pheronmone[2] whether food exist
           
class Pixel(object):
    def __init__(self):
        self.foodsGrad = []
        self.pheroForGroup1 = [0,0,0]
        self.pheroForGroup2 = [1,0,0]
    
class Food:
    def __init__(self):
        self.edibleBy = 2
        self.position = (0,0)
        self.units = 1
        self.energyPerUnit = 0.0
    
class Map:
    def __init__(self, width, height, foods):
        self.width = width
        self.height = height
        self.foods = foods
        self.pixels = np.empty((width, height), dtype = object)
        self.initializePixels()           #create matrix for every pixel

    def initializePixels(self):
        for i in range(0, self.width):
            for j in range(0, self.height):
                self.pixels[i][j] = Pixel()
    
    def calFoodGradient(self, food, i, j):
        foodGrad = FoodGradient()
        distance = math.sqrt(i * i + j * j)
        foodGrad.gradient = food.units/distance/10
        foodGrad.direction = calAngle((i/distance,j/distance))
        return foodGrad
    
    def insertFood(self, food): #for every new food create a new food gradient map
        foodPosition = food.position
        self.foods.append(food)
        for x in range(0,self.width):
            for y in range(0,self.height):
                self.pixels[x][y].foodsGrad.append(FoodGradient())
        for i in range(-6, 7):
            for j in range(-6, 7):
                if 0 < foodPosition[0] + i < self.width and 0 < foodPosition[1] + j < self.height:
                    if i != 0 or j != 0:
                        self.pixels[foodPosition[0] + i][foodPosition[1] + j].foodsGrad[-1] = (self.calFoodGradient(food,i,j))
                    if i == 0 and j == 0:
                        foodGrad = FoodGradient()
                        foodGrad.gradient = 1
                        foodGrad.direction = calAngle((0,0))
                        self.pixels[foodPosition[0] + i][foodPosition[1] + j].foodsGrad[-1] = (foodGrad)   

    def updateFood(self, food):
        foodPosition = food.position
        for x in range(len(self.foods)):
            if self.foods[x].position == foodPosition:
                for i in range(-6, 7):
                    for j in range(-6, 7):
                        if 0 < foodPosition[0] - i < self.width - 1 and 0 < foodPosition[1] - j < self.height - 1:
                            if i != 0 or j != 0:
                                self.pixels[foodPosition[0] - i][foodPosition[1] - j].foodsGrad[x] = (self.calFoodGradient(food,i,j))
                            if i == 0 and j == 0:
                                foodGrad = FoodGradient()
                                foodGrad.gradient = 1
                                foodGrad.direction = calAngle((0,0))
                                self.pixels[foodPosition[0] + i][foodPosition[1] + j].foodsGrad[x] = (foodGrad)
    
    def deleteFood(self, food):
        for i in range(0,len(self.foods)):
            if self.foods[i].position == food.position:
                self.foods.remove(self.foods[i]) #remove empty food from foods
                for x in range(self.width): #remove food gradient accordingly
                    for y in range(self.height):
                        del self.pixels[x][y].foodsGrad[i]     
                break
        
    def updatePheronmone(self, position, pheronmone, originPheronmone):
        originPheronmone[1] = originPheronmone[1] + pheronmone.pheronmone[1]
        originPheronmone[2] = originPheronmone[2] + pheronmone.pheronmone[2]
        return originPheronmone
                           
    def updateGroupPheronmone(self, position, pheronmone):
        if 5 < position[0] < self.width - 5 and 5 < position[1] < self.height -5:
            if pheronmone.pheronmone[0] == 0:
                originPheronmone = self.pixels[position[0]][position[1]].pheroForGroup1
                self.pixels[position[0]][position[1]].pheroForGroup1 =  self.updatePheronmone(position, pheronmone, originPheronmone)
            elif pheronmone.pheronmone[0] == 1:
                originPheronmone = self.pixels[position[0]][position[1]].pheroForGroup2
                self.pixels[position[0]][position[1]].pheroForGroup2 = self.updatePheronmone(position, pheronmone, originPheronmone)  
            
    def pheronmoneEvaporation(self):
        for i in range(self.width):
            for j in range(self.height):
                self.pixels[i][j].pheroForGroup1[1] = max(0, self.pixels[i][j].pheroForGroup1[1] - 0.01)
                self.pixels[i][j].pheroForGroup1[2] = max(0, self.pixels[i][j].pheroForGroup1[2] - 0.001)
                self.pixels[i][j].pheroForGroup2[1] = max(0, self.pixels[i][j].pheroForGroup2[1] - 0.01) 
                self.pixels[i][j].pheroForGroup2[2] = max(0, self.pixels[i][j].pheroForGroup2[2] - 0.001)
class Box:
    def __init__(self):
        self.animats = []
    def addAnimat(self,animat):
        self.animats.append(animat)
    def clearBox(self):
        self.animats = []
class Environment:
    def __init__(self, width, height, foods, initNum0, initNum1):
        self.width = width
        self.height = height
        self.map = Map(self.width, self.height,[])
        self.animatsGroup1 = []
        self.animatsGroup2 = []
        self.eachGenerationNum = [] #For group 1 and 2
        self.eachGenerationSurvivalNum1 = [] #For group 1 and 2
        self.eachGenerationSurvivalNum2 = [] #For group 1 and 2
        self.nest1 = Nest(5000,(200,200))
        self.nest2 = Nest(5000,(80,80))
        self.updateCount = 0
        self.collisionBox = []
        self.initializeBox()
        self.collisionPair = []
        female = self.createAnimatsInNest(0)
        female.turnFemale()
        self.nest1.updateFemaleAnimat(female)
        for i in range(initNum0):
            self.animatsGroup1.append(self.createAnimatsInNest(0))
        female = self.createAnimatsInNest(1)
        female.turnFemale()
        self.nest2.updateFemaleAnimat(female)
        for i in range(initNum1):
            self.animatsGroup2.append(self.createAnimatsInNest(1))
        for food in foods:
            self.map.insertFood(food)
    
    def initializeBox(self):
        for i in range(3):
            self.collisionBox.append([])
            for j in range(3):
                box = Box()
                self.collisionBox[i].append(box)
                
    def createAnimatsInNest(self,group):
        motor = motorType.MotorType()
        motor.updateMotor(0)
        if group == 0:
            animat = Animat(0,200,0.5,motor,self.nest1)
        else: 
            animat = Animat(1,300,0.5,motor,self.nest2)
        return animat
   
    def sensor(self, animat):
        sensorInput = []
        for i in range(-1,2):
            for j in range(-1,2):
                if i != 0 or j != 0:
                    if 0 < animat.position[0] + i < self.width - 1 and 0 < animat.position[1] + j < self.height - 1:
                        detectPixel = self.map.pixels[animat.position[0] + i][animat.position[1] + j]
                        if len(detectPixel.foodsGrad) > 1:
                            choosedFood = random.choice(detectPixel.foodsGrad)
                        elif len(detectPixel.foodsGrad) <1:
                            choosedFood = FoodGradient()
                        else:
                            choosedFood = detectPixel.foodsGrad[0]
                        #food gradient as sensor input
                        sensorInput.append(choosedFood.gradient)
                        sensorInput.append(choosedFood.direction)
                        #pheronmone as sensor input
                        if animat.group == 0:
                            sensorInput.append(detectPixel.pheroForGroup1[1])
                            sensorInput.append(detectPixel.pheroForGroup1[2])
                        else:
                            sensorInput.append(detectPixel.pheroForGroup2[1])
                            sensorInput.append(detectPixel.pheroForGroup1[2])
                    else:
                        sensorInput.append(0)
                        sensorInput.append(0)
                        sensorInput.append(0)
                        sensorInput.append(0)
        return sensorInput
        
    def findFood(self, animat):
        for x in range(len(self.map.foods)):
            food = self.map.foods[x]
            foodR = 1
            for i in range(-1*foodR,foodR):
                for j in range(-1*foodR,foodR):
                    if animat.position == (food.position[0] + i, food.position[1] + j) and food.edibleBy != animat.group:
                        self.map.foods[x].units = max(self.map.foods[x].units - 1,0)
                        animat.carryFood = True
                        if self.map.foods[x].units == 0:
                            self.map.deleteFood(self.map.foods[x])
                            animat.foodEmpty = True
                        return True
        return False
        
    def calBox(self,animat):
        boxIndex = []
        x = animat.position[0]
        y = animat.position[1]
        if x < 270 and x >= 10:
            a1 = x/90
            a2 = (x-10)/90
        elif x < 10:
            a1 = x/90
            a2 = a1
        else:
            a2 = min(2,(x-10)/90)
            a1 = a2
        if y < 270 and y >= 10:
            b1 = y/90
            b2 = (y-10)/90
        elif y < 10:
            b1 = y/90
            b2 = b1
        else:
            b2 = min(2,(y-10)/90)
            b1 = b2
        if a1 != a2 and b1 != b2:
            boxIndex = [(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
        elif a1 != a2 or b1 != b2:
            boxIndex = [(a1,b1),(a2,b2)]
        else:
            boxIndex = [(a1,b1)]
        return boxIndex
            
    def updateBox(self):
        for i in range(3):
            for j in range(3):
                self.collisionBox[i][j].clearBox()
        for i in range(len(self.animatsGroup1)):
            index = self.calBox(self.animatsGroup1[i])
            for idx in index:
                self.collisionBox[idx[0]][idx[1]].addAnimat(self.animatsGroup1[i])
        for i in range(len(self.animatsGroup2)):
            index = self.calBox(self.animatsGroup2[i])
            for idx in index:
                self.collisionBox[idx[0]][idx[1]].addAnimat(self.animatsGroup2[i])
    
    def calCollision(self):
        for i in range(3):
            for j in range(3):
                anim = self.collisionBox[i][j].animats
                for idx_1 in range(len(anim)):
                    for idx_2 in range(idx_1+1,len(anim)):
                        if distance(anim[idx_1].position,anim[idx_2].position) < 2 and anim[idx_1].inNest() == False and anim[idx_2].inNest() == False:
                            self.collisionPair.append((self.collisionBox[i][j].animats[idx_1],self.collisionBox[i][j].animats[idx_2]))
    
    def executeCollide(self):
        for i in range(len(self.collisionPair)):
            a1 = self.collisionPair[i][0]
            a2 = self.collisionPair[i][1]
            if a1.group != a2.group:
                minIdx = 0
                maxIdx = 0
                if a1.curEnergy > a2.curEnergy:
                    minIdx = 1
                else:
                    maxIdx = 1
                self.collisionPair[i][maxIdx].curEnergy = self.collisionPair[i][maxIdx].curEnergy - self.collisionPair[i][minIdx].curEnergy*0.2
                self.collisionPair[i][maxIdx].fight()
                self.collisionPair[i][minIdx].dead()
#            else:
#                print("collision")
#                self.collisionPair[i][0].collide()
#                self.collisionPair[i][1].collide()          
    
    def foodFromDeath(self,animat):
        newFood = Food()
        newFood.edibleBy = animat.group
        newFood.units = 5
        newFood.position = animat.position
        newFood.energyPerUnit = 1
        return newFood
    
    def update(self):
        self.updateCount += 1
        isMate = not bool(self.updateCount%200)
        if isMate:
            print("start mate!")
            print("energy in nest", self.nest1.energyInNest )
            self.eachGenerationSurvivalNum1.append(len(self.animatsGroup1))
            for i in range(self.eachGenerationSurvivalNum1[-1]):
                if self.animatsGroup1[i].generation >= 3 or self.animatsGroup1[i].curEnergy < 0:
                    self.animatsGroup1.remove(self.animatsGroup1[i])
                child = self.animatsGroup1[i].mate(self.nest1,len(self.animatsGroup1))
                if child != None:
                    self.animatsGroup1.append(child)  
            self.nest1.female.generation = self.nest1.female.generation + 1
            if self.nest1.female.generation >= 5 and len(self.animatsGroup1) > 0:
                self.nest1.female = random.choice(self.animatsGroup1)
                self.animatsGroup1.remove(self.nest1.female)
                self.nest1.female.generation = 0
                self.nest1.female.turnFemale()
            self.eachGenerationSurvivalNum2.append(len(self.animatsGroup2))
            for i in range(self.eachGenerationSurvivalNum2[-1]):
                if self.animatsGroup2[i].generation >= 3 or self.animatsGroup2[i].curEnergy < 0:
                    self.animatsGroup2.remove(self.animatsGroup2[i])
                child = self.animatsGroup2[i].mate(self.nest2,len(self.animatsGroup2))
                if child != None:
                    self.animatsGroup2.append(child)
            self.nest2.female.generation = self.nest2.female.generation + 1
            if self.nest2.female.generation >= 5 and len(self.animatsGroup2) > 0:
                self.nest2.female = random.choice(self.animatsGroup2)
                self.animatsGroup2.remove(self.nest2.female)
                self.nest2.female.generation = 0
                self.nest2.female.turnFemale()
            if len(self.animatsGroup1) == 0:
                print("group1 all die!")
            else:
                print("group1 new child", abs(self.eachGenerationSurvivalNum1[-1] - len(self.animatsGroup1)))
                print("group1 old generation",self.eachGenerationSurvivalNum1[-1])
                print("female generation",self.nest1.female.generation)
            if len(self.animatsGroup2) == 0:
                print("group2 all die!")
            else:
                print("group2 new child", abs(self.eachGenerationSurvivalNum2[-1] - len(self.animatsGroup2)))
                print("group2 old generation",self.eachGenerationSurvivalNum2[-1])
                print("female generation",self.nest2.female.generation)
            return
        else:
            i = 0
            j = 0              
            while(i<len(self.animatsGroup1)):
                animat = self.animatsGroup1[i]
                if animat.generation >= 3 or animat.curEnergy < 0:
                    if animat.generation == 4:
                        self.map.insertFood(self.foodFromDeath(self.animatsGroup1[i]))
                    self.animatsGroup1.remove(animat)
                else:
                    i = i+1
                    isFindFood = self.findFood(animat)
                    animat.update(self.sensor(animat),isFindFood,False,False, self.width, self.height)
                    self.map.updateGroupPheronmone(animat.position,animat.leavePheronmone(animat.carryFood,animat.foodEmpty))
            while(j<len(self.animatsGroup2)):
                animat = self.animatsGroup2[j]
                if animat.generation >= 3 or animat.curEnergy < 0:
                    if animat.generation == 4:
                        self.map.insertFood(self.foodFromDeath(self.animatsGroup2[j]))
                    self.animatsGroup2.remove(animat)
                else:
                    j = j+1
                    isFindFood = self.findFood(animat)
                    animat.update(self.sensor(animat),isFindFood,False,False, self.width,self.height)
                    self.map.updateGroupPheronmone(animat.position,animat.leavePheronmone(animat.carryFood,animat.foodEmpty))
            for food in self.map.foods:
                self.map.updateFood(food)
            self.map.pheronmoneEvaporation() 
            self.updateBox()
            self.calCollision()
            self.executeCollide()
            
class Animat:
    speeds = [0,2,1,0,0,0]
    energyCosts = [0.5,0.8,1.2,10,-5,5]
    def __init__(self,group,curEnergy,initialEnergyRatio,motor,nest):
        self.gender = False
        self.generation = 0
        self.group = group
        self.nest = nest
        self.curEnergy = curEnergy
        self.initialEnergyRatio = initialEnergyRatio
        self.maxEnergy = self.calMaxEnergy()
        self.requireEat = False
        self.motor = motor
        self.position = nest.position
        self.isCollide = False
        self.speed = self.calSpeed()
        self.direction = random.uniform(-1*PI,PI)
        #neural network
        self.net = neuralNetwork.NeuralNetwork(4,5,4)
        self.gene = self.net.toList()    
        self.praise = 0.2
        self.moveThreshold = 0.5
        self.eatThreshold = 0.5
        self.carryFood = False
        self.wantFollowTrail = False
        self.isBack = False
        self.wantExploitation = True
        self.isFull = False
        self.outsideNestTime = 0
        self.exploitationSteps= 0
        self.exploitationPos = random.uniform(0,1)
        self.findFoodPheromone = False
        self.foodEmpty = False

    def calMaxEnergy(self):
        i = self.generation
        if i==0:
            maxEnergy = 350
        elif i==1:
            maxEnergy = 500
        elif i==2:
            maxEnergy = 400
        return maxEnergy
    
    def calSpeed(self):
        return self.speeds[self.motor.state]
    
    def updateEnergy(self): ##call every frame
        self.curEnergy = min(self.curEnergy - self.energyCosts[self.motor.state],self.calMaxEnergy)

    def nextPosition(self,position):
        self.position = position
    
    def collisionDetected(self): ##update states and energy cost mode, if the other ant die, substract the energy from this ant directly
        self.motor.updateMotor(3)
        self.calSpeed(self)
#        self.updateEnergy(self)
    def eatFood(self,energy): #decide how much to eat outside
        self.motor.updateMotor(4)
        self.calSpeed()
        self.curEnergy = min(self.curEnergy + energy,self.maxEnergy)
        self.nest.energyInNest = self.nest.energyInNest - 2
        if self.curEnergy == self.maxEnergy:
            self.isFull = True
#        self.updateEnergy(self)
    def turnFemale(self):
        self.gender = True
        self.position = self.nest.position
        self.gene = self.gene + self.gene
    def inNest(self):
        return self.nest.isInNest(self.position)     
        
    def mate(self,nest,num):
        self.position = nest.position
        self.generation = self.generation + 1
        self.motor.updateMotor(5)
        self.nest.energyInNest = self.nest.energyInNest - 5
        self.calSpeed()
        self.gene = self.net.toList()
        motor = motorType.MotorType()
        matePos = 0.3
        if num < 20:
            matePos = min(matePos + 0.2,1)
        if num > 80:
            matePos = max(matePos - 0.2,0.3)
        if self.praise >= 0.1 and self.generation > 0 or random.uniform(0,1) < matePos:
            child = Animat(self.group,200,0,motor,nest)
            for i in range(len(self.gene)):
                child.gene[i] = random.choice([self.gene[i],nest.female.gene[i]])
            self.motor.updateMotor(0)
            child.net.toWeight(child.gene)
            child.exploitationPos = self.exploitationPos
        else:
            child = None
        return child
            
    def update(self, sensorInput, isFindFood,isCreateNewGeneration, isCollision, width, height):  
#        print(self.position)
        nnInput = [self.curEnergy/self.maxEnergy, self.generation, self.initialEnergyRatio,self.nest.energyInNest/500]
        nnDecision = self.net.calResult(nnInput)
        self.wantMove = (abs(nnDecision[0]) > self.moveThreshold)
        if self.wantExploitation and not self.findFoodPheromone:
            if random.uniform(0,1) > self.exploitationPos and self.exploitationSteps > 4:
                self.wantExploitation = False
        else:
            self.wantExploitation = True
        self.wantBack = (abs(nnDecision[2]) > 0.5)
        self.wantEat = (nnDecision[3] > 0.5) #only work in nest
#        print(self.direction)
        if self.curEnergy <= 0:
            self.naturalDead()
        if self.nest.isInNest(self.position): # in nest
            self.curEnergy = self.curEnergy - self.energyCosts[0]
            self.foodEmpty = False
#            print("in nest")
            self.outsideNestTime = 0
            self.isBack = False
            if self.carryFood:
                self.carryFood = False #put down food
                self.praise = self.praise + 1
                self.nest.energyInNest = self.nest.energyInNest + 30
                self.motor.updateMotor(0)
                return 
            if not self.isFull:
                if self.wantEat:
                    if self.curEnergy == self.maxEnergy:
                        self.net.feedback(0,3,nnInput)
                    self.motor.updateMotor(4)
                    self.eatFood(5)
                    self.praise = max(self.praise - 0.05,0)
                    return
                else:
                    if self.curEnergy < 0.5 * self.maxEnergy:
                        self.net.feedback(1,3,nnInput)
                        return
            if self.wantMove:
                if self.curEnergy < 100:
                    self.net.feedback(0,0,nnInput)
                self.motor.updateMotor(1)
                self.initialEnergyRatio = self.curEnergy/self.maxEnergy
                self.updateUtils(width,height)
                return                
            else:
                self.net.feedback(1,0,nnInput)
                
        else:  # out of nest
#                if collision:
#                    return  ###collision
            self.outsideNestTime += 1
            if self.carryFood:
#                print('carry')
                self.motor.updateMotor(2)
                self.backHome()
                self.updateUtils(width,height)
                return
            else:
                self.motor.updateMotor(1)
                direction = self.detectPhero(sensorInput)
                if self.curEnergy < 0.3 * self.maxEnergy:
                    self.net.feedback(1,2,nnInput)
                if self.isBack or self.wantBack:
                    if self.curEnergy > 0.5 * self.maxEnergy or self.outsideNestTime < 20:
#                        print(self.curEnergy,"lots energy")
                        self.net.feedback(0,2,nnInput)
#                    print('back')
                    self.backHome()
                    self.updateUtils(width,height)                      
                elif isFindFood:
                    self.carryFood = True
                    self.action = 3
#                    print('Find!')
                    self.motor.updateMotor(2)
                    self.backHome()
                    self.updateUtils(width,height)
                    return
                elif self.searchFood(sensorInput):
                    self.findFoodPheromone = False
                    self.action = 2
                    self.updateUtils(width,height)
                    return
                elif self.wantExploitation or self.findFoodPheromone:
                    self.followTrail(direction)
                    self.exploitationSteps += 1
                    self.updateUtils(width,height)
                    return
                else:
                    self.explore()
                    self.exploitationSteps = 0
                    self.updateUtils(width,height)
                    return
                    
#                elif self.wantExploitation or self.findFoodPheromone:
#                    self.exploitationSteps += 1
##                    print('exploitation')
#                    self.followTrail(sensorInput)
#                    self.updateUtils(width,height)
#                    return
#                else:
#                    self.explore(sensorInput)
#                    self.exploitationSteps = 0
##                    print('explore')
#                    self.updateUtils(width,height)
#                    return
        if isCreateNewGeneration:
            self.generation = self.generation + 1
            self.motor.startMate()
            #CREATE NEW GENERATION
            self.position = self.nest.position
            return          
        
    def updateUtils(self,width,height):
        self.speed = self.calSpeed()       
        self.position = calPosition(self.position,self.speed, self.direction)
        if self.meetWall(width, height):
            self.direction = self.direction - PI
        self.updateEnergy()

    def backHome(self): 
        self.isBack = True
        self.direction = calAngle((self.nest.position[0] - self.position[0],self.nest.position[1] -  self.position[1]))       
    def searchFood(self,sensorInput):
        maxFoodGradient = 0
        direction = None
        for i in range(8):
            foodGradient = sensorInput[4*i]
            foodDirection = sensorInput[4*i + 1]
            if foodGradient > maxFoodGradient:
                maxFoodGradient = foodGradient
                direction = foodDirection
        if direction != None:
            self.direction = direction 
            return True
        else:
            return False

    def detectPhero(self,sensorInput):
        maxPhero = 2.8
        direction = None        
        for i in range(8):
            if self.nest.isInNest(self.position):
                pheroStrength = sensorInput[4 * i + 2]              
            else:
                pheroStrength = sensorInput[4 * i + 2]
            pheroIsEmpty = sensorInput[4 * i + 3]
            if pheroStrength > maxPhero and pheroIsEmpty <= 0:
                maxPhero = pheroStrength
                self.findFoodPheromone = True
                direction = directionAngleOrder[i] 
        return direction   
             
    def followTrail(self,direction):
        if direction != None:
            if almostEqual(direction,calAngle((self.nest.position[0] - self.position[0],self.nest.position[1] -  self.position[1]))):
                direction = direction + PI
            self.direction = direction
        else:
            self.direction = self.direction

    def explore(self):
        self.direction = self.direction + np.random.normal(0.93,0.22)* random.choice([-1,1])
        
    def leavePheronmone(self, isCarryFood, isFoodEmpty):
        if isFoodEmpty:
            return Pheronmone([self.group, 0, 1])
        elif isCarryFood:
            return Pheronmone([self.group, 3, 0])
        else:
            return Pheronmone([self.group, 0.01, 0])
        
    def meetWall(self,width,height):
        if 0 < self.position[0] < width and 0 < self.position[1] < height:
            return False
        else:
            return True
    def updateNest(self,nest):
        self.nest = nest
        
    def returnNest(self):
        return self.nest
    def dead(self):
        self.curEnergy = 0
        self.generation = 4
        
    def naturalDead(self):
        self.curEnergy = 0
        self.generation = 3
    def fight(self): ##update states and energy cost mode, if the other ant die, substract the energy from this ant directly
        self.motor.updateMotor(3)
        self.calSpeed()
    
    def collide(self):
        self.direction = self.direction + random.randrange(-1,1)
class Nest:
    def __init__(self, energyInNest, position):
        self.energyInNest = energyInNest
        self.position = position
        self.R = 2
    
    def isInNest(self, position):
        for i in range(-2,3):
            for j in range(-2,3):   
                if (position[0] + i,position[1] + j) == self.position:
                    return True
        return False
        
    def updateFemaleAnimat(self,animat):
        self.female = animat
