# NLP Final Group Project : Dependency parser of Nivre for the Korean Language version 
# Yonsei University
# Groupe 8

import matplotlib.pyplot as pyplot
import random as r
import numpy as np
from math import *

# Test
L = [0]*2
print(L)

T = np.arange(8)
print(T)

T = np.ones((2, 3))
print(T)

T = np.random.rand(3)
print(T)

# Vertical
# np.concatenate((a,b),axis=0)
# Horizontal
# np.concatenate((a,b),axis=1)

# Trees - print trees
from anytree import Node, RenderTree
udo = Node("Uddazdazo")
marc = Node("Madzadazrc", parent=udo)
lian = Node("Liadazdan", parent=marc)
dan = Node("Dadazdazn", parent=udo)
jet = Node("Jedazdazt", parent=dan)
jan = Node("Jadddn", parent=dan)
joe = Node("Jdazdzaoe", parent=dan)

print(udo)
# Node('/Udo')
print(joe)
# Node('/Udo/Dan/Joe')

# for pre, fill, node in RenderTree(udo):	
# 	print("%s%s" % (pre, node.name))

# from anytree.exporter import DotExporter
# # graphviz needs to be installed for the next line!
# DotExporter(udo).to_picture("udo.png")


def main():
	print(1)

if __name__ == '__main__':
    main()
