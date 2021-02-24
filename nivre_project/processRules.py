rule = "(S (NP_SBJ 나/NP + 는/JX) (VNP 돈/NNG + 이/VCP + 다/EF + ./SF))"

filtered = ""
R = []
for char in rule:
	if char == '(':
		char = '( '
	elif char == ')':
		char = ' )'
	filtered = filtered + char
filtered_list = filtered.split()
print(filtered_list)

arcs = []
for i in range(len(filtered_list)):
	if filtered_list[i] == '(':
		arcs += []

print(arcs)

rule = ["S", ["NP_SB", ["NP", "JX"]], ["VNP", ["NNG", "VCP", "EF", "SF"]]]
tags = ["S", "NP_SB", "NP", "JX", "VNP", "NNG", "VCP", "EF", "SF"]


int2tag = {}
tag2int = {}
for i, tag in enumerate(tags):
    tag2int[tag] = i
    int2tag[i] = tag
print(int2tag)

rule2int = []




import numpy as np
# Count the number of neighbours
nTags = len(tags)
trainTags = np.zeros((nTags, nTags))
# print(trainTags)

rule = [0, [1, [2, 3]], [4, [5, [6, 7, 8]]]]

def trainTag(rule, trainTags):
	# print(trainTags)
	# print(rule)
	# if type(rule) is not list:
	# 	print(T)
	# 	return trainTags
	if type(rule[1]) is list and (all(isinstance(x, int) for x in rule[1])):
		for i in range(len(rule[1])):
			trainTags[rule[0]][rule[1][i]] +=1
		return trainTags
	if type(rule[0]) is not list and type(rule[1]) is list:
		for i in range(1, len(rule)):
			trainTags[rule[0]][rule[i][0]] += 1
			trainTags = trainTag(rule[i], trainTags)
	return trainTags

t = trainTag(rule, trainTags)
print(t)

