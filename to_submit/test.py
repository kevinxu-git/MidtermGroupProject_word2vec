f = open("test.txt", "r")
contenu = f.read()
f.close()

L = [[1], [1]]
f = open("testWrite.txt", "w")
f.write(str(L))
f.close()
print(contenu)
