import os

# obj_extractor has two functions that supports user to save either .obj file of smpl vertices,
# or .txt file of pose (3x3 rotmat) and shape (10 float32)

# Function for creating .obj file.
def writeObjFile(path, vertices, faces, objnum):
    """
        args:
            path: absolute path to project folder
            vertices: smpl vertices (about 6900)
            faces: smpl face for vertex indices
            objnum: used for deciding file name
    """
    with open(os.path.join(path, 'Output/res' + str(objnum) + ".obj"), 'w+') as objFile:
        for vert in vertices:
            objFile.write("v ")
            objFile.write(str(vert[0]))
            objFile.write(" ")
            objFile.write(str(-vert[1]))
            objFile.write(" ")
            objFile.write(str(-vert[2]))
            objFile.write("\n")
        objFile.write("s off\n")
        for face in faces:
            objFile.write("f ")
            objFile.write(str(face[0] + 1))
            objFile.write(" ")
            objFile.write(str(face[1] + 1))
            objFile.write(" ")
            objFile.write(str(face[2] + 1))
            objFile.write("\n")
        objFile.close()

# Function for creating .txt file.
def writePoseNShapeFile(path, pose, shape, objnum):
    """
        args:
            path: absolute path to project folder
            pose: smpl pose rot mat (1, 24, 3, 3)
            shape: smpl beta (1, 10)
            objnum: used for deciding file name
    """
    jointCount = 24
    matSize = 3

    with open(os.path.join(path, 'PSOutput/res' + str(objnum) + ".txt"), 'w+') as objFile:
        #objFile.write("rotmats \n")
        for g in range(jointCount):
            for i in range(matSize):
                for j in range(matSize):
                    objFile.write(str(pose[0, g, i, j]))
                    objFile.write(" ")
            objFile.write("\n")
        #objFile.write("pose done\n")
        #objFile.write("betas \n")
        for beta in shape[0]:
            objFile.write(str(beta))
            objFile.write("\n")
        objFile.close()