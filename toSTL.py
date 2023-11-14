import numpy as np
import csv
from stl import mesh
import sys
import os

#tutej ustawiamy krawędzie na 1
def is_edge(data, i, j):
    nrows, ncols = data.shape
    if np.isnan(data[i][j]):
        return False 
    if i == 0 or i == nrows - 1 or j == 0 or j == ncols - 1:
        return True   
    neighbours = [data[i-1][j], data[i+1][j], data[i][j-1], data[i][j+1]]
    return any(np.isnan(n) for n in neighbours)


def makeSTL(data,file_path, file_name="printME.stl", scale_factor=4):
    points = []
    file_path = os.path.join(os.path.dirname(file_path), file_name)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if is_edge(data, i, j):
                data[i][j] = 1 


    for i in range(data.shape[0] - 1):
        for j in range(data.shape[1] - 1):
            z1 = data[i][j]
            z2 = data[i][j + 1]
            z3 = data[i + 1][j]
            z4 = data[i + 1][j + 1]
    

#tu dodajemy do poinsów nie tylko wartości pixeli, ale też "1" stanowjące podstawe
            if not np.isnan(z1) and not np.isnan(z2) and not np.isnan(z3):
                points.append([[j, i, z1 * scale_factor], [j + 1, i, z2 * scale_factor], [j, i + 1, z3 * scale_factor]])       
                points.append([[j, i, max_z], [j + 1, i, max_z], [j, i + 1, max_z]])


            max_z = scale_factor 

            if not np.isnan(z2) and not np.isnan(z3) and not np.isnan(z4):
                points.append([[j + 1, i, z2 * scale_factor], [j + 1, i + 1, z4 * scale_factor], [j, i + 1, z3 * scale_factor]] )
                points.append([[j + 1, i, max_z], [j + 1, i + 1, max_z], [j, i + 1, max_z]])

    your_mesh = mesh.Mesh(np.zeros(len(points), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(points):
        for j in range(3):
            your_mesh.vectors[i][j] = np.array(f[j])
    your_mesh.save(file_path)



