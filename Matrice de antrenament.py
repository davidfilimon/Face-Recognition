import cv2, os, numpy as np, matplotlib.pyplot as plt
from statistics import mode
import time

def creareMatrAntr(caleDS, nrPers, nrPozePers, rezolutie) :
    A = np.zeros((rezolutie, nrPers * nrPozePers))
    for i in range (1, nrPers + 1) :
        caleFolder = caleDS + '/s' + str(i) + '/'
        for j in range (1,nrPozePers + 1) :
            calePoza = caleFolder + str(j) + '.pgm'
            poza = cv2.imread(calePoza,0)
            poza = np.array(poza)

            pozaVect = np.reshape(poza,(10304,))
            A[:,(i-1) * nrPozePers + j - 1] = pozaVect 

    return A

def createTestMatrix(caleDS, nrPers, nrPozeTest, rezolutie):
    B = np.zeros((rezolutie, nrPers * nrPozePers))
    for i in range (1, nrPers + 1) :
        caleFolder = caleDS + '/s' + str(i) + '/'
        for j in range (9, 11) :
            calePoza = caleFolder + str(j) + '.pgm'
            poza = cv2.imread(calePoza,0)
            poza = np.array(poza)

            pozaVect = np.reshape(poza,(10304,))
            B[:,(i-1) * nrPozeTest + j - 9] = pozaVect 
    
    return B
    
def NN(A, photoToTest, norm):
      
    z = np.zeros(len(A[0]))
    for i in range(len(z)):
        if norm >= 1 and norm <= 2:
            z[i] = np.linalg.norm(photoToTest - A[:, i], norm)
        if norm == 3:
            z[i] = np.linalg.norm(photoToTest - A[:, i], np.inf)
        if norm == 4:
            z[i] = (1 - np.dot(A[:,i], photoToTest)) / (np.linalg.norm(A[:,i], 2) * np.linalg.norm(photoToTest, 2))
            
    return np.argmin(z)
    
    
    
def kNN(A, photoToTest, norm, k):
   
    z = np.zeros(len(A[0]))
    for i in range(len(z)):
        if norm >= 1 and norm <= 2:
            z[i] = np.linalg.norm(photoToTest - A[:, i], norm)
        if norm == 3:
            z[i] = np.linalg.norm(photoToTest - A[:, i], np.inf)
        if norm == 4:
            z[i] = (1 - np.dot(A[:,i], photoToTest)) / (np.linalg.norm(A[:,i], 2) * np.linalg.norm(photoToTest, 2))
        
    return mode(np.argsort(z)[:k] // 8 + 1)
    
    
def calculate_statistics_NN(B, A):
    f = open('NN.txt', 'x')
    f = open('NN.txt', 'a')
    
    for norm in range(1,5):
        contor_poze = 0 
        suma_timp = 0
        for i in range (len(B[0])):
            photo = np.array(B[:, i])
            x = norm
            start_time = time.time();
            index_poza = NN(A, photo, x)
            end_time = time.time();
            
            exec_time = end_time - start_time
            suma_timp += exec_time
            
            eticheta_clasa = index_poza // 8 + 1 #kNN -> index_poza
            if eticheta_clasa == i // 2 + 1:
                contor_poze += 1
            
        RR = contor_poze/80
        AQT = suma_timp/80
        
        f.writelines(['{}\n'.format(norm), '{}\n'.format(RR * 100), '{}\n'.format(AQT),'\n'])
    
    f.close()
    print('done')
    
    
def calculate_statistics_kNN(B, A):
    f = open('kNN.txt', 'x')
    f = open('kNN.txt', 'a')
    
    for k in range(3,10,2):
        f.write('{}\n'.format(k))
        for norm in range(1,5):
            contor_poze = 0 
            suma_timp = 0
            for i in range (len(B[0])):
                photo = np.array(B[:, i])
                x = norm
                y = k
                start_time = time.time();
                index_poza = kNN(A, photo, x, y)
                end_time = time.time();
                
                exec_time = end_time - start_time
                suma_timp += exec_time
                
                eticheta_clasa = index_poza#kNN -> index_poza
                if eticheta_clasa == i // 2 + 1:
                    contor_poze += 1
                
            RR = contor_poze/80
            AQT = suma_timp/80
            
            f.writelines(['\t{}\n'.format(norm), '\t{}\n'.format(RR * 100), '\t{}\n'.format(AQT)])
    
    f.close()
    print('done')    


nrPers = 40
nrPozePers = 8
rezolutie = 112*92
nrPozeTest = 2

caleDS=r'D:\\att_faces'

A = creareMatrAntr(caleDS, nrPers, nrPozePers, rezolutie)
B = createTestMatrix(caleDS, nrPers, nrPozeTest, rezolutie)

calculate_statistics_kNN(B, A)



#A:/Facultate/Anul 3/Pattern recognition/poze/s1/10.pgm