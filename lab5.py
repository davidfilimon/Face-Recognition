import cv2
import numpy as np
import os
from numpy import linalg as la
import time

def creareMatrAntr(caleDS, nrPers, nrPozePers, rezolutie):
    A = np.zeros((rezolutie, nrPers * nrPozePers))
    for i in range(1, nrPers + 1):
        caleFolder = os.path.join(caleDS, f's{i}')
        for j in range(1, nrPozePers + 1):
            calePoza = os.path.join(caleFolder, f'{j}.pgm')
            if not os.path.exists(calePoza):
                print(f"Fisierul nu exista: {calePoza}")
                continue
            poza = cv2.imread(calePoza, 0)
            if poza is None:
                print(f"Nu s-a putut citi imaginea: {calePoza}")
                continue
            pozaVect = np.reshape(poza, (rezolutie,))
            A[:, (i - 1) * nrPozePers + j - 1] = pozaVect
    return A

def nn(A, pozaCautata, norma):
    z = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        z[i] = norma_dif(A[:, i], pozaCautata, norma)
    pozitia = np.argmin(z)
    return pozitia


def eigenPreproc(A, k):
    media = np.mean(A, axis=1)
    A_centrat = (A.T - media).T
    # c = np.dot(A_centrat, A_centrat.T)
    l = np.dot(A_centrat.T, A_centrat)
    d, v = la.eig(l)
    v = A_centrat@v
    idx = np.argsort(d)[::-1][:k]
    HQPB = v[:, idx]
    proiectii = np.dot(A_centrat.T, HQPB)
    return media, HQPB, proiectii

def norma_dif(x, y, norma):
    if norma == '1':
        return la.norm(x - y, 1)
    elif norma == '2':
        return la.norm(x - y)
    elif norma == 'inf':
        return la.norm(x - y, np.inf)
    elif norma == 'cos':
        return 1 - np.dot(x, y) / (la.norm(x) * la.norm(y))
    else:
        raise ValueError("Norma specificata nu este valida.")
    
def statistici(caleDS, nrPers, nrPoze, nrPozeTest, rezolutie, k, norma):
    start_preproc = time.time()
    A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)
    media, HQPB, proiectii = eigenPreproc(A, k)
    timp_preproc_total = time.time() - start_preproc

    corecte = 0
    total = nrPers * nrPozeTest
    start_proc = time.time()

    for i in range(1, nrPers + 1):
        caleFolder = os.path.join(caleDS, f's{i}')
        for j in range(nrPoze + 1, nrPoze + 1 + nrPozeTest):
            calePozaTest = os.path.join(caleFolder, f'{j}.pgm')
            if not os.path.exists(calePozaTest):
                print(f"Fisierul nu exista: {calePozaTest}")
                continue

            poza_test = cv2.imread(calePozaTest, 0)
            if poza_test is None:
                print(f"Nu s-a putut citi imaginea de test: {calePozaTest}")
                continue

            poza_test = np.reshape(poza_test, (rezolutie,))
            poza_test = poza_test - media
            proiectie_poza_test = np.dot(poza_test, HQPB)

            pozitia = nn(proiectii.T, proiectie_poza_test, norma)
            persoana_identificata = pozitia // nrPoze + 1

            if persoana_identificata == i:
                corecte += 1

    timp_proc_total = time.time() - start_proc

    rata_recunoastere = (corecte / total) * 100

    return {
        "Rata de recunoastere (%)": rata_recunoastere,
        "Timp preprocesare (s)": timp_preproc_total,
        "Timp procesare (s)": timp_proc_total
    }


if __name__ == "__main__":
    nrPoze = 8
    k = 20  
    nrPers = 40
    rezolutie = 10304
    caleDS = "D:\\att_faces"
    caleTest = "D:\\att_faces\\s10\\10.pgm"

    # Creare matrice de antrenare
    A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)

    # Preprocesare
    media, HQPB, proiectii = eigenPreproc(A, k)

    # Vectorizare si normalizare imagine test
    if not os.path.exists(caleTest):
        print(f"Fisierul de test nu exista: {caleTest}")
        exit(1)

    poza_cautata = cv2.imread(caleTest, 0)
    if poza_cautata is None:
        print(f"Nu s-a putut citi imaginea de test: {caleTest}")
        exit(1)

    poza_cautata = np.reshape(poza_cautata, (rezolutie,))
    poza_cautata = poza_cautata - media
    proiectie_poza_cautata = np.dot(poza_cautata, HQPB)

    # Cautare imagine similara
    pozitia = nn(proiectii.T, proiectie_poza_cautata, norma='1')
    persoana = pozitia // nrPoze + 1

    # Afisare rezultate
    print(f"Imaginea gasita se potriveste cu persoana {persoana}")

    statistici_rezultate = statistici(
        caleDS="D:\\att_faces",
        nrPers=40,
        nrPoze=8,
        nrPozeTest=2,
        rezolutie=10304,
        k=20,
        norma='1'
    )
    print(statistici_rezultate)

