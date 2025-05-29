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
    # Preprocesare date de antrenament
    start_preproc = time.time()
    A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)
    reprezentari_clase = eigenPreprocClase(A, nrPers, nrPoze, k)
    timp_preproc_total = time.time() - start_preproc

    corecte = 0
    total = nrPers * nrPozeTest
    start_proc = time.time()

    # Testare fiecare imagine din setul de test
    for i in range(1, nrPers + 1):  # Iterăm prin clase
        caleFolder = os.path.join(caleDS, f's{i}')
        for j in range(nrPoze + 1, nrPoze + nrPozeTest + 1):  # Imagini de test
            calePozaTest = os.path.join(caleFolder, f'{j}.pgm')
            if not os.path.exists(calePozaTest):
                print(f"Fisierul nu exista: {calePozaTest}")
                continue

            poza_test = cv2.imread(calePozaTest, 0)
            if poza_test is None:
                print(f"Nu s-a putut citi imaginea de test: {calePozaTest}")
                continue

            poza_test = np.reshape(poza_test, (rezolutie,))
            distante_minime = []

            # Comparăm poza de test cu reprezentările claselor
            for clasa, reprezentare in reprezentari_clase.items():
                media_clasa = reprezentare["mean_face"]
                HQPB_clasa = reprezentare["eigenfaces"]

                poza_test_cent = poza_test - media_clasa
                proiectie_test = np.dot(poza_test_cent, HQPB_clasa)

                proiectii_clasa = reprezentare["proiectii"]
                diferenta = np.min([norma_dif(proiectie_test, p, norma) for p in proiectii_clasa])
                distante_minime.append((clasa, diferenta))

            # Identificăm clasa cu distanța minimă
            clasa_identificata = min(distante_minime, key=lambda x: x[1])[0]

            if clasa_identificata == i:
                corecte += 1

    timp_proc_total = time.time() - start_proc

    rata_recunoastere = (corecte / total) * 100

    return {
        "Rata de recunoastere (%)": rata_recunoastere,
        "Timp preprocesare (s)": timp_preproc_total,
        "Timp procesare (s)": timp_proc_total
    }


def eigenPreprocClase(A, nrPers, nrPozePers, k):
    reprezentari_clase = {}
    media_generala = np.mean(A, axis=1)  # media tuturor imaginilor
    A_centrat = (A.T - media_generala).T # centrare fata de media generala

    l = np.dot(A_centrat.T, A_centrat)  
    d, v = la.eigh(l) 
    idx = np.argsort(d)[::-1][:k]  # selectam cele mai mari k valori
    v = np.dot(A_centrat, v[:, idx])  # transformam in spatiul original
    v = v / np.linalg.norm(v, axis=0)  # normalizare

    for i in range(nrPers):
        start_idx = i * nrPozePers
        end_idx = start_idx + nrPozePers
        A_clasa = A[:, start_idx:end_idx]
        media_clasa = np.mean(A_clasa, axis=1)  
        A_clasa_centrat = (A_clasa.T - media_clasa).T
        proiectii_clasa = np.dot(A_clasa_centrat.T, v)

        reprezentari_clase[i + 1] = {
            "mean_face": media_clasa,
            "eigenfaces": v,
            "proiectii": proiectii_clasa
        }

    return reprezentari_clase


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
    reprezentari_clase = eigenPreprocClase(A, nrPers, nrPoze, k)

    # Vectorizare si normalizare imagine test
    if not os.path.exists(caleTest):
        print(f"Fisierul de test nu exista: {caleTest}")
        exit(1)

    poza_cautata = cv2.imread(caleTest, 0)
    if poza_cautata is None:
        print(f"Nu s-a putut citi imaginea de test: {caleTest}")
        exit(1)

    poza_cautata = np.reshape(poza_cautata, (rezolutie,))
    distante_minime = []

    for clasa, reprezentare in reprezentari_clase.items():
        media_clasa = reprezentare["mean_face"]
        HQPB_clasa = reprezentare["eigenfaces"]

        poza_cautata_cent = poza_cautata - media_clasa
        proiectie_test = np.dot(poza_cautata_cent, HQPB_clasa)

        proiectii_clasa = reprezentare["proiectii"]
        diferenta = np.min([norma_dif(proiectie_test, p, '1') for p in proiectii_clasa])
        distante_minime.append((clasa, diferenta))

    persoana = min(distante_minime, key=lambda x: x[1])[0]

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
