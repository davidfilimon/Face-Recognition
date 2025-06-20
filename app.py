import time
from PyQt6 import uic
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QMessageBox, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QImage, QPixmap
from interface import Ui_MainWindow
import numpy as np
import cv2
import sys
from statistics import mode
from numpy import linalg as la
import matplotlib.pyplot as plt

def creareMatrAntr(caleDS, nrPers, nrPozePers, rezolutie):
    A = np.zeros((rezolutie, nrPers * nrPozePers))
    for i in range(1, nrPers + 1):
        caleFolder = os.path.join(caleDS, f's{i}')
        for j in range(1, nrPozePers + 1):
            calePoza = os.path.join(caleFolder, f'{j}.pgm')
            poza = cv2.imread(calePoza, 0)
            pozaVect = np.reshape(poza, (rezolutie,))
            A[:, (i - 1) * nrPozePers + j - 1] = pozaVect
    return A


def createTestMatrix(caleDS, nrPers, nrPozeTest, rezolutie):
    B = np.zeros((rezolutie, nrPers * nrPozeTest))
    for i in range(1, nrPers + 1):
        caleFolder = os.path.join(caleDS, f's{i}')
        for j in range(9, 11):
            calePoza = os.path.join(caleFolder, f'{j}.pgm')
            poza = cv2.imread(calePoza, 0)
            pozaVect = np.reshape(poza, (rezolutie,))
            B[:, (i - 1) * nrPozeTest + j - 9] = pozaVect
    return B


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Pattern Recognition")

        # parameters
        self.caleDS = "D:\\att_faces\\att_faces"
        self.training = 8
        self.num_images_per_person = 8
        self.num_test_images = 2
        self.resolution = 112 * 92
        self.num_people = 40
        self.scene = self.ui.pozaGasita.scene()

        # initialize data matrices
        self.A = None
        self.B = None
        self.load_data()

        
# button connect
        self.ui.bd1Button.clicked.connect(lambda: self.set_training(6))
        self.ui.bd2Button.clicked.connect(lambda: self.set_training(8))
        self.ui.bd3Button.clicked.connect(lambda: self.set_training(9))
        self.ui.selectButton.clicked.connect(self.calculeaza)
        self.ui.searchButton.clicked.connect(self.cautaPoza)
        self.ui.statsButton.clicked.connect(self.afiseazaStatistici)
       
        
    def norma_dif(self, x, y, norma):
        if norma == 1:
            return la.norm(x - y, 1)
        elif norma == 2:
            return la.norm(x - y)
        elif norma == 3:
            return la.norm(x - y, np.inf)
        elif norma == 4:
            return 1 - np.dot(x, y) / (la.norm(x) * la.norm(y))
        else:
            raise ValueError("Norma specificata nu este valida.")
    
    def set_training(self, value):
        self.num_images_per_person = value
        self.load_data()
    
    def afisare_imagine_cautata(self, file_path):
        image = QImage(file_path)
        if image.isNull():
            QMessageBox.warning(self, "Atentie", "Imaginea nu a putut fi deschisa.")
            return
    
        pixmap = QPixmap.fromImage(image)
        scene = QGraphicsScene(self)
        scene.addPixmap(pixmap)
        self.ui.pozaSelectata.setScene(scene)
    
    def load_data(self):
        self.A = creareMatrAntr(self.caleDS, self.num_people, self.num_images_per_person, self.resolution)
        self.B = createTestMatrix(self.caleDS, self.num_people, self.num_test_images, self.resolution)
    
    def cautaPoza(self):
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecteaza o poza", self.caleDS, "Images (*.pgm)")
        
        
        if file_path:
            self.afisare_imagine_cautata(file_path)
            self.poza_selectata = file_path
        
        
    def afisare_imagine_calculata(self, file_path):
        
        image = QImage(file_path)
        print(file_path)
        if image.isNull():
            QMessageBox.warning(self, "Atentie", "Imaginea nu a putut fi deschisa.")
            return
    
        pixmap = QPixmap.fromImage(image)   
        scene = QGraphicsScene(self)
        scene.addPixmap(pixmap)
        self.ui.pozaGasita.setScene(scene)


            
    def calculeaza(self):
        self.norma_selectata = 0;
        if self.ui.normaManhattan.isChecked():
            self.norma_selectata = 1
        elif self.ui.normaEuclidian.isChecked():
            self.norma_selectata = 2
        elif self.ui.normaInfinite.isChecked():
            self.norma_selectata = 3
        elif self.ui.normaCosine.isChecked():
            self.norma_selectata = 4
        else: print("Selectati o norma.")
        
        metoda_selectata = 0
        if self.ui.nnButton.isChecked():
            self.metoda_selectata = 1
            self.nn(self.A, self.poza_selectata, self.norma_selectata)           
        elif self.ui.knnButton.isChecked():
            self.metoda_selectata = 2
            self.k = int(self.ui.kBox.toPlainText())
            self.knn(self.A, self.poza_selectata, self.norma_selectata, self.k)
        elif self.ui.eigenfacesButton.isChecked():
            self.metoda_selectata = 3
            self.k = int(self.ui.kBox.toPlainText())
            self.eigenCalcul(self.caleDS, self.num_people, self.num_images_per_person, self.num_test_images, self.resolution, self.k, self.norma_selectata, self.poza_selectata)
        elif self.ui.rcButton.isChecked():
            self.metoda_selectata = 4
            self.k = int(self.ui.kBox.toPlainText())
            self.calcul_eigen_clase(self.caleDS, self.num_people, self.num_images_per_person, self.num_test_images, self.resolution, self.k, self.norma_selectata, self.scene, self.poza_selectata)
        elif self.ui.lanczosButton.isChecked():
            self.metoda_selectata = 5
            self.k = int(self.ui.kBox.toPlainText())
            self.lanczos_calcul(self.caleDS, self.num_people, self.num_images_per_person, self.num_test_images, self.resolution, self.k, self.norma_selectata, self.poza_selectata)
        else: print("Selectati o metoda.")   
        
        
    def afiseazaStatistici(self):
        if self.metoda_selectata == 1:
            print(self.calculate_statistics_NN(self.B, self.A))
        elif self.metoda_selectata == 2:
            print(self.calculate_statistics_kNN(self.B, self.A))
        elif self.metoda_selectata == 3:
            print(self.eigenStatistici(self.caleDS, self.num_people, self.num_images_per_person, self.num_test_images, self.resolution, self.k, self.norma_selectata, self.poza_selectata))
        elif self.metoda_selectata == 4:
            print(self.clase_statistici(self.caleDS, self.num_people, self.num_images_per_person, self.num_test_images, self.resolution, self.k, self.norma_selectata, self.poza_selectata))
        elif self.metoda_selectata == 5:
            print(self.lanczos_statistici(self.caleDS, self.num_people, self.num_images_per_person, self.num_test_images, self.resolution, self.k, self.norma_selectata, self.poza_selectata))
        else: print("Selectati o metoda.")
    
    # nn
    
    def nn(self, A, poza_selectata, norma):
        z = np.zeros(A.shape[1])
        
       
        if isinstance(poza_selectata, str) and not os.path.exists(poza_selectata):
            print(f"Fisierul {poza_selectata} nu exista. Sarim peste aceasta imagine.")
            return -1 
        
        
        if isinstance(poza_selectata, str):
            poza_selectata = cv2.imread(poza_selectata, 0)
        
        poza_selectata = np.reshape(poza_selectata, (self.resolution,))
        
        for i in range(A.shape[1]):
            if norma == 1:
                z[i] = np.linalg.norm(poza_selectata - A[:, i], 1)
            elif norma == 2:
                z[i] = np.linalg.norm(poza_selectata - A[:, i])
            elif norma == 3:
                z[i] = np.linalg.norm(poza_selectata - A[:, i], np.inf)
            elif norma == 4:
                z[i] = 1 - np.dot(A[:, i], poza_selectata) / (np.linalg.norm(A[:, i]) * np.linalg.norm(poza_selectata))
        
        index_gasit = np.argmin(z)  
        cale_imagine = f"D:\\att_faces\\att_faces\s{index_gasit // 8 + 1}\\1.pgm"
        self.afisare_imagine_calculata(cale_imagine)  
        return index_gasit
    
    def calculate_statistics_NN(self, B, A):
        print("nn")
        norm_values = []  
        recognition_rates = []  
        avg_times = []  
        
        for norm in range(1, 5):
            contor_poze = 0 
            suma_timp = 0
            for i in range(len(B[0])):
                photo = np.array(B[:, i])
                x = norm
                start_time = time.time()
                index_poza = self.nn(A, photo, x)
                end_time = time.time()
                
                exec_time = end_time - start_time
                suma_timp += exec_time
                
                eticheta_clasa = index_poza // self.num_images_per_person + 1
                expected_class = i // self.num_test_images + 1  
    
                print(f"Index poza: {index_poza}, Eticheta calculata: {eticheta_clasa}, Eticheta asteptata: {expected_class}")
                
                if eticheta_clasa == expected_class:
                    contor_poze += 1
            
            RR = contor_poze / 80
            AQT = suma_timp / 80
            
            norm_values.append(norm)
            recognition_rates.append(RR * 100)
            avg_times.append(AQT)
            
            print(f"Norma: {norm}")
            print(f"Recunoastere corecta: {RR * 100:.2f}%")
            print(f"Timp mediu de calcul: {AQT:.4f} secunde")
            print()
            
        with open("nn.txt", "w") as f:
            f.write(f"Rata de recunoastere: {recognition_rates}\n Timpul mediu de procesare:{avg_times[-1]}")
    
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(norm_values, recognition_rates, marker='o')
        plt.title('Rate de recunoastere in functie de norma')
        plt.xlabel('Norma')
        plt.ylabel('Rata de recunoastere (%)')
        
        plt.subplot(1, 2, 2)
        plt.plot(norm_values, avg_times, marker='o')
        plt.title('Timp mediu de calcul in functie de norma')
        plt.xlabel('Norma')
        plt.ylabel('Timp mediu de calcul (secunde)')
        
        plt.tight_layout()
        plt.show()

    
    # knn  

    def knn(self, A, poza_selectata, norma, k):
        z = np.zeros(A.shape[1])  
   
          
        if isinstance(poza_selectata, str) and not os.path.exists(poza_selectata):
               print(f"Fisierul {poza_selectata} nu exista. Sarim peste aceasta imagine.")
               return -1
           
           
        if isinstance(poza_selectata, str):
               poza_selectata = cv2.imread(poza_selectata, 0)
               
        poza_selectata = np.reshape(poza_selectata, (self.resolution,))
        
        for i in range(A.shape[1]):
            if norma >= 1 and norma <= 2:
                z[i] = np.linalg.norm(poza_selectata - A[:, i], norma)
            if norma == 3:
                z[i] = np.linalg.norm(poza_selectata - A[:, i], np.inf)
            if norma == 4:
                z[i] = (1 - np.dot(A[:,i], poza_selectata)) / (np.linalg.norm(A[:,i], 2) * np.linalg.norm(poza_selectata, 2))
        
        index_gasit = mode(np.argsort(z)[:k]// 8 + 1)
        cale_imagine = f"D:\\att_faces\\att_faces\\s{index_gasit}\\1.pgm"
        self.afisare_imagine_calculata(cale_imagine)
        return mode(np.argsort(z)[:k] // 8 + 1)
    


    def calculate_statistics_kNN(self, B, A):
        print("knn")
        k_values = []  
        norm_values = []  
        recognition_rates = []  
        avg_times = []  
        
        for k in range(3, 10, 2):  
            print(f"K = {k}")
            contor_poze = 0 
            suma_timp = 0
            recognition_rate_per_k = []
            avg_time_per_k = []
            
            for norm in range(1, 5):
                print(f"\tNorma = {norm}")
                contor_poze = 0 
                suma_timp = 0
                
                for i in range(len(B[0])):
                    photo = np.array(B[:, i])
                    x = norm
                    y = k
                    start_time = time.time()
                    index_poza = self.knn(self.A, photo, x, y)
                    end_time = time.time()
        
                    exec_time = end_time - start_time
                    suma_timp += exec_time
        
                    eticheta_clasa = index_poza
                    if eticheta_clasa == i // self.num_test_images + 1:
                        contor_poze += 1
        
                RR = contor_poze / 80 
                AQT = suma_timp / 80  
        
                recognition_rate_per_k.append(RR * 100)
                avg_time_per_k.append(AQT)
        
                
                print(f"\t\tRata de recunoastere: {RR * 100:.2f}%")
                print(f"\t\tTimp mediu de calcul: {AQT:.4f} secunde")
                
            with open("knn.txt", "w") as f:
                f.write("f\t\tRata de recunoastere: {RR * 100:.2f}%\n \t\tTimp mediu de calcul: {AQT:.4f} secunde")
                    
            
            k_values.append(k)
            recognition_rates.append(np.mean(recognition_rate_per_k))
            avg_times.append(np.mean(avg_time_per_k))
        
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(k_values, recognition_rates, marker='o')
        plt.title('Rate de recunoastere in functie de K')
        plt.xlabel('K')
        plt.ylabel('Rata de recunoastere (%)')
        
       
        plt.subplot(1, 2, 2)
        plt.plot(k_values, avg_times, marker='o')
        plt.title('Timp mediu de calcul in functie de K')
        plt.xlabel('K')
        plt.ylabel('Timp mediu de calcul (secunde)')
        
        plt.tight_layout()
        plt.show()

    # eigenfaces
    
    def eigenPreproc(self, A, k):
        media = np.mean(A, axis=1)
        A_centrat = (A.T - media).T
        # c = np.dot(A_centrat, A_centrat.T)
        l = np.dot(A_centrat.T, A_centrat) # inlocuit cu c
        d, v = la.eig(l)
        v = A_centrat@v # dispare
        idx = np.argsort(d)[::-1][:k]
        HQPB = v[:, idx]
        proiectii = np.dot(A_centrat.T, HQPB)
        return media, HQPB, proiectii
    
    def eigenCalcul(self, caleDS, nrPers, nrPoze, nrPozeTest, rezolutie, k, norma, poza_test_selectata):
        start_preproc = time.time()
        A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)
        media, HQPB, proiectii = self.eigenPreproc(A, k)
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
    
                pozitia = self.nn2(proiectii.T, proiectie_poza_test, norma)
                persoana_identificata = pozitia // nrPoze + 1
    
                if os.path.normpath(calePozaTest) == os.path.normpath(poza_test_selectata):
                    print(f"Poza selectata: {calePozaTest}, Clasa identificata: {persoana_identificata}")
                    prima_poza_identificata = os.path.join(caleDS, f's{persoana_identificata}', '1.pgm')
                    if os.path.exists(prima_poza_identificata):
                        self.afisare_imagine_calculata(prima_poza_identificata)  
    
                if persoana_identificata == i:
                    corecte += 1
    
        timp_proc_total = time.time() - start_proc
    
        rata_recunoastere = (corecte / total) * 100
        
    
    def eigenStatistici(self, caleDS, nrPers, nrPoze, nrPozeTest, rezolutie, k_values, norma_values, poza_test_selectata):
        k_values = [20, 40, 60, 80, 100]
        norma_values = [1, 2, 3, 4]
        preproc_times = []
        recognition_rates = {k: [] for k in k_values}  
        avg_times = {k: [] for k in k_values}  
        
        
        
        for k in k_values:
            start_preproc = time.time()
            A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)
            media, HQPB, proiectii = self.eigenPreproc(A, k)
            timp_preproc_total = time.time() - start_preproc
            preproc_times.append(timp_preproc_total) 
            
            for norma in norma_values:  
                corecte = 0
                total = nrPers * nrPozeTest
                start_proc = time.time()
    
                for i in range(1, nrPers + 1):
                    caleFolder = os.path.join(caleDS, f's{i}')
                    for j in range(nrPoze + 1, nrPoze + 1 + nrPozeTest):
                        calePozaTest = os.path.join(caleFolder, f'{j}.pgm')
                        if not os.path.exists(calePozaTest):
                            continue
    
                        poza_test = cv2.imread(calePozaTest, 0)
                        if poza_test is None:
                            continue
    
                        poza_test = np.reshape(poza_test, (rezolutie,))
                        poza_test = poza_test - media
                        proiectie_poza_test = np.dot(poza_test, HQPB)
    
                        pozitia = self.nn2(proiectii.T, proiectie_poza_test, norma)
                        persoana_identificata = pozitia // nrPoze + 1
    
                        if persoana_identificata == i:
                            corecte += 1
    
                timp_proc_total = time.time() - start_proc
                rata_recunoastere = (corecte / total) * 100
                avg_time = timp_proc_total / (nrPers * nrPozeTest)
    
                recognition_rates[k].append(rata_recunoastere)
                avg_times[k].append(avg_time)
    
        plt.figure(figsize=(18, 6))
    
        plt.subplot(1, 3, 1)
        plt.plot(k_values, preproc_times, marker='o', label='Timp Preprocesare')
        plt.title('K vs. Timpul de Preprocesare')
        plt.xlabel('K (Nivelul de Trunchiere)')
        plt.ylabel('Timp Preprocesare (secunde)')
        plt.legend()
        plt.grid(True)
        plt.xlim(min(k_values), max(k_values))
        plt.ylim(0, max(preproc_times) * 1.2)
    
        plt.subplot(1, 3, 2)
        for k in k_values:
            plt.plot(norma_values, avg_times[k], marker='o', label=f'K={k}')
        plt.title('Norma vs. Timpul de Procesare')
        plt.xlabel('Norma')
        plt.ylabel('Timp Procesare (secunde)')
        plt.legend()
        plt.grid(True)
    
       
        plt.subplot(1, 3, 3)
        for k in k_values:
            plt.plot(norma_values, recognition_rates[k], marker='o', label=f'K={k}')
        plt.title('Norma vs. Ratele de Recunoastere')
        plt.xlabel('Norma')
        plt.ylabel('Rata de Recunoastere (%)')
        plt.legend()
        plt.grid(True)
        plt.xlim(min(norma_values), max(norma_values))
        plt.ylim(80, 100)
    
        plt.tight_layout()
        plt.show()
        
        with open("eigen.txt", "w") as f:
            f.write(f"Rata de recunoastere (%): {recognition_rates[k_values[-1]][-1]},\n Timp de procesare (s): {timp_preproc_total} \n Timp de procesare (s): {timp_proc_total}")
    
        return {
            "Rata de recunoastere (%)": recognition_rates[k_values[-1]][-1],
            "Timp preprocesare (s)": timp_preproc_total,
            "Timp procesare (s)": timp_proc_total
        }


    
    
    # eigenfaces cu reprezentanti de clase
    
    def eigenPreprocClase(self, A, nrPers, nrPozePers, k):
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
    
    def calcul_eigen_clase(self, caleDS, nrPers, nrPoze, nrPozeTest, rezolutie, k, norma, scena, poza_test_selectata):
        start_preproc = time.time()
        A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)
        reprezentari_clase = self.eigenPreprocClase(A, nrPers, nrPoze, k)
        timp_preproc_total = time.time() - start_preproc
    
        corecte = 0
        total = nrPers * nrPozeTest
        start_proc = time.time()
    
        for i in range(1, nrPers + 1): 
            caleFolder = os.path.join(caleDS, f's{i}')
            
            for j in range(nrPoze + 1, nrPoze + nrPozeTest + 1): 
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
    
                for clasa, reprezentare in reprezentari_clase.items():
                    media_clasa = reprezentare["mean_face"]
                    HQPB_clasa = reprezentare["eigenfaces"]
    
                    poza_test_cent = poza_test - media_clasa
                    proiectie_test = np.dot(poza_test_cent, HQPB_clasa)
    
                    proiectii_clasa = reprezentare["proiectii"]
                    diferenta = np.min([self.norma_dif(proiectie_test, p, norma) for p in proiectii_clasa])
                    distante_minime.append((clasa, diferenta))
    
                clasa_identificata = min(distante_minime, key=lambda x: x[1])[0]
                
                if os.path.normpath(calePozaTest) == os.path.normpath(poza_test_selectata):
                    print(f"Poza selectata: {calePozaTest}, Clasa identificata: {clasa_identificata}")
                    prima_poza_identificata = os.path.join(caleDS, f's{clasa_identificata}', '1.pgm')
                    if os.path.exists(prima_poza_identificata):
                        self.afisare_imagine_calculata(prima_poza_identificata)  
    
                if clasa_identificata == i:
                    corecte += 1
    
        timp_proc_total = time.time() - start_proc
    
        rata_recunoastere = (corecte / total) * 100
        
        
    def clase_statistici(self, caleDS, nrPers, nrPoze, nrPozeTest, rezolutie, k_values, norma_values, poza_test_selectata):
        k_values = [20, 40, 60, 80, 100]
        norma_values = [1, 2, 3, 4]
        preproc_times = []
        recognition_rates = {k: [] for k in k_values}
        avg_times = {k: [] for k in k_values}
    
        for k in k_values:
            start_preproc = time.time()
            A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)
            reprezentari_clase = self.eigenPreprocClase(A, nrPers, nrPoze, k)
            timp_preproc_total = time.time() - start_preproc
            preproc_times.append(timp_preproc_total)
    
            for norma in norma_values:
                corecte = 0
                total = nrPers * nrPozeTest
                start_proc = time.time()
    
                for i in range(1, nrPers + 1):
                    caleFolder = os.path.join(caleDS, f's{i}')
                    for j in range(nrPoze + 1, nrPoze + nrPozeTest + 1):
                        calePozaTest = os.path.join(caleFolder, f'{j}.pgm')
                        if not os.path.exists(calePozaTest):
                            continue
    
                        poza_test = cv2.imread(calePozaTest, 0)
                        if poza_test is None:
                            continue
    
                        poza_test = np.reshape(poza_test, (rezolutie,))
                        distante_minime = []
    
                        for clasa, reprezentare in reprezentari_clase.items():
                            media_clasa = reprezentare["mean_face"]
                            HQPB_clasa = reprezentare["eigenfaces"]
    
                            poza_test_cent = poza_test - media_clasa
                            proiectie_test = np.dot(poza_test_cent, HQPB_clasa)
    
                            proiectii_clasa = reprezentare["proiectii"]
                            diferenta = np.min([self.norma_dif(proiectie_test, p, norma) for p in proiectii_clasa])
                            distante_minime.append((clasa, diferenta))
    
                        clasa_identificata = min(distante_minime, key=lambda x: x[1])[0]
    
                        if clasa_identificata == i:
                            corecte += 1
    
                timp_proc_total = time.time() - start_proc
                rata_recunoastere = (corecte / total) * 100
                avg_time = timp_proc_total / total
    
                recognition_rates[k].append(rata_recunoastere)
                avg_times[k].append(avg_time)
    
        plt.figure(figsize=(18, 6))
    
        plt.subplot(1, 3, 1)
        plt.plot(k_values, preproc_times, marker='o', label='Timp Preprocesare')
        plt.title('K vs. Timpul de Preprocesare')
        plt.xlabel('K (Nivel de trunchiere))')
        plt.ylabel('Timp Preprocesare (s)')
        plt.legend()
        plt.grid(True)
    
        plt.subplot(1, 3, 2)
        for k in k_values:
            plt.plot(norma_values, avg_times[k], marker='o', label=f'K={k}')
        plt.title('Norma vs. Timpul de Procesare')
        plt.xlabel('Norma')
        plt.ylabel('Timp Procesare (s)')
        plt.legend()
        plt.grid(True)
    
        plt.subplot(1, 3, 3)
        for k in k_values:
            plt.plot(norma_values, recognition_rates[k], marker='o', label=f'K={k}')
        plt.title('Norma vs. Rata de Recunoastere')
        plt.xlabel('Norma')
        plt.ylabel('Rata de Recunoastere (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        with open("eigen_rc.txt", "w") as f:
            f.write(f"Rata de recunoastere (%): {recognition_rates[k_values[-1]][-1]},\n Timp de procesare (s): {preproc_times[-1]} \n Timp de procesare (s): {avg_times[k_values[-1]][-1]}")
    
        return {
            "Rata de recunoastere (%)": recognition_rates[k_values[-1]][-1],
            "Timp preprocesare (s)": preproc_times[-1],
            "Timp procesare (s)": avg_times[k_values[-1]][-1]
        }



    # lanczos
    
    def lanczosPreproc(self, A, k):
        media = np.mean(A, axis=1)
        A_centrat = A - media[:, None]
        l = np.dot(A_centrat.T, A_centrat)

        HQPB_lanc = self.lanczos(l, k)
        HQPB = np.dot(A_centrat, HQPB_lanc)
        proiectii = np.dot(HQPB.T, A_centrat).T

        return media, HQPB, proiectii
    
    def lanczos(self, A, k):
        n = A.shape[0]
        q = np.zeros((n, k + 2))  # matrice cu k+2 coloane
        beta = 0
        alpha = np.zeros(k)
        
        # vectorul q1 (alegem q1 unitar)
        q[:, 1] = np.random.rand(n)
        q[:, 1] /= la.norm(q[:, 1])

        for i in range(1, k + 1):  # incepe de la 1 pana la k inclusiv
            w = A @ q[:, i] - beta * q[:, i - 1]  # produs matrice-vector si scadere termen anterior
            alpha[i - 1] = np.dot(w, q[:, i])  # calcul alpha
            w = w - alpha[i - 1] * q[:, i]  # substract alpha * q[:, i]
            beta = la.norm(w)  # norma w devine beta

            if beta != 0 and i < k:  # evitam impartirea la 0 sau iesirea din dimensiune
                q[:, i + 1] = w / beta  # normalizeaza si intra in urmatorul vector
            
        # returnam HQPB fara primele 2 coloane (q[:, 2:])
        return q[:, 2:k+2]
    
    def lanczos_calcul(self, caleDS, nrPers, nrPoze, nrPozeTest, rezolutie, k, norma, poza_test_selectata):
        start_preproc = time.time()
        A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)
        media, HQPB, proiectii = self.lanczosPreproc(A, k)
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
    
                pozitia = self.nn2(proiectii.T, proiectie_poza_test, norma)
                persoana_identificata = pozitia // nrPoze + 1
    
                if os.path.normpath(calePozaTest) == os.path.normpath(poza_test_selectata):
                    print(f"Poza selectata: {calePozaTest}, Clasa identificata: {persoana_identificata}")
                    prima_poza_identificata = os.path.join(caleDS, f's{persoana_identificata}', '1.pgm')
                    if os.path.exists(prima_poza_identificata):
                        self.afisare_imagine_calculata(prima_poza_identificata)  
    
                if persoana_identificata == i:
                    corecte += 1
    
        timp_proc_total = time.time() - start_proc
    
        rata_recunoastere = (corecte / total) * 100
    
    
    def lanczos_statistici(self, caleDS, nrPers, nrPoze, nrPozeTest, rezolutie, k_values, norma_values, poza_test_selectata):
        recognition_rates = {}
        avg_times = {}  
        preproc_times = [] 
        k_values = [20, 40, 60, 80, 100]
        norma_values = [1, 2, 3, 4]
        
        for k in k_values:
            recognition_rates[k] = []
            avg_times[k] = []
            
           
            start_preproc = time.time()
            A = creareMatrAntr(caleDS, nrPers, nrPoze, rezolutie)
            media, HQPB, proiectii = self.lanczosPreproc(A, k)
            timp_preproc_total = time.time() - start_preproc
            preproc_times.append(timp_preproc_total)
    
            
            for norma in norma_values:
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
                        
                        pozitia = self.nn2(proiectii.T, proiectie_poza_test, norma)
                        persoana_identificata = pozitia // nrPoze + 1
                        
                        
                        if os.path.normpath(calePozaTest) == os.path.normpath(poza_test_selectata):
                            print(f"Poza selectata: {calePozaTest}, Clasa identificata: {persoana_identificata}")
                            prima_poza_identificata = os.path.join(caleDS, f's{persoana_identificata}', '1.pgm')
                            if os.path.exists(prima_poza_identificata):
                                self.afisare_imagine_calculata(prima_poza_identificata)
                        
                        if persoana_identificata == i:
                            corecte += 1
                
                timp_proc_total = time.time() - start_proc
                rata_recunoastere = (corecte / total) * 100
                avg_time = timp_proc_total / total
                
                recognition_rates[k].append(rata_recunoastere)
                avg_times[k].append(avg_time)
                with open("lanczos.txt", "w") as f:
                    f.write(f"K = {k}, Norma = {norma}, Rata recunoastere: {rata_recunoastere:.2f}%, Timp procesare: {avg_time:.4f} secunde")
                print(f"K = {k}, Norma = {norma}, Rata recunoastere: {rata_recunoastere:.2f}%, Timp procesare: {avg_time:.4f} secunde")
        
        
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(k_values, preproc_times, marker='o', label='Timp Preprocesare')
        plt.title('K vs. Timpul de Preprocesare')
        plt.xlabel('K (Nivelul de Trunchiere)')
        plt.ylabel('Timp Preprocesare (secunde)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        for k in k_values:
            plt.plot(norma_values, avg_times[k], marker='o', label=f'K={k}')
        plt.title('Norma vs. Timpul de Procesare')
        plt.xlabel('Norma')
        plt.ylabel('Timp Procesare (secunde)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        for k in k_values:
            plt.plot(norma_values, recognition_rates[k], marker='o', label=f'K={k}')
        plt.title('Norma vs. Rata de Recunoastere')
        plt.xlabel('Norma')
        plt.ylabel('Rata de Recunoastere (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return {
            "Rata de recunoastere (%)": recognition_rates[k_values[-1]][-1],
            "Timp preprocesare (s)": preproc_times[-1],
            "Timp procesare (s)": avg_times[k_values[-1]][-1] * total
        }

    
    def nn2(self, A, pozaCautata, norma):
        z = np.zeros(A.shape[1])
        for i in range(A.shape[1]):
            z[i] = self.norma_dif(A[:, i], pozaCautata, norma)
        pozitia = np.argmin(z)
        return pozitia


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    app.exec()
