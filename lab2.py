import os
import numpy as np
from numpy import linalg as la
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from statistics import mode
import time
import pandas as pd
def generate_data_matrix(base_folder, image_width=112, image_height=92, num_people=40, num_images_per_person=8):
    data_matrix = np.zeros((image_width * image_height, num_people * num_images_per_person))
    
    column_index = 0
    for person_index in range(num_people):
        person_folder = os.path.join(base_folder, f's{person_index + 1}')
        image_files = sorted(os.listdir(person_folder))[:num_images_per_person]
        for image_file in image_files:
            image_path = os.path.join(person_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (image_width, image_height))
            image_vector = image.reshape(-1)
            data_matrix[:, column_index] = image_vector
            column_index += 1
    
    return data_matrix

def nn_algorithm(A, poza_cautata, norma='Euclidean', afiseaza=False):
    z = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        if norma == 'Manhattan':  
            z[i] = la.norm(A[:, i] - poza_cautata, ord=1)
        elif norma == 'Euclidean':  
            z[i] = la.norm(A[:, i] - poza_cautata, ord=2)
        elif norma == 'Infinit':  
            z[i] = la.norm(A[:, i] - poza_cautata, ord=np.inf)
        elif norma == 'Cosine':  
            z[i] = 1 - np.dot(A[:, i], poza_cautata) / (la.norm(A[:, i]) * la.norm(poza_cautata))
    
    pozitia_minima = np.argmin(z)
    pozitii_minime = np.where(z == np.min(z))[0]

    if afiseaza:
        plt.imshow(A[:, pozitia_minima].reshape(112, 92), cmap='gray')
        plt.title(f"Poza cea mai apropiată - Index: {pozitia_minima}")
        plt.show()

    return pozitia_minima, pozitii_minime

def knn_algorithm(A, poza_cautata, norma='Euclidean', k=3, afiseaza=False):
    z = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        if norma == 'Manhattan':
            z[i] = la.norm(A[:, i] - poza_cautata, ord=1)
        elif norma == 'Euclidean':
            z[i] = la.norm(A[:, i] - poza_cautata, ord=2)
        elif norma == 'Infinit':
            z[i] = la.norm(A[:, i] - poza_cautata, ord=np.inf)
        elif norma == 'Cosine':
            z[i] = 1 - np.dot(A[:, i], poza_cautata) / (la.norm(A[:, i]) * la.norm(poza_cautata))
    
    pozitii = np.argsort(z)[:k]
    persoane = [pozitie // 8 for pozitie in pozitii]  
    persoana_cautata = mode(persoane)

    for pozitie in pozitii:
        if pozitie // 8 == persoana_cautata:
            pozitia_minima = pozitie
            break
    
    if afiseaza:
        plt.imshow(A[:, pozitia_minima].reshape(112, 92), cmap='gray')
        plt.title(f"Poza cea mai apropiată - Index: {pozitia_minima}")
        plt.show()
    
    return pozitia_minima, pozitii

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        test_image = cv2.resize(test_image, (image_width, image_height))
        poza_cautata = test_image.reshape(-1)
        
        img = Image.fromarray(test_image)  
        img_tk = ImageTk.PhotoImage(image=img)  
        
        image_label.config(image=img_tk)
        image_label.image = img_tk  
        
        return poza_cautata, file_path
    else:
        messagebox.showwarning("Atenție", "Nu a fost selectată nicio imagine.")
        return None, None
def calculate_rr_and_aqt_nn(A, norma='Euclidean'):
    t0 = time.time()
    suma_timp = 0
    contor_corecte = 0  
    
    for i in range(40):

        for img_num in [9, 10]:
            person_folder = os.path.join(base_folder, f's{i + 1}')
            image_path = os.path.join(person_folder, f'{img_num}.pgm')

            test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            test_image = cv2.resize(test_image, (image_width, image_height))
            poza_cautata = test_image.reshape(-1)
            
            t1 = time.time()
            i0, _ = nn_algorithm(A, poza_cautata, norma=norma, afiseaza=False)
            durata = time.time() - t1

            p0 = i0 // 8 + 1  

            if p0 == (i + 1):
                contor_corecte += 1
            
            suma_timp += durata
    
    rr = contor_corecte / 80.0  
    aqt = suma_timp / 80.0
    
    t_total = time.time() - t0
    return rr, aqt, t_total
def calculate_rr_and_aqt(A, norma='Euclidean', algorithm='NN', k=3):
    t0 = time.time()  
    suma_timp = 0  
    contor_corecte = 0  
    
    for i in range(40):

        for img_num in [9, 10]:
            person_folder = os.path.join(base_folder, f's{i + 1}')
            image_path = os.path.join(person_folder, f'{img_num}.pgm')
            

            test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            test_image = cv2.resize(test_image, (image_width, image_height))
            poza_cautata = test_image.reshape(-1)
            

            t1 = time.time()

            if algorithm == 'NN':
                i0, _ = nn_algorithm(A, poza_cautata, norma=norma, afiseaza=False)
            elif algorithm == 'kNN':
                i0, _ = knn_algorithm(A, poza_cautata, norma=norma, k=k, afiseaza=False)

            durata = time.time() - t1

            p0 = i0 // num_images_per_person + 1 

            if p0 == (i + 1):
                contor_corecte += 1

            suma_timp += durata
    

    rr = contor_corecte / 80.0  
    aqt = suma_timp / 80.0  
    

    t_total = time.time() - t0
    
    return rr, aqt, t_total

def plot_rr_aqt():
    norme = ['Euclidean', 'Manhattan', 'Infinit', 'Cosine']
    rr_values = []
    aqt_values = []
    
    algorithm = algorithm_var.get() 
    k_value = int(k_entry.get()) if algorithm == 'kNN' else 1  
    

    for norma in norme:
        rr_nn, aqt_nn, _ = calculate_rr_and_aqt(data_matrix, norma=norma, algorithm=algorithm, k=k_value)
        rr_values.append(rr_nn)
        aqt_values.append(aqt_nn)

    rr_df = pd.DataFrame({
        'Norma': norme,
        'Rata de Recunoaștere (RR)': rr_values
    })
    

    aqt_df = pd.DataFrame({
        'Norma': norme,
        'Average Query Time (AQT)': aqt_values
    })
    

    rr_df.to_csv(f'rr_values_{algorithm}.csv', index=False)
    aqt_df.to_csv(f'aqt_values_{algorithm}.csv', index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(norme, rr_values, color='red', marker='o', linestyle='--', label='RR')
    plt.xlabel('Norme')
    plt.ylabel('Rata de Recunoaștere (RR)')
    plt.title(f'Rata de Recunoaștere pentru algoritmul {algorithm}')
    
    for i, txt in enumerate(rr_values):
        plt.annotate(f'{txt:.4f}', (norme[i], rr_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')  # Afișează valoarea exactă
    
    plt.grid(True)
    plt.show()
    

    plt.figure(figsize=(8, 6))
    plt.plot(norme, aqt_values, color='blue', marker='o', linestyle='-', label='AQT')
    plt.xlabel('Norme')
    plt.ylabel('Average Query Time (AQT)')
    plt.title(f'Average Query Time pentru algoritmul {algorithm}')
    
    for i, txt in enumerate(aqt_values):
        plt.annotate(f'{txt:.4f}', (norme[i], aqt_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')  # Afișează valoarea exactă
    
    plt.grid(True)
    plt.show()
def run_nn():
    poza_cautata, img_path = load_image()
    if poza_cautata is None:
        return
    
    norma_aleasa = norma_var.get()
    algorithm = algorithm_var.get()
    
    if algorithm == 'NN':
        pozitia_simpla, pozitii_avansate = nn_algorithm(data_matrix, poza_cautata, norma=norma_aleasa, afiseaza=True)
        result_label.config(text=f"Indexul celei mai apropiate poze: {pozitia_simpla}\nToate pozițiile cu distanță minimă: {pozitii_avansate}")
    elif algorithm == 'kNN':
        k_value = int(k_entry.get())
        pozitia_simpla, pozitii_avansate = knn_algorithm(data_matrix, poza_cautata, norma=norma_aleasa, k=k_value, afiseaza=True)
        result_label.config(text=f"Indexul celei mai apropiate poze: {pozitia_simpla}\nToate pozițiile celor mai apropiați {k_value} vecini: {pozitii_avansate}")
    
    messagebox.showinfo("Rezultat", f"Imaginea a fost încărcată din {img_path}\nNorma folosită: {norma_aleasa}\nAlgoritm: {algorithm}\nIndex: {pozitia_simpla}")

if __name__ == "__main__":
    image_width = 92
    image_height = 112
    num_people = 40
    num_images_per_person = 8
    base_folder = 'D:\\att_faces'
    data_matrix = generate_data_matrix(base_folder, image_width, image_height, num_people, num_images_per_person)
    
    root = tk.Tk()
    root.title("Algoritmul NN/kNN")
    
    tk.Label(root, text="Alege norma:").pack(pady=5)
    
    norma_var = tk.StringVar(value='euclidean')
    tk.Radiobutton(root, text="Euclidean", variable=norma_var, value='Euclidean').pack(anchor=tk.W)
    tk.Radiobutton(root, text="Manhattan", variable=norma_var, value='Manhattan').pack(anchor=tk.W)
    tk.Radiobutton(root, text="Infinit", variable=norma_var, value='Infinit').pack(anchor=tk.W)
    tk.Radiobutton(root, text="Cosine", variable=norma_var, value='Cosine').pack(anchor=tk.W)
    
    tk.Label(root, text="Alege algoritmul:").pack(pady=5)
    
    algorithm_var = tk.StringVar(value='NN')
    tk.Radiobutton(root, text="NN", variable=algorithm_var, value='NN').pack(anchor=tk.W)
    tk.Radiobutton(root, text="kNN", variable=algorithm_var, value='kNN').pack(anchor=tk.W)
    
    tk.Label(root, text="Introdu valoarea lui k pentru kNN:").pack(pady=5)
    k_entry = tk.Entry(root)
    k_entry.pack(pady=5)
    
    tk.Button(root, text="Încarcă imagine și rulează", command=run_nn).pack(pady=10)
    
    image_label = tk.Label(root)  
    image_label.pack(pady=10)
    
    result_label = tk.Label(root, text="")
    result_label.pack(pady=10)
    tk.Button(root, text="Afișează grafice RR și AQT", command=plot_rr_aqt).pack(pady=10)
   

    
root.mainloop()