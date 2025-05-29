import os
import numpy as np
from numpy import linalg as la
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk 
from PIL import Image, ImageTk  
import time
import threading

def generate_data_matrix(base_folder, image_width=112, image_height=92, num_people=40, num_images_per_person=8):
    data_matrix = np.zeros((image_width * image_height, num_people * num_images_per_person))
    labels = np.zeros(num_people * num_images_per_person, dtype=int)
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
            labels[column_index] = person_index
            column_index += 1
    
    return data_matrix, labels

def knn_algorithm(A, poza_cautata, k=1, norma='1', afiseaza=False):
    z = np.zeros(A.shape[1])
    
    for i in range(A.shape[1]):
        if norma == '1':  
            z[i] = la.norm(A[:, i] - poza_cautata, ord=1)
        elif norma == '2':  
            z[i] = la.norm(A[:, i] - poza_cautata, ord=2)
        elif norma == 'inf':  
            z[i] = la.norm(A[:, i] - poza_cautata, ord=np.inf)
        elif norma == 'cosine':  
            z[i] = 1 - np.dot(A[:, i], poza_cautata) / (la.norm(A[:, i]) * la.norm(poza_cautata))
    
    pozitiile_minime = np.argsort(z)[:k]  
    valori_minime = z[pozitiile_minime]
    
    return pozitiile_minime, valori_minime

def evaluate_accuracy(data_matrix, labels, k=1, norma='1'):
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(data_matrix.shape[1]):
        poza_cautata = data_matrix[:, i]
        true_label = labels[i]
        
        # Elimină poza curentă pentru evaluare (leave-one-out)
        A = np.delete(data_matrix, i, axis=1)
        labels_without_current = np.delete(labels, i)
        
        pozitiile_minime, _ = knn_algorithm(A, poza_cautata, k=k, norma=norma)
        predicted_label = labels_without_current[pozitiile_minime[0]]
        
        if predicted_label == true_label:
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

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
        messagebox.showwarning("Atentie", "Nu a fost selectata nicio imagine.")
        return None, None

def run_knn():
    poza_cautata, img_path = load_image()
    if poza_cautata is None:
        return
    
    norma_aleasa = norma_combobox.get()
    k_val = int(k_entry.get())  
    pozitiile_minime, valori_minime = knn_algorithm(data_matrix, poza_cautata, k=k_val, norma=norma_aleasa)
    
    result_text = f"Primele {k_val} pozitii:\n"
    for i, poz in enumerate(pozitiile_minime):
        result_text += f"Index: {poz}, Distanta: {valori_minime[i]:.4f}\n"
    
    result_label.config(text=result_text)
    
    # Pornim un thread pentru evaluarea acurateței
    threading.Thread(target=evaluate_and_display_accuracy, args=(k_val,)).start()
    
    messagebox.showinfo("Rezultat", f"Imaginea a fost incarcata din {img_path}\nNorma folosită: {norma_aleasa}")

def evaluate_and_display_accuracy(k_val):
    # Evaluarea acurateței pentru fiecare normă
    accuracies = {}
    for norma in ['1', '2', 'inf', 'cosine']:
        accuracy = evaluate_accuracy(data_matrix, labels, k=k_val, norma=norma)
        accuracies[norma] = accuracy
    
    stats_text = "Statistici de acuratețe:\n"
    for norma, acc in accuracies.items():
        stats_text += f"Norma {norma}: Acuratețe = {acc:.2%}\n"
    
    stats_label.config(text=stats_text)

if __name__ == "__main__":
    image_width = 112
    image_height = 92
    num_people = 40
    num_images_per_person = 8
    base_folder = 'D:/att_faces'
    data_matrix, labels = generate_data_matrix(base_folder, image_width, image_height, num_people, num_images_per_person)
    
    root = tk.Tk()
    root.title("Algoritmul k-NN")
    
    tk.Label(root, text="Alege norma:").pack(pady=5)
    
    norme_disponibile = ['1', '2', 'inf', 'cosine']
    norma_combobox = ttk.Combobox(root, values=norme_disponibile, state="readonly")
    norma_combobox.set('1')  
    norma_combobox.pack(pady=5)
    
    tk.Label(root, text="Introdu valoarea pentru k:").pack(pady=5)
    k_entry = tk.Entry(root)
    k_entry.pack(pady=5)
    k_entry.insert(0, "3") 
    
    tk.Button(root, text="Incarca imaginea si ruleaza k-NN", command=run_knn).pack(pady=10)
    
    image_label = tk.Label(root)  
    image_label.pack(pady=10)
    
    result_label = tk.Label(root, text="")
    result_label.pack(pady=10)
    
    stats_label = tk.Label(root, text="")
    stats_label.pack(pady=10)
    
root.mainloop()
