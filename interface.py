# -*- coding: utf-8 -*-
"""
=============================================================================
interface.py — Version ENSEMBLE (MLP + SVM)
Processus : Rembg -> 128px -> PCA -> Fusion des modèles
=============================================================================
"""

import os
import sys
import traceback
import numpy as np
import cv2
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rembg import remove

MODEL_FILE = "model_aircraft.pkl"

def _get_from_session(name):
    main_mod = sys.modules.get('__main__')
    if main_mod is not None and hasattr(main_mod, name):
        return getattr(main_mod, name)
    return None

class AircraftClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IPSA - Expert Aircraft Classifier (Ensemble Mode)")
        self.root.geometry("1000x900")
        self.root.configure(bg="#f8f9fa")

        self.data = None
        self.mlp = None
        self.svm = None

        self._build_ui()
        self._load_models()

    def _load_models(self):
        if os.path.exists(MODEL_FILE):
            try:
                content = joblib.load(MODEL_FILE)
                self.data = content.get('data')
                self.mlp = content.get('mlp')
                self.svm = content.get('svm')
                print(f"✔ Modèles chargés. Stratégie détectée : {self.data.get('best_strategy', 'weighted')}")
            except Exception as e:
                print(f"Erreur chargement : {e}")

    def _build_ui(self):
        tk.Label(self.root, text="Analyse Multi-Modèles (Ensemble)", 
                 font=("Helvetica", 20, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(pady=15)

        # Zone Image
        self.img_frame = tk.Frame(self.root, width=500, height=350, bg="white", relief="groove", borderwidth=2)
        self.img_frame.pack(pady=10)
        self.img_frame.pack_propagate(False)
        self.img_label = tk.Label(self.img_frame, text="Glissez une image ici", bg="white", fg="#95a5a6")
        self.img_label.pack(expand=True, fill=tk.BOTH)

        # Verdict Final (Ensemble)
        self.verdict_frame = tk.Frame(self.root, bg="#ebf5fb", pady=10)
        self.verdict_frame.pack(fill=tk.X, padx=50, pady=10)
        self.res_title = tk.Label(self.verdict_frame, text="VERDICT ENSEMBLE", font=("Arial", 10, "bold"), bg="#ebf5fb", fg="#2980b9")
        self.res_title.pack()
        self.res_main = tk.Label(self.verdict_frame, text="En attente...", font=("Helvetica", 22, "bold"), bg="#ebf5fb", fg="#2c3e50")
        self.res_main.pack()

        # Boutons
        btn_f = tk.Frame(self.root, bg="#f8f9fa")
        btn_f.pack(pady=15)
        tk.Button(btn_f, text="📂 Ouvrir Image", command=self.load_image, width=15, bg="#34495e", fg="white").grid(row=0, column=0, padx=10)
        self.go_btn = tk.Button(btn_f, text="🚀 ANALYSER", command=self.predict, state=tk.DISABLED, width=15, bg="#27ae60", fg="white", font=("Arial", 10, "bold"))
        self.go_btn.grid(row=0, column=1, padx=10)

        # Détails MLP/SVM
        details_f = tk.Frame(self.root, bg="#f8f9fa")
        details_f.pack(fill=tk.BOTH, expand=True, padx=40)
        
        self.mlp_ui = self._create_box(details_f, " Confiance MLP ")
        self.svm_ui = self._create_box(details_f, " Confiance SVM ")

    def _create_box(self, parent, title):
        box = tk.LabelFrame(parent, text=title, font=("Arial", 9, "bold"))
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        lbl = tk.Label(box, text="", justify=tk.LEFT, font=("Courier", 10), bg="white", anchor="nw", padx=10, pady=10)
        lbl.pack(fill=tk.BOTH, expand=True)
        return lbl

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image_path = path
            img = Image.open(path)
            img.thumbnail((480, 330))
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img, text="")
            self.cv_img = cv2.imread(path)
            self.go_btn.config(state=tk.NORMAL)

    def predict(self):
        try:
            # 1. Variables
            data = self.data or _get_from_session('data')
            mlp = self.mlp or _get_from_session('mlp')
            svm = self.svm or _get_from_session('svm')
            
            if not data: return messagebox.showerror("Erreur", "Modèle non chargé.")

            # Gestion des classes (évite le KeyError 'le')
            if 'le' in data: classes = data['le'].classes_
            elif 'classes' in data: classes = data['classes']
            else: classes = _get_from_session('classes')

            self.res_main.config(text="Analyse en cours...", fg="orange")
            self.root.update()

            # 2. Prétraitement (Crop -> Rembg -> Resize 128)
            img = self.cv_img.copy()
            # Crop
            h_crop = data.get('CROP_BOTTOM', 20)
            if img.shape[0] > h_crop: img = img[:-h_crop, :]
            
            # REMBG & RESIZE 128
            img_rgba = remove(img)
            img_rgb = np.array(Image.fromarray(img_rgba).convert("RGB"))
            size = data.get('IMG_SIZE', 128)
            img_res = cv2.resize(img_rgb, (size, size))

            # PCA
            x = img_res.flatten().reshape(1, -1).astype(np.float32) / 255.0
            x_pca = data['pca'].transform((x - data['mu']) / data['sigma'])

            # 3. Prédictions individuelles
            p_mlp = mlp.predict_proba(x_pca)[0]
            p_svm = svm.predict_proba(x_pca)[0]

            # 4. LOGIQUE ENSEMBLE (Le coeur de ton choix)
            strat = data.get('best_strategy', 'weighted')
            w_mlp = data.get('best_wmlp', 0.5)
            w_svm = data.get('best_wsvm', 0.5)

            if strat == 'weighted':
                p_final = (w_mlp * p_mlp) + (w_svm * p_svm)
            elif strat == 'max':
                p_final = np.maximum(p_mlp, p_svm)
            else: # fallback simple moyenne
                p_final = (p_mlp + p_svm) / 2

            # 5. Affichage
            idx_final = np.argmax(p_final)
            self.res_main.config(text=f"{classes[idx_final]}", fg="#27ae60")

            # Tops 5 pour détails
            def get_top5(p): return sorted(zip(classes, p*100), key=lambda x:x[1], reverse=True)[:5]
            
            self.mlp_ui.config(text="\n".join([f"{c[:20]:<20} {v:>5.1f}%" for c,v in get_top5(p_mlp)]))
            self.svm_ui.config(text="\n".join([f"{c[:20]:<20} {v:>5.1f}%" for c,v in get_top5(p_svm)]))

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Erreur", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = AircraftClassifierApp(root)
    root.mainloop()