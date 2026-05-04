# -*- coding: utf-8 -*-
"""
=============================================================================
interface.py — Interface graphique de prédiction d'avion
Ma422 - Introduction to Machine Learning — Mini Project (IPSA)
=============================================================================
Lance une fenêtre Tkinter qui permet de :
  1. Charger une image d'avion depuis le PC
  2. Afficher l'image chargée
  3. Lancer la prédiction (SVM + MLP) avec les modèles déjà entraînés
     dans Projet.py
  4. Afficher la famille la plus probable + son pourcentage de confiance

PRÉREQUIS : exécuter Projet.py au préalable dans la même session Python
(par exemple dans Spyder) pour que svm_sklearn / mlp_sklearn et la PCA
soient déjà entraînés en mémoire.
=============================================================================
"""

import os
import sys
import traceback
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Récupération depuis la session Python en cours (Projet.py exécuté avant)
# ---------------------------------------------------------------------------
def _get_from_session(name):
    """Récupère une variable du namespace __main__ (la session Python courante)."""
    main_mod = sys.modules.get('__main__')
    if main_mod is not None and hasattr(main_mod, name):
        return getattr(main_mod, name)
    return None


# 1) Récupérer les modèles entraînés depuis Projet.py
svm_model = _get_from_session('svm_sklearn')
mlp_model = _get_from_session('mlp_sklearn')

if svm_model is None or mlp_model is None:
    import functions_ML_sklearn
    if svm_model is None:
        svm_model = functions_ML_sklearn.svm_sklearn
    if mlp_model is None:
        mlp_model = functions_ML_sklearn.mlp_sklearn

# 2) Récupérer le dictionnaire `pp` produit par run_preprocessing dans Projet.py
pp = _get_from_session('pp')
if pp is None:
    raise RuntimeError(
        "Le dictionnaire `pp` (résultat de run_preprocessing) est introuvable "
        "dans la session Python. Lance d'abord Projet.py dans la même session."
    )

# 3) Extraire toutes les infos nécessaires au prétraitement d'une nouvelle image
classes        = pp['classes']
IMG_SIZE       = pp['IMG_SIZE']
NUM_PCS        = pp['NUM_PCS']
COLOR_SIZE     = pp['COLOR_SIZE']
images_couleur = 1 if COLOR_SIZE == 3 else 0

# Objets PCA et moyenne (selon ce qui est exporté par run_preprocessing)
pca = pp.get('pca', None)
mu  = pp.get('mu', None)
scaler = pp.get('scaler', None)  # si jamais un StandardScaler est utilisé

if pca is None:
    # Fallback : essayer de récupérer depuis le module preprocess
    try:
        import preprocess as _pp_mod
        pca = getattr(_pp_mod, 'pca', None)
        if mu is None:
            mu = getattr(_pp_mod, 'mu', None)
    except Exception:
        pass

if pca is None:
    raise RuntimeError(
        "Impossible de récupérer l'objet PCA entraîné. "
        "Vérifie que run_preprocessing expose bien `pca` dans son dictionnaire de sortie."
    )


# ---------------------------------------------------------------------------
# Vérification que les modèles sont entraînés
# ---------------------------------------------------------------------------
def _is_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False

print(f"[interface.py] SVM entraîné : {_is_fitted(svm_model)}")
print(f"[interface.py] MLP entraîné : {_is_fitted(mlp_model)}")
print(f"[interface.py] Mode image   : {'RGB' if images_couleur else 'N&B'} "
      f"({COLOR_SIZE}x{IMG_SIZE}x{IMG_SIZE})")
print(f"[interface.py] PCA          : {NUM_PCS} composantes")

if not _is_fitted(svm_model) or not _is_fitted(mlp_model):
    print("[interface.py] ⚠  Au moins un modèle n'est pas entraîné !")
    print("[interface.py] ⚠  Lance d'abord Projet.py dans la même session.")


# ---------------------------------------------------------------------------
# Prétraitement d'une nouvelle image (identique à run_preprocessing)
# ---------------------------------------------------------------------------
def preprocess_new_image(image_path):
    """
    Charge et prétraite une image de la même façon que le pipeline
    d'entraînement (run_preprocessing dans preprocess.py).

    Étapes :
      1. Lecture (RGB ou N&B selon le mode)
      2. Suppression du bandeau crédit photo (20 px en bas)
      3. Redimensionnement à IMG_SIZE x IMG_SIZE
      4. Normalisation [0, 1]
      5. Aplatissement (order='F' = column-major façon MATLAB)
      6. Centrage par la moyenne du jeu d'entraînement
      7. Projection PCA
    """
    # 1) Lecture
    if images_couleur == 1:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Impossible de charger l'image : {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de charger l'image : {image_path}")

    # 2) Suppression des 20 pixels du bas (bandeau crédit photo)
    if img.shape[0] > 20:
        img = img[:-20, :] if img.ndim == 2 else img[:-20, :, :]

    # 3) Redimensionnement
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 4) Normalisation [0, 1]
    img_norm = img_resized.astype(np.float64) / 255.0

    # 5) Aplatissement (order='F' pour reproduire le column-major MATLAB)
    x_flat = img_norm.flatten(order='F').reshape(1, -1)

    # 6) Centrage (si mu fourni)
    if mu is not None:
        x_centered = x_flat - mu
    else:
        x_centered = x_flat

    # 6 bis) Si un scaler a été utilisé pendant l'entraînement
    if scaler is not None:
        x_centered = scaler.transform(x_centered)

    # 7) Projection PCA
    x_pca = pca.transform(x_centered)
    return x_pca


# ---------------------------------------------------------------------------
# Helper : récupère le top-5 des prédictions avec leurs pourcentages
# ---------------------------------------------------------------------------
def predict_top5(model, x_pca):
    """Retourne une liste de (nom_classe, pourcentage) pour les 5 meilleures prédictions."""
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x_pca)[0]
            top5_idx = np.argsort(probs)[::-1][:5]
            return [(classes[int(i)], float(probs[i]) * 100.0) for i in top5_idx]
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(x_pca)
            scores = np.atleast_2d(scores)[0]
            scores_shift = scores - np.max(scores)
            exp_s = np.exp(scores_shift)
            probs = exp_s / np.sum(exp_s)
            top5_idx = np.argsort(probs)[::-1][:5]
            return [(classes[int(i)], float(probs[i]) * 100.0) for i in top5_idx]
        except Exception:
            pass

    pred = int(model.predict(x_pca)[0])
    return [(classes[pred], 100.0)]


# ---------------------------------------------------------------------------
# Application Tkinter
# ---------------------------------------------------------------------------
class AircraftClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance de famille d'avion — Ma422 ML Project")
        self.root.geometry("850x800")
        self.root.configure(bg="#f0f0f0")

        self.image_path = None
        self.tk_image   = None

        self._build_ui()

    def _build_ui(self):
        tk.Label(
            self.root,
            text="Classification d'avion par Machine Learning",
            font=("Helvetica", 16, "bold"),
            bg="#f0f0f0", fg="#1a3d6d",
        ).pack(pady=10)

        tk.Label(
            self.root,
            text=f"Modèles : SVM + MLP  |  PCA : {NUM_PCS} composantes  |  "
                 f"Image : {COLOR_SIZE}×{IMG_SIZE}×{IMG_SIZE}",
            font=("Helvetica", 9, "italic"),
            bg="#f0f0f0", fg="#555555",
        ).pack()

        # Zone image
        self.image_frame = tk.Frame(
            self.root, width=400, height=400, bg="white",
            highlightbackground="#1a3d6d", highlightthickness=2,
        )
        self.image_frame.pack(pady=15)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            text="(aucune image chargée)",
            bg="white", fg="#888888", font=("Helvetica", 11),
        )
        self.image_label.pack(expand=True)

        # Boutons
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=5)

        tk.Button(
            btn_frame, text="📁  Charger une image",
            command=self.load_image,
            font=("Helvetica", 11, "bold"),
            bg="#1a3d6d", fg="white",
            padx=15, pady=6, cursor="hand2",
        ).pack(side=tk.LEFT, padx=5)

        self.predict_btn = tk.Button(
            btn_frame, text="🔍  Prédire la famille",
            command=self.predict,
            font=("Helvetica", 11, "bold"),
            bg="#2e8b57", fg="white",
            padx=15, pady=6, cursor="hand2",
            state=tk.DISABLED,
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        # Résultat principal
        self.result_label = tk.Label(
            self.root, text="",
            font=("Helvetica", 14, "bold"),
            bg="#f0f0f0", fg="#1a3d6d",
            justify=tk.CENTER, wraplength=620,
        )
        self.result_label.pack(pady=15)

        # Top-5 en deux colonnes
        self.top5_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.top5_frame.pack(pady=10, padx=30, fill=tk.BOTH, expand=True)

        # SVM
        self.svm_frame = tk.Frame(self.top5_frame, bg="#f9f9f9",
                                  highlightbackground="#1a3d6d", highlightthickness=1)
        self.svm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(self.svm_frame, text="🔹 Top-5 SVM", font=("Helvetica", 11, "bold"),
                 bg="#f9f9f9", fg="#1a3d6d").pack(pady=5)
        self.svm_text = tk.Label(
            self.svm_frame, text="", justify=tk.LEFT,
            font=("Courier", 10), bg="#f9f9f9", fg="#333333", anchor="w"
        )
        self.svm_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # MLP
        self.mlp_frame = tk.Frame(self.top5_frame, bg="#f9f9f9",
                                  highlightbackground="#2e8b57", highlightthickness=1)
        self.mlp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(self.mlp_frame, text="🔹 Top-5 MLP", font=("Helvetica", 11, "bold"),
                 bg="#f9f9f9", fg="#2e8b57").pack(pady=5)
        self.mlp_text = tk.Label(
            self.mlp_frame, text="", justify=tk.LEFT,
            font=("Courier", 10), bg="#f9f9f9", fg="#333333", anchor="w"
        )
        self.mlp_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # ------------------------ Callbacks ------------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image d'avion",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if not path:
            return

        self.image_path = path
        try:
            pil_img = Image.open(path)
            pil_img.thumbnail((380, 380))
            self.tk_image = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.tk_image, text="")
            self.predict_btn.config(state=tk.NORMAL)
            self.result_label.config(
                text=f"Image chargée : {os.path.basename(path)}",
                fg="#333333", font=("Helvetica", 11),
            )
            self.svm_text.config(text="")
            self.mlp_text.config(text="")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'image :\n{e}")

    def predict(self):
        if not self.image_path:
            return

        try:
            x_pca = preprocess_new_image(self.image_path)

            svm_top5 = predict_top5(svm_model, x_pca)
            mlp_top5 = predict_top5(mlp_model, x_pca)

            svm_best_class, svm_best_conf = svm_top5[0]
            mlp_best_class, mlp_best_conf = mlp_top5[0]

            if mlp_best_conf >= svm_best_conf:
                best_class, best_conf, best_model = mlp_best_class, mlp_best_conf, "MLP"
            else:
                best_class, best_conf, best_model = svm_best_class, svm_best_conf, "SVM"

            verdict = (
                f"✈  Famille prédite : {best_class}\n"
                f"Confiance : {best_conf:.1f}%   (modèle : {best_model})"
            )
            self.result_label.config(text=verdict, fg="#2e8b57",
                                     font=("Helvetica", 14, "bold"))

            svm_lines = []
            for i, (cls, conf) in enumerate(svm_top5, 1):
                cls_short = cls[:28] if len(cls) <= 28 else cls[:25] + "..."
                svm_lines.append(f"{i}. {cls_short:<30s} {conf:5.1f}%")
            self.svm_text.config(text="\n".join(svm_lines))

            mlp_lines = []
            for i, (cls, conf) in enumerate(mlp_top5, 1):
                cls_short = cls[:28] if len(cls) <= 28 else cls[:25] + "..."
                mlp_lines.append(f"{i}. {cls_short:<30s} {conf:5.1f}%")
            self.mlp_text.config(text="\n".join(mlp_lines))

        except Exception as e:
            err_msg = f"{type(e).__name__} : {e}"
            print("=" * 60)
            print("ERREUR pendant la prédiction :")
            traceback.print_exc()
            print("=" * 60)
            self.result_label.config(text=f"❌  Erreur : {err_msg}",
                                     fg="red", font=("Helvetica", 11))
            self.svm_text.config(text="")
            self.mlp_text.config(text="")
            messagebox.showerror("Erreur de prédiction",
                                 f"{err_msg}\n\nVoir la console pour le détail.")


# ---------------------------------------------------------------------------
# Lancement
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AircraftClassifierApp(root)
    root.mainloop()
