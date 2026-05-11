import os, sys, traceback
import numpy as np
import cv2
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rembg import remove

MODEL_FILE = "model_aircraft.pkl"


def _from_session(name):
    """Retrieve a variable from the __main__ namespace."""
    mod = sys.modules.get('__main__')
    return getattr(mod, name, None) if mod else None


class AircraftClassifierApp:

    def __init__(self, root):
        self.root = root
        self.root.title("IPSA — Aircraft Classifier (Weighted Ensemble)")
        self.root.geometry("1000x900")
        self.root.configure(bg="#f8f9fa")

        self.data = self.mlp = self.svm = None
        self._build_ui()
        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        if not os.path.exists(MODEL_FILE):
            return
        try:
            content   = joblib.load(MODEL_FILE)
            self.data = content.get('data')
            self.mlp  = content.get('mlp')
            self.svm  = content.get('svm')
            print(f"Models loaded — "
                  f"w_mlp={self.data.get('best_wmlp', 0.5):.2f} / "
                  f"w_svm={self.data.get('best_wsvm', 0.5):.2f}")
        except Exception as e:
            print(f"Load error: {e}")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        tk.Label(self.root, text="Multi-Model Ensemble Classifier",
                 font=("Helvetica", 20, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(pady=15)

        # Image display
        self.img_frame = tk.Frame(self.root, width=500, height=350,
                                  bg="white", relief="groove", borderwidth=2)
        self.img_frame.pack(pady=10)
        self.img_frame.pack_propagate(False)
        self.img_label = tk.Label(self.img_frame, text="Open an image to begin",
                                  bg="white", fg="#95a5a6")
        self.img_label.pack(expand=True, fill=tk.BOTH)

        # Verdict panel
        self.verdict_frame = tk.Frame(self.root, bg="#ebf5fb", pady=10)
        self.verdict_frame.pack(fill=tk.X, padx=50, pady=10)
        tk.Label(self.verdict_frame, text="ENSEMBLE PREDICTION",
                 font=("Arial", 10, "bold"), bg="#ebf5fb", fg="#2980b9").pack()
        self.res_main = tk.Label(self.verdict_frame, text="Awaiting input...",
                                 font=("Helvetica", 22, "bold"), bg="#ebf5fb", fg="#2c3e50")
        self.res_main.pack()

        # Buttons
        btn_f = tk.Frame(self.root, bg="#f8f9fa")
        btn_f.pack(pady=15)
        tk.Button(btn_f, text="Open Image", command=self.load_image,
                  width=15, bg="#34495e", fg="white").grid(row=0, column=0, padx=10)
        self.go_btn = tk.Button(btn_f, text="Analyse", command=self.predict,
                                state=tk.DISABLED, width=15, bg="#27ae60",
                                fg="white", font=("Arial", 10, "bold"))
        self.go_btn.grid(row=0, column=1, padx=10)

        # Confidence panels
        details_f = tk.Frame(self.root, bg="#f8f9fa")
        details_f.pack(fill=tk.BOTH, expand=True, padx=40)
        self.mlp_ui = self._confidence_box(details_f, "MLP Confidence")
        self.svm_ui = self._confidence_box(details_f, "SVM Confidence")

    def _confidence_box(self, parent, title):
        box = tk.LabelFrame(parent, text=title, font=("Arial", 9, "bold"))
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        lbl = tk.Label(box, text="", justify=tk.LEFT, font=("Courier", 10),
                       bg="white", anchor="nw", padx=10, pady=10)
        lbl.pack(fill=tk.BOTH, expand=True)
        return lbl

    # ── Image loading ─────────────────────────────────────────────────────────

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.image_path = path
        img = Image.open(path)
        img.thumbnail((480, 330))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img, text="")
        self.cv_img = cv2.imread(path)
        self.go_btn.config(state=tk.NORMAL)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self):
        try:
            data = self.data or _from_session('data')
            mlp  = self.mlp  or _from_session('mlp')
            svm  = self.svm  or _from_session('svm')

            if not data:
                return messagebox.showerror("Error", "No model loaded.")

            if 'le' in data:
                classes = data['le'].classes_
            elif 'classes' in data:
                classes = data['classes']
            else:
                classes = _from_session('classes')

            self.res_main.config(text="Processing...", fg="orange")
            self.root.update()

            # Preprocessing
            img     = self.cv_img.copy()
            h_crop  = data.get('CROP_BOTTOM', 20)
            if img.shape[0] > h_crop:
                img = img[:-h_crop, :]

            img_rgba = remove(img)
            img_rgb  = np.array(Image.fromarray(img_rgba).convert("RGB"))
            size     = data.get('IMG_SIZE', 128)
            img_res  = cv2.resize(img_rgb, (size, size))

            x     = img_res.flatten().reshape(1, -1).astype(np.float32) / 255.0
            x_pca = data['pca'].transform((x - data['mu']) / data['sigma'])

            # Inference
            p_mlp   = mlp.predict_proba(x_pca)[0]
            p_svm   = svm.predict_proba(x_pca)[0]
            w_mlp   = data.get('best_wmlp', 0.5)
            w_svm   = data.get('best_wsvm', 0.5)
            p_final = w_mlp * p_mlp + w_svm * p_svm

            self.res_main.config(text=classes[np.argmax(p_final)], fg="#27ae60")

            def top5(p):
                return sorted(zip(classes, p * 100), key=lambda x: x[1], reverse=True)[:5]

            self.mlp_ui.config(text="\n".join(f"{c[:20]:<20} {v:>5.1f}%" for c, v in top5(p_mlp)))
            self.svm_ui.config(text="\n".join(f"{c[:20]:<20} {v:>5.1f}%" for c, v in top5(p_svm)))

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    AircraftClassifierApp(root)
    root.mainloop()