"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Pipeline de préprocessing complet :
  1. Lecture du fichier d'annotations unique
  2. (optionnel) Équilibrage des classes
  3. Préprocessing de TOUTES les images (train + test confondus) :
       a. Chargement en RGB
       b. Crop des N pixels du bas  (bandeau auteur)
       c. Suppression du fond par rembg (U2Net)   ← avant le resize
       d. Redimensionnement à img_size × img_size
       e. Sauvegarde dans images_withoutback/      ← cache pour relances rapides
  4. Split stratifié 70/30
  5. (optionnel) Data augmentation sur le train uniquement :
       - On ajoute la version "avec fond" (images originales) de chaque image
         de train, en plus de la version sans fond
       - Le mélange final est aléatoire (random_state différent)
       - Le test n'est PAS augmenté → évaluation propre
  6. Centrage-réduction (fit sur train uniquement)
  7. ACP (fit sur train, transform train + test)

Dépendance externe pour le détourage :
  pip install rembg onnxruntime
  Le modèle u2net (~170 MB) est téléchargé automatiquement au 1er lancement.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import cv2
from PIL import Image

from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection  import train_test_split
from sklearn.decomposition    import PCA


# ═══════════════════════════════════════════════════════════════════════════
#  Singleton rembg  (modèle chargé une seule fois en mémoire)
# ═══════════════════════════════════════════════════════════════════════════

_rembg_session = None

def _get_rembg_session():
    """Charge la session rembg (U2Net) une seule fois."""
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            print("  Chargement du modèle rembg (u2net)...")
            print("  (le modèle ~170 MB est téléchargé automatiquement au 1er lancement)")
            _rembg_session = new_session("u2net")
            print("  Modèle rembg prêt.")
        except ImportError:
            raise ImportError(
                "rembg n'est pas installé.\n"
                "Lance : pip install rembg onnxruntime\n"
                "Le modèle u2net (~170 MB) sera téléchargé automatiquement."
            )
    return _rembg_session


# ═══════════════════════════════════════════════════════════════════════════
#  Lecture des annotations
# ═══════════════════════════════════════════════════════════════════════════

def load_annotations(filepath):
    """
    Lit un fichier d'annotations FGVC-Aircraft.
    Format par ligne :  <7-char image_id> <famille>
    Retourne deux listes : img_ids, labels.
    """
    img_ids, labels = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_ids.append(line[:7])
            labels.append(line[8:].strip())
    return img_ids, labels


# ═══════════════════════════════════════════════════════════════════════════
#  Équilibrage des classes
# ═══════════════════════════════════════════════════════════════════════════

def balance_classes_fn(img_ids, labels, target_count=None, random_state=42):
    """
    Sous-échantillonne chaque classe pour qu'elles aient toutes le même
    nombre d'images. Si target_count=None, prend la taille de la plus petite.
    """
    rng     = np.random.default_rng(random_state)
    img_ids = np.array(img_ids)
    labels  = np.array(labels)

    unique_classes, counts = np.unique(labels, return_counts=True)
    if target_count is None:
        target_count = int(counts.min())

    print(f"  Équilibrage : {len(unique_classes)} classes → {target_count} images/classe")

    selected = []
    for cls in unique_classes:
        idx    = np.where(labels == cls)[0]
        chosen = rng.choice(idx, size=min(target_count, len(idx)), replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(selected)
    rng.shuffle(selected)
    print(f"  Dataset équilibré : {len(selected)} images (était {len(img_ids)})")
    return img_ids[selected].tolist(), labels[selected].tolist()


# ═══════════════════════════════════════════════════════════════════════════
#  Préprocessing d'une image individuelle
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_one_image(path, img_size, crop_bottom, use_rembg):
    """
    Pipeline sur une image :
      1. Lecture BGR → RGB
      2. Crop du bas (bandeau auteur)           ← haute résolution
      3. Suppression fond rembg (optionnel)     ← haute résolution
      4. Redimensionnement à img_size²

    Retourne une image numpy uint8 (img_size, img_size, 3).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if crop_bottom > 0 and h > crop_bottom:
        img = img[:h - crop_bottom, :]

    if use_rembg:
        from rembg import remove
        session  = _get_rembg_session()
        pil_in   = Image.fromarray(img)
        pil_rgba = remove(pil_in, session=session)
        pil_out  = Image.new("RGB", pil_rgba.size, (0, 0, 0))
        pil_out.paste(pil_rgba, mask=pil_rgba.split()[3])
        img = np.array(pil_out)

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img   # uint8 (img_size, img_size, 3)


def load_one_raw(path, img_size, crop_bottom):
    """
    Charge une image originale SANS supprimer le fond.
    Utilisé pour la data augmentation (version "avec fond").

    Pipeline :
      1. Lecture BGR → RGB
      2. Crop des crop_bottom pixels du bas  (bandeau auteur)
      3. Redimensionnement à img_size × img_size
    Les images ne sont PAS sauvegardées (pas de cache).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Crop du bas — même crop que pour les images sans fond
    h, w = img.shape[:2]
    if crop_bottom > 0 and h > crop_bottom:
        img = img[:h - crop_bottom, :]

    # 2. Redimensionnement — même taille que les images sans fond
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    return img


# ═══════════════════════════════════════════════════════════════════════════
#  Préprocessing de TOUT le dataset (avec cache)
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_all_images(img_ids, image_dir, img_size, crop_bottom,
                          use_rembg, cache_dir):
    """
    Préprocesse TOUTES les images et les sauvegarde dans cache_dir.
    Si une image existe déjà dans le cache, elle est rechargée directement.

    Retourne un dict  img_id → vecteur float64 normalisé [0,1]  (sans fond).
    """
    os.makedirs(cache_dir, exist_ok=True)

    result     = {}
    errors     = 0
    from_cache = 0
    n          = len(img_ids)

    for i, img_id in enumerate(img_ids):
        src_path   = os.path.join(image_dir, img_id + ".jpg")
        cache_path = os.path.join(cache_dir,  img_id + ".jpg")

        try:
            if os.path.exists(cache_path):
                img = cv2.imread(cache_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Cache corrompu")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Le cache peut avoir été sauvegardé avec un IMG_SIZE différent
                # (ex: 256) → on resize à la taille demandée sans sauvegarder
                if img.shape[0] != img_size or img.shape[1] != img_size:
                    img = cv2.resize(img, (img_size, img_size),
                                     interpolation=cv2.INTER_AREA)
                from_cache += 1
            else:
                img = preprocess_one_image(src_path, img_size,
                                           crop_bottom, use_rembg)
                cv2.imwrite(cache_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            result[img_id] = img.astype(np.float32).flatten() / 255.0

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"    ⚠ {img_id} : {e}")
            result[img_id] = None

        if (i + 1) % 200 == 0 or (i + 1) == n:
            rembg_str = "+rembg" if use_rembg else "sans rembg"
            print(f"    {i+1}/{n}  [{rembg_str}]  cache={from_cache}  erreurs={errors}")

    if errors:
        print(f"\n  ⚠ {errors} image(s) non chargée(s) — exclues du dataset.")
    print(f"  {from_cache}/{n} images rechargées depuis le cache.")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

def run_preprocessing(data_path        = ".",
                      annotation_file  = "images_family_trainval.txt",
                      img_size         = 64,
                      crop_bottom      = 20,
                      use_rembg        = True,
                      augment_with_bg  = True,
                      do_balance       = True,
                      target_per_class = None,
                      num_pcs          = 150,
                      test_split       = 0.30,
                      random_state     = 42,
                      cache_dir        = "images_withoutback"):
    """
    Pipeline complet.

    Paramètres
    ----------
    data_path        : dossier contenant images/ et le fichier d'annotations
    annotation_file  : nom du fichier d'annotations
    img_size         : taille de redimensionnement (carré, pixels)
    crop_bottom      : pixels à retirer en bas de chaque image
    use_rembg        : True = supprime le fond avec rembg (U2Net)
    augment_with_bg  : True = ajoute les images originales (avec fond) au train
                       → double le train, le test reste inchangé
    do_balance       : True = équilibre les classes
    target_per_class : images/classe après équilibrage (None = min)
    num_pcs          : composantes PCA à conserver
    test_split       : fraction réservée au test
    random_state     : graine aléatoire
    cache_dir        : sous-dossier de cache dans data_path (None = désactivé)

    Retourne
    --------
    dict : X_train, X_test, y_train, y_test, classes, pca, mu, sigma
    """
    annot_path = os.path.join(data_path, annotation_file)
    image_dir  = os.path.join(data_path, "images")
    abs_cache  = os.path.join(data_path, cache_dir) if cache_dir else None

    print(f"\n{'─'*55}")
    print(f"  Annotations  : {annot_path}")
    print(f"  Images       : {image_dir}")
    print(f"  IMG_SIZE={img_size}px  crop={crop_bottom}px  rembg={use_rembg}")
    print(f"  augment_bg={augment_with_bg}  PCA={num_pcs}  test={test_split*100:.0f}%")
    if abs_cache:
        print(f"  Cache        : {abs_cache}")
    print(f"{'─'*55}\n")

    # ── 1. Lecture des annotations ────────────────────────────────────────
    print("1. Lecture des annotations...")
    img_ids, labels = load_annotations(annot_path)
    print(f"   {len(img_ids)} images  |  {len(set(labels))} classes")

    # ── 2. Équilibrage des classes ────────────────────────────────────────
    if do_balance:
        print("\n2. Équilibrage des classes...")
        img_ids, labels = balance_classes_fn(
            img_ids, labels,
            target_count=target_per_class,
            random_state=random_state
        )

    # ── 3. Préprocessing de TOUTES les images (avant le split) ───────────
    # On préprocesse tout le dataset avant de splitter pour que le cache
    # contienne TOUTES les images (train ET test).
    print(f"\n3. Préprocessing de toutes les images ({len(img_ids)})...")
    if abs_cache:
        vectors_nobg = preprocess_all_images(
            img_ids, image_dir, img_size, crop_bottom, use_rembg, abs_cache
        )
    else:
        vectors_nobg = {}
        n = len(img_ids)
        for i, img_id in enumerate(img_ids):
            path = os.path.join(image_dir, img_id + ".jpg")
            try:
                img = preprocess_one_image(path, img_size, crop_bottom, use_rembg)
                vectors_nobg[img_id] = img.astype(np.float32).flatten() / 255.0
            except Exception as e:
                vectors_nobg[img_id] = None
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"    {i+1}/{n}...")

    # Retire les images invalides
    valid_mask = [vectors_nobg.get(img_id) is not None for img_id in img_ids]
    img_ids = [img_ids[i] for i in range(len(img_ids)) if valid_mask[i]]
    labels  = [labels[i]  for i in range(len(labels))  if valid_mask[i]]
    print(f"   Images valides : {len(img_ids)}")

    # ── 4. Split stratifié 70/30 ──────────────────────────────────────────
    print("\n4. Split stratifié train / test...")
    (train_ids, test_ids,
     train_labels, test_labels) = train_test_split(
        img_ids, labels,
        test_size    = test_split,
        stratify     = labels,
        random_state = random_state
    )
    print(f"   Train : {len(train_ids)}  |  Test : {len(test_ids)}")

    # ── 5. Encodage des labels ────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(train_labels + test_labels)
    y_train = le.transform(train_labels)
    y_test  = le.transform(test_labels)
    classes = le.classes_

    # ── 6. Construction de X_train (+ data augmentation) ─────────────────
    # Version sans fond (depuis le cache rembg)
    X_nobg = np.array([vectors_nobg[img_id] for img_id in train_ids],
                      dtype=np.float32)

    if augment_with_bg and use_rembg:
        # Version avec fond : crop 20px + resize img_size (pas de cache)
        print(f"\n5. Data augmentation : ajout des images originales (avec fond)...")
        print(f"   Pipeline : crop {crop_bottom}px bas → resize {img_size}×{img_size}  (non sauvegardées)")
        bg_vectors = []
        errors_bg  = 0
        n_aug = len(train_ids)
        for j, img_id in enumerate(train_ids):
            path = os.path.join(image_dir, img_id + ".jpg")
            try:
                img = load_one_raw(path, img_size, crop_bottom)
                bg_vectors.append(img.astype(np.float32).flatten() / 255.0)
            except Exception as e:
                bg_vectors.append(np.zeros(img_size * img_size * 3, dtype=np.float32))
                errors_bg += 1
            if (j + 1) % 500 == 0 or (j + 1) == n_aug:
                print(f"    {j+1}/{n_aug} images originales chargées...")
        X_bg = np.array(bg_vectors, dtype=np.float32)

        # Concaténation : [images sans fond] + [images avec fond]
        X_train_raw = np.concatenate([X_nobg, X_bg], axis=0)
        # Labels doublés dans le même ordre
        y_train     = np.concatenate([y_train, y_train], axis=0)

        # Mélange aléatoire (graine différente pour ne pas avoir les deux
        # versions d'une même image côte à côte)
        rng   = np.random.default_rng(random_state + 1)
        perm  = rng.permutation(len(X_train_raw))
        X_train_raw = X_train_raw[perm]
        y_train     = y_train[perm]

        if errors_bg:
            print(f"   ⚠ {errors_bg} image(s) avec fond non chargée(s) (vecteurs nuls).")
        print(f"   Train augmenté : {len(X_train_raw)} images "
              f"({len(train_ids)} sans fond + {len(train_ids)} avec fond)")
    else:
        X_train_raw = X_nobg
        print(f"\n5. Pas d'augmentation (augment_with_bg=False ou use_rembg=False).")

    # X_test_nobg : images de test sans fond (depuis le cache rembg)
    X_test_raw = np.array([vectors_nobg[img_id] for img_id in test_ids],
                          dtype=np.float32)

    # X_test_bg : images de test avec fond (crop + resize, non sauvegardées)
    # Utilisées pour le test-time augmentation (TTA) dans evaluate.py :
    # on prédit sur les deux versions et on combine les probabilités.
    print(f"\n5b. Chargement des images de test originales (avec fond) pour TTA...")
    test_bg_list = []
    errors_test_bg = 0
    for img_id in test_ids:
        path = os.path.join(image_dir, img_id + ".jpg")
        try:
            img = load_one_raw(path, img_size, crop_bottom)
            test_bg_list.append(img.astype(np.float32).flatten() / 255.0)
        except Exception as e:
            test_bg_list.append(np.zeros(img_size * img_size * 3, dtype=np.float32))
            errors_test_bg += 1
    X_test_bg_raw = np.array(test_bg_list, dtype=np.float32)
    if errors_test_bg:
        print(f"   ⚠ {errors_test_bg} image(s) test avec fond non chargée(s).")
    print(f"   X_test_bg : {X_test_bg_raw.shape}")

    print(f"\n   X_train final : {X_train_raw.shape}")
    print(f"   X_test        : {X_test_raw.shape}")

    # ── 7. Centrage-réduction ─────────────────────────────────────────────
    # mu et sigma sont calculés SUR LES IMAGES SANS FOND uniquement (X_nobg)
    # pour que la normalisation soit cohérente avec l'espace "propre" de l'ACP.
    # Les images avec fond (augmentation) sont normalisées avec ces mêmes stats.
    print("\n6. Centrage-réduction...")
    mu    = X_nobg.mean(axis=0)
    sigma = X_nobg.std(axis=0)
    sigma[sigma == 0] = 1.0

    X_train_norm    = ((X_train_raw    - mu) / sigma).astype(np.float32)
    X_test_norm     = ((X_test_raw     - mu) / sigma).astype(np.float32)
    X_test_bg_norm  = ((X_test_bg_raw  - mu) / sigma).astype(np.float32)

    # ── 8. ACP ────────────────────────────────────────────────────────────
    # L'ACP est fittée SUR LES IMAGES SANS FOND uniquement (X_nobg normalisé).
    # Raison : mélanger images avec/sans fond dans le fit PCA crée des
    # composantes qui capturent la différence fond/pas-fond plutôt que la
    # forme de l'avion. En fittant sur les images propres, les axes principaux
    # décrivent la morphologie des avions → meilleure séparation des classes.
    print(f"\n7. ACP ({num_pcs} composantes)...")
    X_nobg_norm = ((X_nobg - mu) / sigma).astype(np.float32)
    max_pcs = min(X_nobg_norm.shape[0] - 1, X_nobg_norm.shape[1])
    num_pcs = min(num_pcs, max_pcs)

    pca = PCA(n_components=num_pcs, random_state=random_state)
    pca.fit(X_nobg_norm)                         # fit sur images sans fond
    X_train    = pca.transform(X_train_norm)     # transform tout le train augmenté
    X_test     = pca.transform(X_test_norm)      # test sans fond
    X_test_bg  = pca.transform(X_test_bg_norm)   # test avec fond (pour TTA)

    var_exp = pca.explained_variance_ratio_.sum() * 100
    print(f"   Variance expliquée : {var_exp:.1f}%")
    print("\n   ✔ Préprocessing terminé.\n")

    return {
        "X_train":   X_train,
        "X_test":    X_test,      # test sans fond  (version principale)
        "X_test_bg": X_test_bg,   # test avec fond  (pour TTA)
        "y_train":   y_train,
        "y_test":    y_test,
        "classes":   classes,
        "pca":       pca,
        "mu":        mu,
        "sigma":     sigma,
    }