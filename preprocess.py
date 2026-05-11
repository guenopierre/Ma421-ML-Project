import os
import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition   import PCA


# ── rembg singleton ──────────────────────────────────────────────────────────

_rembg_session = None

def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            print("  Loading rembg model (u2net)...")
            _rembg_session = new_session("u2net")
            print("  rembg ready.")
        except ImportError:
            raise ImportError("Install rembg: pip install rembg")
    return _rembg_session


# ── Annotation loading ───────────────────────────────────────────────────────

def load_annotations(filepath):
    img_ids, labels = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                img_ids.append(line[:7])
                labels.append(line[8:].strip())
    return img_ids, labels


# ── Class balancing ──────────────────────────────────────────────────────────

def balance_classes_fn(img_ids, labels, target_count=None, random_state=42):
    """
    Undersample each class to target_count samples.
    If target_count is None, uses the size of the smallest class.
    """
    rng     = np.random.default_rng(random_state)
    img_ids = np.array(img_ids)
    labels  = np.array(labels)

    unique, counts = np.unique(labels, return_counts=True)
    if target_count is None:
        target_count = int(counts.min())

    print(f"  Balancing: {len(unique)} classes → {target_count} samples/class")

    selected = []
    for cls in unique:
        idx = np.where(labels == cls)[0]
        selected.extend(rng.choice(idx, size=min(target_count, len(idx)), replace=False))

    selected = np.array(selected)
    rng.shuffle(selected)
    print(f"  Balanced dataset: {len(selected)} samples (was {len(img_ids)})")
    return img_ids[selected].tolist(), labels[selected].tolist()


# ── Single-image preprocessing ───────────────────────────────────────────────

def preprocess_one_image(path, img_size, crop_bottom, use_rembg):
    """
    Preprocess one image:
        1. Read BGR → RGB
        2. Crop crop_bottom pixels from the bottom (Author removal)
        3. Background removal via rembg
        4. Resize to (img_size, img_size)

    Returns a uint8 numpy array of shape (img_size, img_size, 3).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h = img.shape[0]
    if crop_bottom > 0 and h > crop_bottom:
        img = img[:h - crop_bottom, :]

    if use_rembg:
        from rembg import remove
        pil_rgba = remove(Image.fromarray(img), session=_get_rembg_session())
        pil_out  = Image.new("RGB", pil_rgba.size, (0, 0, 0))
        pil_out.paste(pil_rgba, mask=pil_rgba.split()[3])
        img = np.array(pil_out)

    return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)


def load_one_raw(path, img_size, crop_bottom):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h = img.shape[0]
    if crop_bottom > 0 and h > crop_bottom:
        img = img[:h - crop_bottom, :]

    return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)


# ── Dataset-level preprocessing with cache ───────────────────────────────────

def preprocess_all_images(img_ids, image_dir, img_size, crop_bottom,
                           use_rembg, cache_dir):
    """
    Preprocess all images, writing results to cache_dir.
    Cached images are reloaded directly on subsequent runs.

    Returns a dict {img_id: flat float32 vector normalised to [0, 1]}.
    """
    os.makedirs(cache_dir, exist_ok=True)

    result     = {}
    errors     = 0
    from_cache = 0
    n          = len(img_ids)

    for i, img_id in enumerate(img_ids):
        src   = os.path.join(image_dir, img_id + ".jpg")
        cache = os.path.join(cache_dir,  img_id + ".jpg")

        try:
            if os.path.exists(cache):
                img = cv2.imread(cache, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Corrupted cache entry")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[0] != img_size:
                    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                from_cache += 1
            else:
                img = preprocess_one_image(src, img_size, crop_bottom, use_rembg)
                cv2.imwrite(cache, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            result[img_id] = img.astype(np.float32).flatten() / 255.0

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"    Warning — {img_id}: {e}")
            result[img_id] = None

        if (i + 1) % 200 == 0 or (i + 1) == n:
            tag = "+rembg" if use_rembg else "no-rembg"
            print(f"    {i+1}/{n}  [{tag}]  cache={from_cache}  errors={errors}")

    if errors:
        print(f"  {errors} image(s) failed — excluded from dataset.")
    print(f"  {from_cache}/{n} images loaded from cache.")
    return result


# ── Main pipeline ────────────────────────────────────────────────────────────

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
                      cache_dir        = "images_withoutback",
                      classes_to_keep  = None):

    annot_path = os.path.join(data_path, annotation_file)
    image_dir  = os.path.join(data_path, "images")
    abs_cache  = os.path.join(data_path, cache_dir) if cache_dir else None

    print(f"\n{'─'*55}")
    print(f"  Annotations : {annot_path}")
    print(f"  Images      : {image_dir}")
    print(f"  img_size={img_size}px  crop={crop_bottom}px  rembg={use_rembg}")
    print(f"  augment_bg={augment_with_bg}  PCA={num_pcs}  test={test_split*100:.0f}%")
    if abs_cache:
        print(f"  Cache       : {abs_cache}")
    print(f"{'─'*55}\n")

    # 1. Annotations
    print("1. Loading annotations...")
    img_ids, labels = load_annotations(annot_path)
    print(f"   {len(img_ids)} images  |  {len(set(labels))} classes")

    if classes_to_keep is not None:
        keep_set = set(classes_to_keep)
        before   = len(img_ids)
        img_ids  = [i for i, l in zip(img_ids, labels) if l in keep_set]
        labels   = [l for l in labels if l in keep_set]
        print(f"   Class filter: {before} → {len(img_ids)} images ({len(keep_set)} classes)")

    # 2. Class balancing
    if do_balance:
        print("\n2. Balancing classes...")
        img_ids, labels = balance_classes_fn(
            img_ids, labels, target_count=target_per_class, random_state=random_state)

    # 3. Image preprocessing
    print(f"\n3. Preprocessing {len(img_ids)} images...")
    if abs_cache:
        vectors_nobg = preprocess_all_images(
            img_ids, image_dir, img_size, crop_bottom, use_rembg, abs_cache)
    else:
        vectors_nobg = {}
        n = len(img_ids)
        for i, img_id in enumerate(img_ids):
            path = os.path.join(image_dir, img_id + ".jpg")
            try:
                img = preprocess_one_image(path, img_size, crop_bottom, use_rembg)
                vectors_nobg[img_id] = img.astype(np.float32).flatten() / 255.0
            except Exception:
                vectors_nobg[img_id] = None
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"    {i+1}/{n}")

    valid_mask = [vectors_nobg.get(img_id) is not None for img_id in img_ids]
    img_ids    = [img_ids[i] for i in range(len(img_ids)) if valid_mask[i]]
    labels     = [labels[i]  for i in range(len(labels))  if valid_mask[i]]
    print(f"   Valid images: {len(img_ids)}")

    # 4. Stratified split
    print("\n4. Stratified train/test split...")
    train_ids, test_ids, train_labels, test_labels = train_test_split(
        img_ids, labels, test_size=test_split, stratify=labels, random_state=random_state)
    print(f"   Train: {len(train_ids)}  |  Test: {len(test_ids)}")

    # 5. Label encoding
    le = LabelEncoder()
    le.fit(train_labels + test_labels)
    y_train = le.transform(train_labels)
    y_test  = le.transform(test_labels)
    classes = le.classes_

    # 6. Augmentation
    X_nobg = np.array([vectors_nobg[i] for i in train_ids], dtype=np.float32)

    if augment_with_bg and use_rembg:
        print(f"\n5. Augmentation: loading background-retaining images...")
        bg_vectors = []
        errors_bg  = 0
        n_aug      = len(train_ids)
        for j, img_id in enumerate(train_ids):
            path = os.path.join(image_dir, img_id + ".jpg")
            try:
                img = load_one_raw(path, img_size, crop_bottom)
                bg_vectors.append(img.astype(np.float32).flatten() / 255.0)
            except Exception:
                bg_vectors.append(np.zeros(img_size * img_size * 3, dtype=np.float32))
                errors_bg += 1
            if (j + 1) % 500 == 0 or (j + 1) == n_aug:
                print(f"    {j+1}/{n_aug}")

        X_bg    = np.array(bg_vectors, dtype=np.float32)
        y_nobg  = y_train.copy()

        X_train_raw = np.concatenate([X_nobg, X_bg], axis=0)
        y_train     = np.concatenate([y_train, y_train], axis=0)

        rng  = np.random.default_rng(random_state + 1)
        perm = rng.permutation(len(X_train_raw))
        X_train_raw = X_train_raw[perm]
        y_train     = y_train[perm]

        if errors_bg:
            print(f"   Warning: {errors_bg} background image(s) failed (zero vectors used).")
        print(f"   Augmented train set: {len(X_train_raw)} samples "
              f"({len(train_ids)} no-bg + {len(train_ids)} with-bg)")
    else:
        X_train_raw = X_nobg
        y_nobg      = y_train.copy()
        print(f"\n5. No augmentation (augment_with_bg=False or use_rembg=False).")

    X_test_raw = np.array([vectors_nobg[i] for i in test_ids], dtype=np.float32)
    print(f"\n   X_train: {X_train_raw.shape}  |  X_test: {X_test_raw.shape}")

    # 7. Normalisation (fit on no-background train images only)
    print("\n6. Normalisation (zero-mean, unit-variance)...")
    mu    = X_nobg.mean(axis=0)
    sigma = X_nobg.std(axis=0)
    sigma[sigma == 0] = 1.0

    X_train_norm = ((X_train_raw - mu) / sigma).astype(np.float32)
    X_test_norm  = ((X_test_raw  - mu) / sigma).astype(np.float32)

    # 8. PCA (fit on no-background train images only)
    print(f"\n7. PCA ({num_pcs} components)...")
    X_nobg_norm = ((X_nobg - mu) / sigma).astype(np.float32)
    num_pcs     = min(num_pcs, X_nobg_norm.shape[0] - 1, X_nobg_norm.shape[1])

    pca = PCA(n_components=num_pcs, random_state=random_state)
    pca.fit(X_nobg_norm)
    X_train = pca.transform(X_train_norm)
    X_test  = pca.transform(X_test_norm)

    print(f"   Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print("\n   Preprocessing complete.\n")

    X_nobg_train = pca.transform(X_nobg_norm)

    return {
        "X_train":      X_train,
        "X_nobg_train": X_nobg_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_nobg_train": y_nobg,
        "y_test":       y_test,
        "classes":      classes,
        "pca":          pca,
        "mu":           mu,
        "sigma":        sigma,
    }
