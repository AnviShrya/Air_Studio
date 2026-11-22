"""
train_model.py  -  trains the handwriting recogniser used by main.py

Classes (15):  0 1 2 3 4 5 6 7 8 9  +  -  *  /  =
  * digits come from MNIST (with thickness augmentation, because air-drawn
    strokes are much thicker than MNIST pen strokes)
  * operators are generated synthetically (MNIST has none):
        +  plus          -  minus           *  a drawn "x"
        /  the division sign (a line with a dot above and below)
        =  two parallel lines

Run once:   python train_model.py
Output:     symbol_model.keras
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

keras = tf.keras

LABELS = list("0123456789+-*/=")
SYMBOLS = list("+-*/=")
TRAIN_PER_SYMBOL = 6000
TEST_PER_SYMBOL = 800
EPOCHS = 12
MODEL_PATH = "symbol_model.keras"

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Pre-processing  (IDENTICAL to prep_symbol() in main.py - keep them in sync)
# ---------------------------------------------------------------------------
def prep_symbol(crop):
    """crop: uint8 mask (0/255) of ONE symbol  ->  28x28 uint8, MNIST style."""
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = crop
    small = cv2.resize(sq, (20, 20), interpolation=cv2.INTER_AREA)
    return np.pad(small, 4)  # 20x20 -> 28x28 (4px border, like MNIST)


# ---------------------------------------------------------------------------
# Synthetic operator generator
# ---------------------------------------------------------------------------
def make_symbol(sym):
    S = 280
    img = np.zeros((S, S), np.uint8)
    t = int(rng.integers(6, 20))                      # stroke thickness
    cx = S // 2 + int(rng.integers(-15, 16))
    cy = S // 2 + int(rng.integers(-15, 16))
    L = int(rng.integers(35, 85))                     # half length (square-ish symbols)
    Lh = int(rng.integers(45, 95))                    # half length (wide symbols)

    def j(a=6):
        return int(rng.integers(-a, a + 1))

    def line(x1, y1, x2, y2):
        cv2.line(img, (x1 + j(), y1 + j()), (x2 + j(), y2 + j()), 255, t, cv2.LINE_AA)

    if sym == "+":
        L2 = max(30, L + j(10))
        line(cx - L, cy, cx + L, cy)
        line(cx, cy - L2, cx, cy + L2)
    elif sym == "-":
        line(cx - Lh, cy, cx + Lh, cy)
    elif sym == "*":
        line(cx - L, cy - L, cx + L, cy + L)
        line(cx - L, cy + L, cx + L, cy - L)
    elif sym == "/":                                  # division sign
        g = int(rng.integers(28, 55))
        r = max(4, t // 2 + int(rng.integers(0, 4)))
        line(cx - Lh, cy, cx + Lh, cy)
        cv2.circle(img, (cx + j(8), cy - g), r, 255, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + j(8), cy + g), r, 255, -1, cv2.LINE_AA)
    elif sym == "=":
        g = int(rng.integers(22, 50))
        line(cx - Lh, cy - g, cx + Lh, cy - g)
        line(cx - Lh, cy + g, cx + Lh, cy + g)

    # small random rotation (hand-drawn symbols are never perfectly straight)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), float(rng.uniform(-12, 12)), 1.0)
    img = cv2.warpAffine(img, M, (S, S))

    ys, xs = np.nonzero(img > 40)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return prep_symbol(crop)


def build_symbol_set(n_per_class):
    xs, ys = [], []
    for sym in SYMBOLS:
        for _ in range(n_per_class):
            xs.append(make_symbol(sym))
            ys.append(LABELS.index(sym))
    return np.array(xs, np.uint8), np.array(ys, np.int64)


def thicken(images):
    """Randomly dilate MNIST digits so they look like thick air-drawn strokes."""
    kernel = np.ones((3, 3), np.uint8)
    out = images.copy()
    for i in range(len(out)):
        r = rng.random()
        if r > 0.5:
            out[i] = cv2.dilate(out[i], kernel, iterations=1 if r < 0.8 else 2)
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Loading MNIST ...")
(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()

print("Generating synthetic operator symbols (about 30-60 seconds) ...")
sx_tr, sy_tr = build_symbol_set(TRAIN_PER_SYMBOL)
sx_te, sy_te = build_symbol_set(TEST_PER_SYMBOL)

x_train = np.concatenate([thicken(x_tr), sx_tr])
y_train = np.concatenate([y_tr.astype(np.int64), sy_tr])
x_test = np.concatenate([x_te, sx_te])
y_test = np.concatenate([y_te.astype(np.int64), sy_te])

perm = rng.permutation(len(x_train))
x_train, y_train = x_train[perm], y_train[perm]

x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
x_test = (x_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)

print("Train:", x_train.shape, " Test:", x_test.shape, " Classes:", len(LABELS))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
L_ = keras.layers
model = keras.Sequential([
    L_.Input(shape=(28, 28, 1)),

    # augmentation - only active while training
    L_.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
    L_.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
    L_.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),

    L_.Conv2D(32, 3, padding="same", activation="relu"),
    L_.BatchNormalization(),
    L_.Conv2D(32, 3, padding="same", activation="relu"),
    L_.MaxPooling2D(),
    L_.Dropout(0.25),

    L_.Conv2D(64, 3, padding="same", activation="relu"),
    L_.BatchNormalization(),
    L_.Conv2D(64, 3, padding="same", activation="relu"),
    L_.MaxPooling2D(),
    L_.Dropout(0.25),

    L_.Flatten(),
    L_.Dense(128, activation="relu"),
    L_.Dropout(0.4),
    L_.Dense(len(LABELS), activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=128,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%")

pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print("\nPer-class accuracy:")
for i, name in enumerate(LABELS):
    m = y_test == i
    print(f"  {name}: {np.mean(pred[m] == i) * 100:5.1f}%")

model.save(MODEL_PATH)
print(f"\nSaved -> {MODEL_PATH}")
