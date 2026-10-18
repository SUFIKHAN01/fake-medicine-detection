import os, zipfile, tempfile, random, shutil
from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Medicine Image Security Lab", page_icon="🛡️", layout="wide")
st.title("🛡️ Fake Medicine Image Classification")
st.caption("Train an actual image model, evaluate it, and test an uploaded image.")
st.warning("Research prototype only. Visual predictions cannot prove that a medicine is genuine or safe. Use trustworthy labels and do not use this app for real medical decisions.")

CLASSES = ["counterfeit", "genuine"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMG_SIZE = (224, 224)

def label_for(name):
    n = name.lower().strip().replace("_", " ").replace("-", " ")
    if n in {"genuine", "real", "authentic", "original", "legitimate"}: return "genuine"
    if n in {"counterfeit", "fake", "fraud", "forged"}: return "counterfeit"
    return None

def prepare_zip(upload):
    work = Path(tempfile.mkdtemp(prefix="medimg_"))
    zp = work / "data.zip"
    zp.write_bytes(upload.getvalue())
    extracted = work / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(zp) as z:
        for item in z.infolist():
            target = (extracted / item.filename).resolve()
            if not str(target).startswith(str(extracted.resolve()) + os.sep) and target != extracted.resolve():
                raise ValueError("Unsafe path found in ZIP.")
        z.extractall(extracted)
    found = None
    for d, subdirs, _ in os.walk(extracted):
        recognized = {label_for(x): x for x in subdirs if label_for(x)}
        if "genuine" in recognized and "counterfeit" in recognized:
            found = (Path(d), recognized); break
    if not found:
        raise ValueError("ZIP must contain two folders: genuine/ and counterfeit/ (or real/ and fake/).")
    parent, names = found
    out = work / "split"
    counts = {}
    for split in ["train", "val", "test"]:
        for cls in CLASSES: (out / split / cls).mkdir(parents=True, exist_ok=True)
    random.seed(42)
    for cls in CLASSES:
        paths = [p for p in (parent / names[cls]).rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        valid = []
        for p in paths:
            try:
                with Image.open(p) as im: im.verify()
                valid.append(p)
            except Exception: pass
        random.shuffle(valid)
        if len(valid) < 5:
            raise ValueError(f"Found only {len(valid)} valid images for {cls}. Add more images.")
        a, b = int(.70*len(valid)), int(.85*len(valid))
        groups = {"train": valid[:a], "val": valid[a:b], "test": valid[b:]}
        for split, files in groups.items():
            for i, src in enumerate(files):
                shutil.copy2(src, out / split / cls / f"{cls}_{i}{src.suffix.lower()}")
            counts[f"{split} {cls}"] = len(files)
    return work, out, counts

def datasets(path, batch):
    result = []
    for split in ["train", "val", "test"]:
        ds = tf.keras.utils.image_dataset_from_directory(
            str(path / split), class_names=CLASSES, image_size=IMG_SIZE,
            batch_size=batch, shuffle=(split == "train"), seed=42)
        result.append(ds.prefetch(tf.data.AUTOTUNE))
    return result

def build_cnn():
    m = tf.keras.Sequential([
        tf.keras.layers.Input((224,224,3)), tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16,3,activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32,3,activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dropout(.35), tf.keras.layers.Dense(1,activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def build_mobile():
    base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
    base.trainable = False
    x0 = tf.keras.Input((224,224,3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(.3)(x)
    y = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    m = tf.keras.Model(x0, y)
    m.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="binary_crossentropy", metrics=["accuracy"])
    return m

def evaluate(m, test_ds):
    y = np.concatenate([b.numpy().astype(int) for _, b in test_ds]).ravel()
    p = m.predict(test_ds, verbose=0).ravel()
    pred = (p >= .5).astype(int)
    met = {
        "Accuracy": accuracy_score(y,pred),
        "Counterfeit precision": precision_score(y,pred,pos_label=0,zero_division=0),
        "Counterfeit recall": recall_score(y,pred,pos_label=0,zero_division=0),
        "Counterfeit F1": f1_score(y,pred,pos_label=0,zero_division=0),
        "Genuine precision": precision_score(y,pred,pos_label=1,zero_division=0),
        "Genuine recall": recall_score(y,pred,pos_label=1,zero_division=0),
        "Genuine F1": f1_score(y,pred,pos_label=1,zero_division=0)}
    return met, y, pred

if "models" not in st.session_state: st.session_state.models = {}
if "results" not in st.session_state: st.session_state.results = {}

tab1, tab2, tab3, tab4 = st.tabs(["1 · Dataset", "2 · Train", "3 · Predict image", "4 · Results"])

with tab1:
    st.subheader("Upload labelled image dataset")
    st.markdown("ZIP structure:\n```text\nmedicine_dataset.zip\n├── genuine/       # reliably labelled genuine package images\n└── counterfeit/   # reliably labelled counterfeit package images\n```")
    st.info("Common aliases real/fake are accepted. A folder of ordinary medicine photos without verified counterfeit labels is not sufficient.")
    upload = st.file_uploader("Choose dataset ZIP", type=["zip"])
    if st.button("Prepare dataset", type="primary", disabled=upload is None):
        try:
            with st.spinner("Checking images and preparing splits..."):
                work, split, counts = prepare_zip(upload)
            st.session_state.work = str(work)
            st.session_state.split = str(split)
            st.session_state.counts = counts
            st.session_state.models = {}
            st.session_state.results = {}
            st.success("Dataset prepared.")
        except Exception as e: st.error(str(e))
    if "counts" in st.session_state:
        st.dataframe(pd.DataFrame([{"Split/Class":k,"Images":v} for k,v in st.session_state.counts.items()]), hide_index=True, use_container_width=True)
        st.caption("70% train / 15% validation / 15% test. Random splitting can leak near-duplicate images; split by original package/source when possible.")

with tab2:
    st.subheader("Train a real machine-learning model")
    if "split" not in st.session_state:
        st.info("Prepare a dataset first in the Dataset tab.")
    else:
        choice = st.selectbox("Model", ["Baseline CNN", "Improved MobileNetV2 (transfer learning)"])
        epochs = st.slider("Epochs", 2, 15, 5)
        batch = st.selectbox("Batch size", [8,16,32], index=1)
        if st.button("Start training", type="primary"):
            try:
                train_ds, val_ds, test_ds = datasets(Path(st.session_state.split), batch)
                with st.spinner("Training model and evaluating on held-out test images..."):
                    model = build_cnn() if choice == "Baseline CNN" else build_mobile()
                    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)],
                        verbose=0)
                    met, y, pred = evaluate(model, test_ds)
                key = "Baseline CNN" if choice == "Baseline CNN" else "MobileNetV2"
                st.session_state.models[key] = model
                st.session_state.results[key] = {"metrics":met,"history":hist.history,"y":y,"pred":pred}
                st.success(f"{key} training finished.")
            except Exception as e: st.error(f"Training failed: {e}")
        if st.session_state.results:
            fig, ax = plt.subplots()
            for name, r in st.session_state.results.items():
                ax.plot(r["history"].get("val_accuracy",[]), marker="o", label=name)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy"); ax.set_ylim(0,1); ax.legend(); ax.grid(alpha=.2)
            st.pyplot(fig); plt.close(fig)
            rows = []
            for name,r in st.session_state.results.items():
                rows.append({"Model":name, **{k:round(v*100,2) for k,v in r["metrics"].items()}})
            st.subheader("Held-out test metrics (%)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Try a medicine-package image")
    if not st.session_state.models:
        st.info("Train at least one model first.")
    else:
        model_name = st.selectbox("Choose trained model", list(st.session_state.models.keys()))
        image_file = st.file_uploader("Upload JPG or PNG", type=["jpg","jpeg","png","webp"])
        if image_file:
            try:
                image = Image.open(image_file).convert("RGB")
                st.image(image, caption="Uploaded image", width=350)
                if st.button("Predict image", type="primary"):
                    arr = np.array(image.resize(IMG_SIZE), dtype="float32")[None,...]
                    score = float(st.session_state.models[model_name].predict(arr,verbose=0)[0][0])
                    label = "Genuine-like" if score >= .5 else "Counterfeit-like"
                    confidence = score if score >= .5 else 1-score
                    st.subheader("Prediction: " + label)
                    st.metric("Confidence for predicted class", f"{confidence*100:.1f}%")
                    st.progress(min(max(confidence,0),1))
                    st.warning("Visual prediction only; it cannot authenticate a medicine or determine its chemical contents or safety.")
            except Exception as e: st.error(f"Could not process image: {e}")

with tab4:
    st.subheader("Evaluation and research limitations")
    if not st.session_state.results:
        st.info("Train a model to display evaluation results.")
    else:
        names = list(st.session_state.results.keys())
        rows = [{"Model":n,**{k:round(v*100,2) for k,v in st.session_state.results[n]["metrics"].items()}} for n in names]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        name = st.selectbox("Confusion matrix", names)
        r = st.session_state.results[name]
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(r["y"],r["pred"],labels=[0,1]),display_labels=CLASSES).plot(ax=ax,colorbar=False)
        st.pyplot(fig); plt.close(fig)
        st.download_button("Download metrics CSV",pd.DataFrame(rows).to_csv(index=False),"model_metrics.csv","text/csv")
    st.markdown("""
**Limitations to report**
- The model learns visual patterns; it does not examine chemical contents.
- Labels must be independently trustworthy.
- Duplicate or near-duplicate images across splits can inflate scores.
- High accuracy on one dataset does not prove real-world counterfeit detection.
- Treat this as a research screening prototype, not a real authentication system.
""")
