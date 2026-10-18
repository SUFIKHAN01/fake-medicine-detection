# Fake Medicine Image Classification — Streamlit

This app trains an actual CNN or MobileNetV2 image classifier from a labelled image dataset, evaluates it on held-out test images, displays metrics, and predicts on an uploaded image.

## Dataset
Upload a ZIP with two folders at the same level:
- `genuine/`: reliably labelled genuine images
- `counterfeit/`: reliably labelled counterfeit images

Aliases `real/` and `fake/` are accepted. The app needs at least five valid images per class just to run; a meaningful project needs much more diverse, trustworthy data.

## Run locally
1. Install a compatible Python version and dependencies: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. Open the local URL printed in the terminal.
4. Prepare dataset, train a model, test an image, and view results.

## Deploy
Upload `app.py` and `requirements.txt` to GitHub, then deploy through https://share.streamlit.io. Training or loading MobileNetV2 may exceed free hosting resources. For larger training runs, train in Colab and adapt the app to load the saved model.

## Research cautions
Visual classification cannot prove that a medicine is genuine or safe. Dataset labels, source diversity, duplicate leakage, and independent test data are critical. Do not claim real-world accuracy from a small or synthetic dataset.
