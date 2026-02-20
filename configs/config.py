DATA_PATH="../brain-tumor-dataset"
IMG_SIZE=256
BATCH_SIZE=8
LR=1e-4
NUM_EPOCHS=50
DEVICE="cuda" if __import__("torch").cuda.is_available() else "cpu"