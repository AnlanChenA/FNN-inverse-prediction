import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# ---------Model definition

def build_model():
    """FNN architecture (must match training script)."""
    return nn.Sequential(
        nn.Linear(784 * 2, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 784),
    )

#---------------- Load trained model
def load_checkpoint(ckpt_path):
    """Load model weights and normalization statistics."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm = {
        "m1": ckpt["X1_mean"],
        "s1": ckpt["X1_std"],
        "m2": ckpt["X2_mean"],
        "s2": ckpt["X2_std"],
    }

    return model, norm



# --------Load and normalize data

def load_test_data(x1_path, x2_path, y_path, norm):
    """Load test inputs and apply training-set normalization."""
    X_test1 = np.loadtxt(x1_path)
    X_test2 = np.loadtxt(x2_path)
    y_test = np.loadtxt(y_path)

    X_test1 = (X_test1 - norm["m1"]) / norm["s1"]
    X_test2 = (X_test2 - norm["m2"]) / norm["s2"]

    X_test = np.hstack((X_test1, X_test2))
    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_test, y_test

# -------Visualization
def visualize_prediction(y_true, y_pred, idx=None):
    """Visualize one random prediction."""
    if idx is None:
        idx = np.random.randint(len(y_true))

    true_img = y_true[idx].reshape(28, 28)
    pred_img = y_pred[idx].reshape(28, 28)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(true_img, cmap="viridis")
    plt.title("True")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_img, cmap="viridis")
    plt.title("Predicted")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#------main
def main():
    ckpt_path = "inverse_fnn_checkpoint.pt"

    x1_path = "test_data/testx1"
    x2_path = "test_data/testx2"
    y_path = "test_data/testy"

    model, norm = load_checkpoint(ckpt_path)
    X_test, y_test = load_test_data(x1_path, x2_path, y_path, norm)

    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    visualize_prediction(y_test, y_pred)


if __name__ == "__main__":
    main()