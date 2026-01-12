import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.datasets import fetch_openml
from skimage import io, color, transform

choice = True

# 1. ------- LOAD MNIST DATASET----------

print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float64)
y = mnist.target.astype(int)
X = mnist.data[:20000].astype(np.float64)
y = mnist.target[:20000].astype(int)


print("Dataset shape:", X.shape)  # (70000, 784)


# 2. ----- NORMALIZE (MEAN CENTERING)--------

mean = np.mean(X, axis=0)
X_centered = X - mean


# 3. ------- COVARIANCE MATRIX --------

N = X_centered.shape[0]
cov_matrix = np.dot(X_centered.T, X_centered) / N


# 4. ----- EIGEN DECOMPOSITION ---------

print("Computing eigenvalues and eigenvectors...")
eig_vals, eig_vecs = eigh(cov_matrix)

# Sort in descending order
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]


# 5. ------ PCA FUNCTIONS ----------

def project_pca(X, eigenvectors, k):
    W = eigenvectors[:, :k]
    return np.dot(X, W)

def reconstruct_pca(Z, eigenvectors, mean, k):
    W = eigenvectors[:, :k]
    return np.dot(Z, W.T) + mean


# 6.------ REDUCE DIMENSION -------

k = 50   # number of principal components
print(f"Projecting data onto top {k} components...")

Z_train = project_pca(X_centered, eig_vecs, k)

print("Reduced training shape:", Z_train.shape)


# 7. ----- IMAGE PREPROCESSING --------

def preprocess_image(image_path):

   # Loads an image, converts to grayscale, resizes to 28x28,normalizes it, and flattens it.

    img = io.imread(image_path)

    # If image has alpha channel (RGBA), remove it
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]   # Drop alpha channel

    # Convert to grayscale if RGB
    if img.ndim == 3:
        img = color.rgb2gray(img)

    # Resize to 28x28
    img_resized = transform.resize(img, (28, 28), anti_aliasing=True)

    # Invert if background is white (MNIST: white digit on black)
    if np.mean(img_resized) > 0.5:
        img_resized = 1.0 - img_resized

    # Normalize to [0,255]
    img_resized = img_resized * 255.0

    # Flatten to 784 vector
    img_vector = img_resized.flatten()

    return img_vector

# 8. ---- DIGIT PREDICTION (NEAREST NEIGHBOR) -------

def predict_digit(image_path, eig_vecs, mean, Z_train, y_train, k):
    # Preprocess input image
    x_input = preprocess_image(image_path)

    # Mean center
    x_centered = x_input - mean

    # Project onto PCA space
    z_input = project_pca(x_centered.reshape(1, -1), eig_vecs, k)

    # Compute Euclidean distances to all training samples
    distances = np.linalg.norm(Z_train - z_input, axis=1)

    # Find nearest neighbor
    nearest_index = np.argmin(distances)
    predicted_label = y_train[nearest_index]

    return predicted_label, x_input


# 9.--- USER INPUT -------------

while choice == True:
    image_path = input("Enter path to digit image (28x28 or any size): ")

    predicted_digit, processed_img = predict_digit(image_path, eig_vecs, mean, Z_train, y, k)

    print("\nPredicted Digit:", predicted_digit)

    # 10. --- DISPLAY INPUT IMAGE --------

    plt.figure()
    plt.imshow(processed_img.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.axis('off')
    plt.show()
    plt.pause(2.0)  # Keep it open for 2 seconds
    plt.close()

    ans = input("Do you want to predict another digit? Y/N ").lower()
    if ans == "n" :
        choice = False



