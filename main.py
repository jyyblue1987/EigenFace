import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("yalefaces/")


def loadFaceMatrixForTraning():
    train_images = None
    test_images = None

    image_size = (100, 100)

    for i in range(0, 100):
        subject_prefix = "subject{:02d}".format(i)
        filename = [os.listdir()[i] for i in range(len(os.listdir())) if os.listdir()[i].startswith(subject_prefix)]

        for name in filename:
            image = plt.imread(name)
            image_size = image.shape
            image_vec = image.flatten("C")

            if "test" in name:
                if test_images is None:
                    test_images = image_vec
                else:
                    test_images = np.column_stack((test_images, image_vec))
            else:
                if train_images is None:
                    train_images = image_vec
                else:
                    train_images = np.column_stack((train_images, image_vec))

    return train_images, test_images, image_size


# load train, test image and vectorize
train, test, image_size = loadFaceMatrixForTraning()

mean = train.mean(axis=1, keepdims=True)

X = train - mean

# calculate covriance matrix
XTX = X.T.dot(X) / X.shape[0] # compute the X^TX matrix

# get the eigenvalues and eigenvectors
vals, eigen_vecs =  np.linalg.eig(XTX)

# sort them from high to low
eigen_vecs = -eigen_vecs[:,np.argsort(-vals)]
vals = vals[np.argsort(-vals)]

print("Eigen Values =", vals)

eigen_faces = train.dot(eigen_vecs[:, 0:6]) # compute the eigenface

# normalize eigen_faces
for i in range(eigen_faces.shape[1]):
    norm = np.linalg.norm(eigen_faces[:,i])
    eigen_faces[:,i] = eigen_faces[:,i] / norm


# check = eigen_faces.T.dot(eigen_faces)
# print(check)

print ("The top 6 eigenfaces")
for i in range(eigen_faces.shape[1]):
    face = eigen_faces[:,i]
    face_reshape = np.reshape(face, image_size) # reshape the data to its original form
    plt.subplot(2,3,i+1)
    plt.title('Eigenface_'+str(i))
    plt.imshow(face_reshape,cmap="Greys_r")
    plt.axis('off')

plt.show()

print("Calculate the normalized inner product score")
weight_train = train.T.dot(eigen_faces)
weight_test = test.T.dot(eigen_faces)
print("Train's Weight:", weight_train)
print("Test's Weight:", weight_test)
