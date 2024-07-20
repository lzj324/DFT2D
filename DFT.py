import matplotlib.pyplot as plt
import numpy as np
import cv2

def set_close_elements_to_zero(matrix):
    rows, cols = matrix.shape
    center_row, center_col = rows // 2, cols // 2
    
    row_indices, col_indices = np.ogrid[:rows, :cols]
    distances_squared = (row_indices - center_row) ** 2 + (col_indices - center_col) ** 2
    matrix[distances_squared <= 100] = 0  
    
    return matrix

def DFT2D(x, shift=True):
    '''
    Discrete space fourier transform
    x: Input matrix
    '''
    pi2 = 2*np.pi
    N1, N2 = x.shape
    X = np.zeros((N1, N2), dtype=np.complex64)
    n1, n2 = np.mgrid[0:N1, 0:N2]

    for w1 in range(N1):
        for w2 in range(N2):
            j2pi = np.zeros((N1, N2), dtype=np.complex64)
            j2pi.imag = pi2*(w1*n1/N1 + w2*n2/N2)
            X[w1, w2] = np.sum(x*np.exp(-j2pi))
            print(w1,w2)
    if shift:
        X = np.roll(X, N1//2, axis=0)
        X = np.roll(X, N2//2, axis=1)
    return X


def iDFT2D(X, shift=True):
    '''
    Inverse discrete space fourier transform
    X: Complex matrix
    '''
    pi2 = 2*np.pi
    N1, N2 = X.shape
    x = np.zeros((N1, N2))
    k1, k2 = np.mgrid[0:N1, 0:N2]
    if shift:
        X = np.roll(X, -N1//2, axis=0)
        X = np.roll(X, -N2//2, axis=1)
    for n1 in range(N1):
        for n2 in range(N2):
            j2pi = np.zeros((N1, N2), dtype=np.complex64)
            j2pi.imag = pi2*(n1*k1/N1 + n2*k2/N2)
            x[n1, n2] = abs(np.sum(X*np.exp(j2pi)))
    return 1/(N1*N2)*x


if __name__ == "__main__":
    


    image = cv2.imread('./sample_img.png',0)
    image=cv2.resize(image,(128,128))
    N1, N2 = image.shape
    IMAGE = DFT2D(image)

    IMAGE_magnitude = np.log10(1+abs(IMAGE)).real
    IMAGE_magnitude = IMAGE_magnitude/np.max(IMAGE_magnitude)*255

    cv2.imwrite("./image_magnitude.png",IMAGE_magnitude)

    image_ = iDFT2D(IMAGE)
    cv2.imwrite("./restored_image.png",image_)
    
    IMAGE_highpass = IMAGE.copy()
    set_close_elements_to_zero(IMAGE_highpass)
    image_highpass = iDFT2D(IMAGE_highpass)
    cv2.imwrite("./restored_image_highpass.png",image_highpass)
    IMAGE_highpass_magnitude = np.log10(1+abs(IMAGE_highpass)).real
    IMAGE_highpass_magnitude = IMAGE_highpass_magnitude/np.max(IMAGE_highpass_magnitude)*255
    cv2.imwrite("./IMAGE_highpass_magnitude.png",IMAGE_highpass_magnitude)

