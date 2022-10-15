import numpy as np
import cv2

def line_energy(image):
    # implement line energy (i.e. image intensity)
    image = (image - (np.amin(image))) * (1. / (np.amax(image) - np.amin(image)))
    return image.astype(np.float64)

def edge_energy(image):
    # implement edge energy (i.e. gradient magnitude)
    sx = np.array([[-1, 0, 1]])
    sy = np.array([[-1, 0, 1]]).T

    dx = cv2.filter2D(image, cv2.CV_64F, sx)
    dy = cv2.filter2D(image, cv2.CV_64F, sy)

    d_mag = - np.sqrt(dx**2 + dy**2)

    # normalize gradient magnitude
    d_mag = (d_mag - (np.amin(d_mag))) * (1. / (np.amax(d_mag) - np.amin(d_mag)))

    cv2.imshow(f'Edge Energy', d_mag)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return d_mag

def term_energy(image):
    # implement term energy (i.e. curvature)
    sx = np.array([[-1, 0, 1]])
    sy = np.array([[-1, 0, 1]]).T

    cx = cv2.filter2D(image, cv2.CV_64F, sx)
    cv2.imshow(f'cx ', cx)
    cy = cv2.filter2D(image, cv2.CV_64F, sy)
    cv2.imshow(f'cy ', cy)
    cxx = cv2.Sobel(image, cv2.CV_64F, 2, 0)
    cv2.imshow(f'cxx ', cxx)
    cyy = cv2.Sobel(image, cv2.CV_64F, 0, 2)
    cv2.imshow(f'cyy ', cyy)
    cxy = cv2.filter2D(cx, cv2.CV_64F, sy)
    cv2.imshow(f'cxy ', cxy)


    nu = (cxx * (cy**2)) - (2 * cxy * cx * cy) + (cyy * (cx**2))
    de = (cx**2 + cy**2)**(3/2)
    e_term = cv2.divide(nu, de)

    cv2.imshow(f'Term Energy Raw', e_term)
    e_term[np.isnan(e_term)] = 0.

    # Normalization of Eterm
    e_term = (e_term - (np.amin(e_term))) * (1. / (np.amax(e_term) - np.amin(e_term)))
    cv2.imshow(f'Term Energy Normalized', e_term)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    return e_term

def external_energy(image, w_line, w_edge, w_term):
    # implement external energy
    e_line = line_energy(image)
    e_edge = edge_energy(image)
    e_term = term_energy(image)

    e_external = (w_line * e_line) + (w_edge * e_edge) + (w_term * e_term)

    # Normalisation
    e_external = (e_external - (np.amin(e_external))) * (1. / (np.amax(e_external) - np.amin(e_external)))

    cv2.imshow(f'External Energy Normalized ({w_line}, {w_edge}, {w_term})', e_external)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return e_external