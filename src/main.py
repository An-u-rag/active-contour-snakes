import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import math

from internal_energy_matrix import get_matrix
from external_energy import external_energy

# Event to handle click event to generate initial user selected points for snake
def initial_point_select(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append array of points
        xs.append(x)
        ys.append(y)

        # display point
        cv2.circle(img, (x, y), 3, 128, -1)
        cv2.imshow("square", img)

if __name__ == '__main__':
    # read the image in grayscale
    img_color = cv2.imread('./venv/images/square.jpg')
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    image = img.copy()
    print('Original: ', img)

    # Define n snake points (x(s), y(s)) where xs and ys are arrays holding x and y coordinates respectively
    xs = []
    ys = []

    # Show the image and listen for click events and quit on pressing 'q'
    cv2.imshow("square", img)
    cv2.setMouseCallback("square", initial_point_select)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        img = image.copy()

    print(f'xs : ${xs}')
    print(f'ys : ${ys}')

    # interpolate
    # implement part 1: interpolate between the selected points
    # Need to define number of equidistant points including the user selected points
    n = 150
    # number of user generated points
    knots = len(xs)

    # Interpolate user generated points evenly
    xs = np.r_[xs, xs[0]]
    ys = np.r_[ys, ys[0]]
    contours = np.zeros((knots+1, 2))
    contours[:, 0] = xs[:]
    contours[:, 1] = ys[:]
    tck, u = interpolate.splprep(contours.T, u=None, k=1, per=1)
    u_new = np.linspace(u.min(), u.max(), n)
    xs_new, ys_new = interpolate.splev(u_new, tck, der=0)

    contours = np.zeros((len(xs_new), 2))
    contours[:, 0] = xs_new[:]
    contours[:, 1] = ys_new[:]
    contours = contours.reshape((-1, 1, 2)).astype(np.int32)

    # Using drawContours function we can plot our points(contours)
    cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("square", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        img = image.copy()

    # To get the internal energy matrix
    m = 1.0
    alpha = 0.08 * m
    beta = 0.5 * m
    gamma = 0.5
    kappa = 2.0 * m
    filter_size = 5
    num_points = len(xs_new)

    # get matrix
    M = get_matrix(alpha, beta, gamma, num_points)

    # Apply a gaussian filter on image to smooth and get better results
    smoothed = cv2.GaussianBlur(img, (filter_size, filter_size), filter_size)

    # get external energy
    w_line = 0.5
    w_edge = 1.0
    w_term = 0.8
    E = external_energy(smoothed, w_line, w_edge, w_term)

    # snake = np.zeros((n, 2))
    iterations = int(n * 2)
    for i in range(iterations):
        print(f'iteration number: {i}')

        # Boundary Checks
        xs_new[ xs_new < 0 ] = 0
        ys_new[ ys_new < 0 ] = 0
        xs_new[xs_new > img.shape[1]-2] = image.shape[1]-2
        ys_new[ys_new > img.shape[0]-2] = image.shape[0]-2

        # Define gradients for external energy in x and y direction
        fx = cv2.Sobel(E, cv2.CV_64F, 1, 0, ksize=1)
        fy = cv2.Sobel(E, cv2.CV_64F, 0, 1, ksize=1)
        cv2.imshow(f'External Energy: ', E)
        cv2.imshow(f'External Force x: ', fx)
        cv2.imshow(f'External Force y: ', fy)

        # Rounding / Bilinear Interpolation#
        bfx = np.zeros(n)
        bfy = np.zeros(n)

        for i in range(n):
            x_diff = xs_new[i] - math.floor(xs_new[i])
            prev_ptx = math.floor(xs_new[i])
            next_ptx = prev_ptx + 1
            y_diff = ys_new[i] - math.floor(ys_new[i])
            prev_pty = math.floor(ys_new[i])
            next_pty = prev_pty + 1

            p1x = ((1.0 - y_diff) * fx[(prev_pty, prev_ptx)]) + (y_diff * fx[(next_pty, prev_ptx)])
            p2x = ((1.0 - y_diff) * fx[(prev_pty, next_ptx)]) + (y_diff * fx[(next_pty, next_ptx)])
            bfx[i] = (((1.0 - x_diff) * p1x) + (x_diff * p2x))

            p1y = ((1.0 - y_diff) * fy[(prev_pty, prev_ptx)]) + (y_diff * fy[(next_pty, prev_ptx)])
            p2y = ((1.0 - y_diff) * fy[(prev_pty, next_ptx)]) + (y_diff * fy[(next_pty, next_ptx)])
            bfy[i] = (((1.0 - x_diff) * p1y) + (x_diff * p2y))

        x_ = np.dot(M, ((gamma * xs_new) - (kappa * bfx)))
        y_ = np.dot(M, ((gamma * ys_new) - (kappa * bfy)))

        # x_ = np.dot(M, ((gamma * xs_new) - (kappa * fx[(ys_new, xs_new)])))
        # y_ = np.dot(M, ((gamma * ys_new) - (kappa * fy[(ys_new, xs_new)])))

        # for v in range(n-1):
        #     if not np.all(snake[v]):
        #         if abs(x_[v] - xs_new[v]) <= 0.3 and abs(y_[v] - ys_new[v]) <= 0.3:
        #             print(f'Reached at v : {v}')
        #             snake[v] = [(xs_new[v].copy()).round(), (ys_new[v].copy()).round()]
        #
        # print(f' {len(snake)} : {np.count_nonzero((snake != 0).sum(1))} : {len(xs_new)}')
        # if np.count_nonzero((snake != 0).sum(1)) == len(xs_new)-1:
        #     break

        xs_new = x_.copy()
        ys_new = y_.copy()
        #
        # img = image.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # contours = np.zeros((len(xs_new), 2))
        # contours[:, 0] = xs_new[:]
        # contours[:, 1] = ys_new[:]
        # contours = contours.reshape((-1, 1, 2)).astype(np.int32)
        # cv2.drawContours(img, contours, -1, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        #
        # cv2.imshow("Iterative Contours", img)
        # cv2.imshow("Fx", fx)
        # cv2.imshow("Fy", fy)
        # cv2.waitKey(0)

    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    contours = np.zeros((len(xs_new), 2))
    contours[:, 0] = xs_new[:]
    contours[:, 1] = ys_new[:]
    contours = contours.reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(img, contours, -1, (25, 95, 255), thickness=3, lineType=cv2.LINE_AA)

    cv2.imshow(f'Iterative Contours Snake:', img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()