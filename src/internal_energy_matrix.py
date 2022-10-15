import numpy as np
import cv2

def get_matrix(alpha, beta, gamma, num_points):
    """Return the matrix for the internal energy minimization.
    # Arguments
        alpha: The alpha parameter.
        beta: The beta parameter.
        gamma: The gamma parameter.
        num_points: The number of points in the curve.
    # Returns
        The matrix for the internal energy minimization. (i.e. A + gamma * I)
    """

    # Create matrix A which is NxN (N = no. of discrete points on snake)
    A = np.zeros((num_points, num_points))

    # Create the row which defines the derivatives, alpha and beta values
    A_row = A[0,:]

    A_row[2] = beta
    A_row[1] = - alpha - 4 * beta
    A_row[0] = 2 * alpha + 6 * beta
    A_row[num_points-1] = - alpha - 4 * beta
    A_row[num_points-2] = beta

    # Replace in matrix A with a right shift in every row
    for i in range(num_points):
        A[i] = np.roll(A_row, i)
    # A is now a pentadiagonal matrix

    # Now we need to create matrix M such that M = (A-(gamma*I)**-1
    # Create Identity matrix I first
    I = np.identity(num_points)
    # Now get gamma * I
    gI = gamma * I

    # Generate M and return it
    M = np.linalg.inv(A + gI)

    cv2.imshow(f'Internal Energy Matrix ({alpha}, {beta}, {gamma})', M)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return(M)