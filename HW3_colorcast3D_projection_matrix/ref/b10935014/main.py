import numpy as np
from PIL import Image

image = Image.open('Santa.jpg')
data = np.loadtxt('Santa.xyz')

image_points = np.array([
    [1185, 1299],
    [1675, 1293],
    [1668, 1507],
    [1189, 1507],
    [1414, 161],
    [1285, 880],
    [1594, 880],
    [1436, 1759],
    [1432, 716]
])

world_points = np.array([
    [10.871879, 13.849994, 20.553978, 1],
    [-10.168534, 13.716664, 20.602055, 1],
    [-9.942225, 15.374445, 11.166523, 1],
    [10.557071, 14.952327, 10.993542, 1],
    [0.298323, -13.327416, 68.322968, 1],
    [7.861877, -5.205806, 32.961178, 1],
    [-6.883900, 8.464606, 37.572487, 1],
    [-0.326691, 15.936172, 0.088914, 1],
    [-0.261571, 5.835245, 44.361362, 1],
])

# Setup matrix A for the equations Ax = 0
num_points = image_points.shape[0]
A = np.zeros((2 * num_points, 12))

for i in range(num_points):
    X, Y, Z, W = world_points[i]
    x, y = image_points[i]
    A[2 * i] = [X, Y, Z, W, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x * W]
    A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, W, -y * X, -y * Y, -y * Z, -y * W]

# Solve for P using Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)  # The last row of Vt reshaped to 3x4 gives the solution P


# 整個過程就是將3D模型點投影到2D圖像上，並將投影後的點的坐標轉換為普通的二維坐標，以便後續計算顏色和存儲結果。
ones = np.ones((data.shape[0], 1))
homogeneous_world_points = np.hstack((data[:, :3], ones))
projected_points = (P @ homogeneous_world_points.T).T
projected_points[:, :2] /= projected_points[:, [2]]


output_data = []
M = P[:, :3]  # The first 3x3 part of P
m_4 = P[:, 3]  # The last column of P

# Compute camera center C
C = -np.linalg.inv(M).dot(m_4)

for point, projected_point in zip(data, projected_points):
    x, y, z, nx, ny, nz = point
    px, py = projected_point[:2]

    # Vector from camera center to the point
    view_vector = np.array([x, y, z]) - C
    normal_vector = np.array([nx, ny, nz])

    # Check if the point is within the image bounds and if it is facing towards the camera
    if np.dot(view_vector, normal_vector) < 0:
        # Extract color from the image
        color = image.getpixel((int(px), int(py)))
        r, g, b = color[:3]
        a = 255
    else:
        r, g, b, a = (0, 0, 0, 0)  # Here I choose black and fully transparent for non-visible or out-of-bounds points

    output_data.append([x, y, z, nx, ny, nz, r, g, b, a])

np.savetxt('B10935014.txt', np.array(output_data), fmt='%f %f %f %f %f %f %d %d %d %d')
print("Result is saved!")
