from PIL import Image
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from treesim import gen_simtree

image_path = 'basemodelado.jpeg'
image = Image.open(image_path).convert('L')
bw_image = image.convert('L')
bw_image.save('basemodelado_bw.jpeg')
bw_image.show()

bw_array = np.array(bw_image)
threshold = 100
target_points = np.column_stack(np.where(bw_array < threshold))
target_points = np.hstack((target_points, np.zeros((target_points.shape[0], 1))))
max_vals = np.max(target_points, axis=0)
max_vals[max_vals == 0] = 1
target_points_normalized = target_points / max_vals

def rotate_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotate_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotate_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def error_func(params, target_points, generated_points):
    scale, tx, ty, tz, theta_x, theta_y, theta_z = params
    transformed_points = generated_points * scale
    transformed_points = np.dot(transformed_points, rotate_x(theta_x))
    transformed_points = np.dot(transformed_points, rotate_y(theta_y))
    transformed_points = np.dot(transformed_points, rotate_z(theta_z))
    transformed_points[:, 0] += tx
    transformed_points[:, 1] += ty
    transformed_points[:, 2] += tz
    error = transformed_points - target_points
    return np.ravel(error)

generated_points = gen_simtree(Np=target_points.shape[0])
generated_points_xyz = generated_points[:, :3]
initial_params = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
result = least_squares(error_func, initial_params, args=(target_points, generated_points_xyz))
final_params = result.x

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue', s=1, label="Puntos objetivo")
ax.scatter(generated_points_xyz[:, 0], generated_points_xyz[:, 1], generated_points_xyz[:, 2], c='red', s=1, label="Puntos ajustados")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Visualización de Nube de Puntos Ajustada a Imagen")
ax.legend()
plt.show()
