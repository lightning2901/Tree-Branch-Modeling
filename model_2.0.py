from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from treesim import gen_simtree

image_path = 'basemodelado.jpeg'
image = Image.open(image_path).convert('L')

bw_array = np.array(image)
smoothed_image = gaussian_filter(bw_array, sigma=1)

threshold = threshold_otsu(smoothed_image)
target_points = np.column_stack(np.where(smoothed_image < threshold))
target_points = np.hstack((target_points, np.zeros((target_points.shape[0], 1))))

max_vals = np.max(target_points, axis=0)
max_vals[max_vals == 0] = 1
target_points_normalized = target_points / max_vals

simulated_points = gen_simtree(Np=target_points_normalized.shape[0])
simulated_points_xyz = simulated_points[:, :3]

target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(target_points_normalized)

simulated_pcd = o3d.geometry.PointCloud()
simulated_pcd.points = o3d.utility.Vector3dVector(simulated_points_xyz)

threshold_icp = 0.02
icp_result = o3d.pipelines.registration.registration_icp(
    simulated_pcd, target_pcd, threshold_icp, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

transformation_matrix = icp_result.transformation
simulated_pcd.transform(transformation_matrix)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(target_points_normalized[:, 0], target_points_normalized[:, 1], target_points_normalized[:, 2],
           c='blue', s=1, label="Puntos objetivo")

simulated_transformed_points = np.asarray(simulated_pcd.points)
ax.scatter(simulated_transformed_points[:, 0], simulated_transformed_points[:, 1], simulated_transformed_points[:, 2],
           c='red', s=1, label="Puntos ajustados con ICP")
