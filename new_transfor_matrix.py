threshold_icp = 0.02
icp_result = o3d.pipelines.registration.registration_icp(
    simulated_pcd, target_pcd, threshold_icp, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

transformation_matrix = icp_result.transformation

simulated_pcd.transform(transformation_matrix)
