def simulacion_ajustada(x, *params):
    height_min, height_max, diam_min, diam_max, split_prob = params[:5]

    min_can_height = height_min * 0.1
    max_can_width = (height_max - height_min) * 0.15
    max_can_width_height = [diam_min * 0.4, diam_max * 0.8]
    split_prob = split_prob * (height_max - height_min) * (diam_max - diam_min)

    split_height_range = [0.15, 0.5]
    num_branches = [60, 100]
    foliage_noise = 0.5

    model_params = {
        'height_range': [height_min, height_max],
        'diam_range': [diam_min, diam_max],
        'split_prob': split_prob,
        'split_height_range': split_height_range,
        'num_branches': num_branches,
        'min_can_height': min_can_height,
        'max_can_width': max_can_width,
        'max_can_width_height': max_can_width_height,
        'tree_top_dist': 2.5,
        'tree_mid_dist': 0.5,
        'foliage_noise': foliage_noise,
    }

    puntos = gen_simtree(model_params=model_params)
    result = np.full(x.shape, np.mean(puntos[:, 2]), dtype=np.float64)

    return result

params_iniciales = [30, 50, 0.5, 1, 0.5]

xdata = np.array([0], dtype=np.float64)
target_points_normalized = np.array([1.0], dtype=np.float64)

opt_params, _ = curve_fit(simulacion_ajustada, xdata, target_points_normalized, p0=params_iniciales)
print(f"Parámetros óptimos: {opt_params}")
