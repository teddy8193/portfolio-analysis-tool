import numpy as np
import colorsys


def generate_rand_color(
    seed: int,
    hue_range: tuple[float, float] = (0.0, 1.0),
    sat_range: tuple[float, float] = (0.0, 1.0),
    val_range: tuple[float, float] = (0.0, 1.0),
) -> str:
    hsv_min_max = np.array([hue_range, sat_range, val_range])
    hsv_range = np.abs(hsv_min_max[:, 1] - hsv_min_max[:, 0])
    rng = np.random.default_rng(seed=seed)
    rng_hsv = rng.random(3)
    rng_hsv = rng_hsv * hsv_range + hsv_min_max[:, 0]
    r, g, b = colorsys.hsv_to_rgb(rng_hsv[0] % 1, rng_hsv[1] % 1, rng_hsv[2] % 1)
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    return hex_color
