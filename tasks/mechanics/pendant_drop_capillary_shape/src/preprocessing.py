# Adapted from OpenDrop, copyright2020 Joseph Berry/Rico Tabor; GPL3. See LICENSE and test/validation.md.
import math
from pathlib import Path
import numpy as np
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:image=z['image'].copy();contour=z['contour'].copy()
    if image.shape!=(1,1024,1280) or image.dtype!=np.uint8:raise ValueError('Invalid camera image')
    if contour.shape!=(1,293,2) or contour.dtype.kind!='f' or not np.isfinite(contour).all():raise ValueError('Invalid observed contour')
    return image[0],contour[0]
def prepare(contour):
    out=np.asarray(contour,dtype=float).copy();out[:,1]*=-1
    return out[out[:,1].argsort()]
def bond_number(contour, apex_x, apex_y, apex_radius) -> float:
    r_z2 = scaled_radius_at_scaled_height(contour, apex_x, apex_y, apex_radius, 2)
    if r_z2 > 0:
        bond = 0.1756 * r_z2 ** 2 + 0.5234 * r_z2 ** 3 - 0.2563 * r_z2 ** 4
        return bond
    r_z1 = scaled_radius_at_scaled_height(contour, apex_x, apex_y, apex_radius, 1)
    return 0.15

def fit_circle(contour):
    sumX = np.double(0.0)
    sumY = np.double(0.0)
    sumX2 = np.double(0.0)
    sumY2 = np.double(0.0)
    sumXY = np.double(0.0)
    sumX3 = np.double(0.0)
    sumY3 = np.double(0.0)
    sumX2Y = np.double(0.0)
    sumXY2 = np.double(0.0)
    n = min(max(int(0.1 * len(contour)), 10), len(contour))
    for k in range(0, n):
        xk, yk = contour[k]
        sumX += xk
        sumY += yk
        sumX2 += xk ** 2
        sumY2 += yk ** 2
        sumXY += xk * yk
        sumX3 += xk ** 3
        sumY3 += yk ** 3
        sumX2Y += xk ** 2 * yk
        sumXY2 += xk * yk ** 2
    d11 = n * sumXY - sumX * sumY
    d20 = n * sumX2 - sumX ** 2
    d02 = n * sumY2 - sumY ** 2
    d30 = n * sumX3 - sumX2 * sumX
    d03 = n * sumY3 - sumY2 * sumY
    d21 = n * sumX2Y - sumX2 * sumY
    d12 = n * sumXY2 - sumX * sumY2
    x = (0.5 * (d30 + d12) * d02 - 0.5 * (d03 + d21) * d11) / (d20 * d02 - d11 ** 2)
    y = (0.5 * (d03 + d21) * d20 - 0.5 * (d30 + d12) * d11) / (d20 * d02 - d11 ** 2)
    c = (sumX2 + sumY2 - 2 * x * sumX - 2 * y * sumY) / n
    R = math.sqrt(c + x ** 2 + y ** 2)
    return [x, y - R, R]

def scaled_radius_at_scaled_height(contour, apex_x, apex_y, apex_radius, height):
    num_points = len(contour)
    if num_points == 0:
        return math.nan
    points_to_return = 5
    z_value = apex_y + height * apex_radius
    if contour[-1][1] < z_value:
        return math.nan
    index = 0
    while contour[index][1] < z_value:
        index += 1
    if index < points_to_return or num_points - index < points_to_return:
        return math.nan
    sum_radius = 0.0
    for k in range(index - points_to_return, index + points_to_return):
        sum_radius += abs(contour[k][0] - apex_x)
    scaled_radius = sum_radius / (2 * points_to_return * apex_radius)
    return scaled_radius
