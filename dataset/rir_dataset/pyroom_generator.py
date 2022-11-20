import numpy as np
import pyroomacoustics as pra
import random

def get_random_ir(sr = 16000, eps = 1e-5, maximum_order = 50, render_sec = 1.):

    x = 3 ** np.random.uniform(1, 3)
    y = 3 ** np.random.uniform(1, 3)
    z = 3 ** np.random.uniform(1, 3)
    
    room_dim = np.array([x, y, z])

    V = np.prod(room_dim)
    S1, S2, S3 = x * y, y * z, z * x

    materials = random_materials2(sr = sr)

    S_freqs = (materials['ceiling'].absorption_coeffs + materials['floor'].absorption_coeffs) * S1 \
               + (materials['east'].absorption_coeffs + materials['west'].absorption_coeffs) * S2 \
               + (materials['north'].absorption_coeffs + materials['south'].absorption_coeffs) * S3

    S_min = np.min(S_freqs)

    RT60 = .16 * V / S_min
    RT30 = min(render_sec, RT60 * .5)

    temperature = np.random.uniform(-40, 45)
    humidity = np.random.uniform(0, 100)

    c = 331.4 + 0.6 * temperature + 0.0124 * humidity

    max_dist = RT30 * c
    max_order_point = max_dist / np.sqrt((1 / x) ** 2 + (1 / y) ** 2 + (1 / z) ** 2) * np.array([1 / x, 1 / y, 1 / z])
    
    max_order = int(max_order_point[0] / x) + int(max_order_point[1] / y) + int(max_order_point[2] / z)
    max_order = min(max_order, maximum_order)

    room = pra.ShoeBox(
        room_dim,
        fs = sr,
        materials = materials,
        max_order = max_order,
        air_absorption = True,
        temperature = temperature,
        humidity = humidity,
        sigma2_awgn = 200
    )


    room.add_source([np.random.uniform(eps, x - eps),
                     np.random.uniform(eps, y - eps),
                     np.random.uniform(eps, z - eps)],
                    signal = np.array([1]),
                    delay = 0)

    mic_locs = np.c_[[np.random.uniform(eps, x - eps),
                      np.random.uniform(eps, y - eps),
                      np.random.uniform(eps, z - eps)]]

    room.add_microphone_array(mic_locs)
    room.compute_rir()

    return room.rir[0][0]

def random_materials():
    materials = {}
    center_freqs, num_bands = [50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170,
                               1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000,
                               4800, 5800, 7000], 21

    num_unique_materials = random.randrange(1, 7)
    center_decay = np.random.uniform(0.01, 0.6)
    allowed_deviation = np.random.uniform(0.05, 20)

    wall_materials = []
    for i in range(num_unique_materials):

        wall_materials.append(random_absorption(center_decay, allowed_deviation, num_bands))

    while len(wall_materials) < 6:
        wall_materials.append(wall_materials[-1])
        random.shuffle(wall_materials)

    i = 0
    for wall in ['ceiling', 'floor', 'east', 'west', 'north', 'south']:
        materials[wall] = (pra.parameters.Material(dict(description = wall,
                                                     coeffs = wall_materials[i],
                                                     center_freqs = center_freqs)))
        i += 1
    return materials

def random_materials2(sr = 16000, freq_dev = .3):
    materials = {}

    center_freqs = [50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170,
                    1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000,
                    4800, 5800, 7000]

    dev = np.exp(np.random.uniform(-1, 1, size = (len(center_freqs),)) * freq_dev)
    center_freqs = center_freqs * dev
    center_freqs = np.sort(center_freqs)
    center_freqs = center_freqs[center_freqs < sr / 2]

    num_bands = len(center_freqs)

    num_unique_materials = random.randrange(1, 7)
    center_decay = 0.1 ** np.random.uniform(0, 2) #np.random.uniform(0.01, 0.7)
    allowed_deviation = 0.05 ** np.random.uniform(-1, 1) #np.random.uniform(0.05, 20)

    wall_materials = []
    for i in range(num_unique_materials):

        wall_materials.append(random_absorption(center_decay, allowed_deviation, num_bands))

    while len(wall_materials) < 6:
        wall_materials.append(wall_materials[-1])
        random.shuffle(wall_materials)

    i = 0
    for wall in ['ceiling', 'floor', 'east', 'west', 'north', 'south']:
        materials[wall] = (pra.parameters.Material(dict(description = wall,
                                                     coeffs = wall_materials[i],
                                                     center_freqs = center_freqs)))
        i += 1
    return materials

def random_absorption(center_decay, allowed_deviation, num_bands):
    exps = np.random.uniform(-1, 1, size = (num_bands,))
    smoothing = random.randrange(1, 2)
    smoother = np.hanning(smoothing * 2 + 1)
    exps = np.convolve(exps, smoother)[smoothing:-smoothing]
    absorptions = center_decay * allowed_deviation ** exps
    darker = 1.1 ** np.linspace(0, 1, num_bands,) ## ?
    absorptions = absorptions * darker ## ?
    absorptions = np.minimum(np.maximum(absorptions, 0.001), 1)
    return absorptions

"""
from tqdm import tqdm

for i in tqdm(range(100)):
    print(get_random_ir().shape)
"""
