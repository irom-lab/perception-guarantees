import trimesh
import math
import numpy as np


moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def add_mat(input_mesh, output_mesh, mat_name):
    use_mat = "mtllib {}.mtl\nusemtl default\n"
    with open(input_mesh, "r") as fin:
        lines = fin.readlines()
    for l in lines:
        if l == "mtllib {}.mtl\n".format(mat_name):
            return
    with open(output_mesh, "w") as fout:
        fout.write(use_mat.format(mat_name))
        for line in lines:
            if not line.startswith("o") and not line.startswith("s"):
                fout.write(line)


def remove_np_duplicates(data):
    # Perform lex sort and get sorted data
    sorted_idx = np.lexsort(data.T)
    sorted_data = data[sorted_idx, :]

    # Get unique row mask
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))

    # Get unique rows
    out = sorted_data[row_mask]
    return out


def get_grid_cells_btw(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0:  # will divide by dx later, this will cause err. Catch this case up here
        step = np.sign(dy)
        ys = np.arange(0, dy + step, step)
        xs = np.repeat(x1, ys.shape[0])
    else:
        m = dy / (dx+0.0)
        b = y1 - m*x1

        step = 1.0 / (max(abs(dx), abs(dy)))
        xs = np.arange(x1, x2, step * np.sign(x2 - x1))
        ys = xs*m + b

    xs = np.rint(xs)
    ys = np.rint(ys)
    pts = np.column_stack((xs, ys))
    pts = remove_np_duplicates(pts)

    return pts.astype(int)


def state_lin_to_bin(state_lin, grid_shape):
    return np.unravel_index(state_lin, grid_shape)


def state_bin_to_lin(state_coord, grid_shape):
    return np.ravel_multi_index(state_coord, grid_shape)


def apply_move(coord_in, move):
    coord = coord_in.copy()
    coord[:2] += move[:2]
    return coord


def check_free(grid, coord):
    N, M = grid.shape
    return (
        coord[0] >= 0 and coord[0] < N and coord[1] >= 0 and coord[1] < M
        and not grid[coord[0], coord[1]]
    )


def get_neighbor(grid, coord, radius):
    # radius excludes coord itself
    min_x = max(0, coord[0] - radius)
    max_x = min(grid.shape[0], coord[0] + radius + 1)
    min_y = max(0, coord[1] - radius)
    max_y = min(grid.shape[1], coord[1] + radius + 1)
    return grid[min_x:max_x, min_y:max_y]


def slice_mesh(mesh):
    mesh_below = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0, 0, -1], plane_origin=[0, 0, 2]
    )
    mesh_below = trimesh.intersections.slice_mesh_plane(
        mesh_below, plane_normal=[0, 0, 1], plane_origin=[0, 0, 0.05]
    )
    return mesh_below


def dist(a, b):
    return math.sqrt((b[1] - a[1])**2 + (b[0] - a[0])**2)


def rect_distance(a, b):
    """https://stackoverflow.com/questions/4978323/how-to-calculate-distance-between-two-rectangles-context-a-game-in-lua

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [description]
    """
    (x1, y1, x1b, y1b) = a
    (x2, y2, x2b, y2b) = b
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0.
