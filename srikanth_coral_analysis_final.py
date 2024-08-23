import numpy as np
import pyvista as pv
import os
import time
from tqdm import tqdm
import geopandas as gpd
import multiprocessing as mp
import csv 
import scipy.spatial

class AABB:
    def __init__(self, min_bound, max_bound):
        self.min_bound = min_bound
        self.max_bound = max_bound

    def intersect(self, ray_origin, ray_direction):
        t_min = np.zeros(3)
        t_max = np.zeros(3)
        for i in range(3):
            if ray_direction[i] != 0:
                t_min[i] = (self.min_bound[i] - ray_origin[i]) / ray_direction[i]
                t_max[i] = (self.max_bound[i] - ray_origin[i]) / ray_direction[i]
                if t_min[i] > t_max[i]:
                    t_min[i], t_max[i] = t_max[i], t_min[i]
            else:
                t_min[i] = float('-inf') if ray_origin[i] >= self.min_bound[i] else float('inf')
                t_max[i] = float('inf') if ray_origin[i] <= self.max_bound[i] else float('-inf')
        t_enter = max(t_min)
        t_exit = min(t_max)
        return t_enter <= t_exit and t_exit > 0

class BVHNode:
    def __init__(self, start, end, aabb):
        self.start = start
        self.end = end
        self.aabb = aabb
        self.left = None
        self.right = None

def build_bvh(triangles, indices, start, end, depth=0, max_depth=20):
    if start >= end or depth > max_depth:
        return None

    aabb_min = np.min(triangles[indices[start:end]], axis=(0, 1))
    aabb_max = np.max(triangles[indices[start:end]], axis=(0, 1))
    node = BVHNode(start, end, AABB(aabb_min, aabb_max))

    if end - start <= 4 or depth == max_depth:  # Leaf node
        return node

    # Choose longest axis to split
    axis = np.argmax(aabb_max - aabb_min)
    mid = (start + end) // 2
    
    # Sort indices based on triangle centroids
    centroids = np.mean(triangles[indices[start:end]], axis=1)
    sorted_indices = indices[start:end][np.argsort(centroids[:, axis])]
    
    # Update the original indices array
    indices[start:end] = sorted_indices

    node.left = build_bvh(triangles, indices, start, mid, depth + 1, max_depth)
    node.right = build_bvh(triangles, indices, mid, end, depth + 1, max_depth)
    return node

def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2):
    epsilon = 1e-6
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if abs(a) < epsilon:
        return False
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(edge2, q)
    return t > epsilon

def intersect_bvh(node, ray_origin, ray_direction, triangles, indices):
    if node is None or not node.aabb.intersect(ray_origin, ray_direction):
        return False

    if node.left is None and node.right is None:
        for i in range(node.start, node.end):
            triangle = triangles[indices[i]]
            if ray_triangle_intersect(ray_origin, ray_direction, triangle[0], triangle[1], triangle[2]):
                return True
        return False

    return intersect_bvh(node.left, ray_origin, ray_direction, triangles, indices) or \
           intersect_bvh(node.right, ray_origin, ray_direction, triangles, indices)

def process_chunk(args):
    chunk, bvh_root, triangles, indices, light_dir = args
    shadowed = np.zeros(len(chunk), dtype=bool)
    for i, point in enumerate(chunk):
        if intersect_bvh(bvh_root, point, light_dir, triangles, indices):
            shadowed[i] = True
    return shadowed

def point_in_box(point, box_min, box_max):
    return np.all(point >= box_min) and np.all(point <= box_max)

def triangle_intersects_box(v0, v1, v2, box_min, box_max):
    # Check if any vertex is inside the box
    if any(point_in_box(v, box_min, box_max) for v in [v0, v1, v2]):
        return True
    
    # Check if the triangle intersects any of the 12 edges of the box
    edges = [
        (box_min, [box_max[0], box_min[1], box_min[2]]),
        (box_min, [box_min[0], box_max[1], box_min[2]]),
        (box_min, [box_min[0], box_min[1], box_max[2]]),
        (box_max, [box_min[0], box_max[1], box_max[2]]),
        (box_max, [box_max[0], box_min[1], box_max[2]]),
        (box_max, [box_max[0], box_max[1], box_min[2]]),
        ([box_min[0], box_max[1], box_min[2]], [box_max[0], box_max[1], box_min[2]]),
        ([box_min[0], box_max[1], box_min[2]], [box_min[0], box_max[1], box_max[2]]),
        ([box_min[0], box_min[1], box_max[2]], [box_max[0], box_min[1], box_max[2]]),
        ([box_min[0], box_min[1], box_max[2]], [box_min[0], box_max[1], box_max[2]]),
        ([box_max[0], box_min[1], box_min[2]], [box_max[0], box_max[1], box_min[2]]),
        ([box_max[0], box_min[1], box_min[2]], [box_max[0], box_min[1], box_max[2]])
    ]
    
    for edge_start, edge_end in edges:
        if ray_triangle_intersect(edge_start, np.array(edge_end) - np.array(edge_start), v0, v1, v2):
            return True
    
    return False

def interactive_bounding_box(mesh):
    print("Interactive Bounding Box Selection")
    print("1. Click on the mesh to select the first corner of the bounding box.")
    print("2. Click again to select the opposite corner.")
    print("3. Or press 'm' to manually enter coordinates and diameter.")
    print("4. Press 'q' to finish selection and proceed with calculations.")

    print("Creating plotter...")
    plotter = pv.Plotter()
    
    print("Preparing mesh for visualization...")
    
    # Check if the mesh has color data
    if 'RGB' in mesh.array_names:
        print("Using original color data...")
        color_data = mesh['RGB']
        # Normalize RGB values to 0-1 range if they're in 0-255 range
        if color_data.max() > 1:
            color_data = color_data / 255.0
        mesh['colors'] = color_data
        print("Adding mesh to plotter with original colors...")
        plotter.add_mesh(mesh, scalars='colors', rgb=True, opacity=1.0, show_edges=False, smooth_shading=True)
    else:
        print("No color data found. Displaying mesh without color...")
        plotter.add_mesh(mesh, opacity=1.0, show_edges=False, smooth_shading=True)

    # Reset camera to focus on the mesh
    plotter.reset_camera()

    points = []
    box_actor = None

    def callback(pos):
        nonlocal box_actor
        print(f"Point selected at: {pos}")
        points.append(pos)
        if len(points) == 1:
            print("Adding first point...")
            plotter.add_mesh(pv.PolyData(points[0]), color='red', point_size=10)
        elif len(points) == 2:
            print("Adding bounding box...")
            if box_actor:
                plotter.remove_actor(box_actor)
            
            bounds = [min(points[0][0], points[1][0]), max(points[0][0], points[1][0]),
                      min(points[0][1], points[1][1]), max(points[0][1], points[1][1]),
                      min(points[0][2], points[1][2]), max(points[0][2], points[1][2])]
            box = pv.Box(bounds)
            box_actor = plotter.add_mesh(box, color='red', style='wireframe', line_width=5)

    def manual_input():
        nonlocal box_actor, points
        x = float(input("Enter x coordinate: "))
        y = float(input("Enter y coordinate: "))
        z = float(input("Enter z coordinate: "))
        diameter = float(input("Enter diameter: "))
        
        center = np.array([x, y, z])
        half_size = diameter / 2
        
        points = [center - half_size, center + half_size]
        
        if box_actor:
            plotter.remove_actor(box_actor)
        
        bounds = [x - half_size, x + half_size,
                  y - half_size, y + half_size,
                  z - half_size, z + half_size]
        box = pv.Box(bounds)
        box_actor = plotter.add_mesh(box, color='red', style='wireframe', line_width=5)
        plotter.render()

    print("Enabling point picking...")
    plotter.enable_point_picking(callback=callback, show_message=False)
    plotter.add_key_event('m', manual_input)
    plotter.add_text("Select two points or press 'm' for manual input. Press 'q' to finish.", position='upper_left')
    
    print("Showing plotter...")
    plotter.show()

    if len(points) != 2:
        raise ValueError("Two points must be selected to define the bounding box.")

    point_of_interest = np.mean(points, axis=0)
    window_size = np.abs(points[1] - points[0])

    print(f"Bounding box selected: center={point_of_interest}, size={window_size}")
    return point_of_interest, window_size

def triangle_intersects_box_wrapper(triangle, box_min, box_max):
    return triangle_intersects_box(triangle[0], triangle[1], triangle[2], box_min, box_max)

def filter_triangles_with_bvh(bvh_root, box_min, box_max, triangles, indices):
    filtered_indices = []
    
    def traverse(node):
        if node is None:
            return
        
        # Check if the node's AABB intersects with the bounding box
        if not (np.any(node.aabb.max_bound < box_min) or np.any(node.aabb.min_bound > box_max)):
            if node.left is None and node.right is None:
                # Leaf node: check individual triangles
                for i in range(node.start, node.end):
                    triangle = triangles[indices[i]]
                    if triangle_intersects_box(triangle[0], triangle[1], triangle[2], box_min, box_max):
                        filtered_indices.append(indices[i])
            else:
                # Internal node: recurse
                traverse(node.left)
                traverse(node.right)
    
    traverse(bvh_root)
    return filtered_indices

def parallel_triangle_filtering_with_bvh(triangles, box_min, box_max, cpu_limit):
    num_processes = min(mp.cpu_count(), cpu_limit) if cpu_limit else mp.cpu_count()
    
    print("Building BVH...")
    build_start = time.time()
    indices = np.arange(len(triangles), dtype=np.int32)  # Ensure integer type
    bvh_root = build_bvh(triangles, indices, 0, len(indices))
    build_time = time.time() - build_start
    print(f"BVH built in {build_time:.2f} seconds.")
    
    print("Filtering triangles...")
    filter_start = time.time()
    filtered_indices = filter_triangles_with_bvh(bvh_root, box_min, box_max, triangles, indices)
    filter_time = time.time() - filter_start
    print(f"Filtering completed in {filter_time:.2f} seconds.")
    print(f"Filtered {len(filtered_indices)} triangles out of {len(triangles)}")
    
    return np.array(filtered_indices, dtype=np.int32)

def calculate_structure_shading(mesh, light_dir, point_of_interest, window_size, sample_size=1000000, cpu_limit=None):
    print("Preparing mesh data...")
    with tqdm(total=3, desc="Mesh preparation") as pbar:
        mesh.compute_normals(inplace=True)
        pbar.update(1)
        points = mesh.points
        pbar.update(1)
        faces = mesh.faces.reshape(-1, 4)[:, 1:4]
        pbar.update(1)
    triangles = points[faces]
    
    # Convert window_size to numpy array if it's not already
    window_size = np.array(window_size)
    
    # Bounding box calculation
    box_min = point_of_interest - window_size / 2
    box_max = point_of_interest + window_size / 2
    
    print("Filtering points...")
    mask = np.all((points >= box_min) & (points <= box_max), axis=1)
    window_points = points[mask]
    
    if len(window_points) > sample_size:
        print(f"Sampling {sample_size} points from {len(window_points)} points in the window...")
        sampled_indices = np.random.choice(len(window_points), sample_size, replace=False)
        sampled_points = window_points[sampled_indices]
    else:
        sampled_points = window_points
    
    print("Beginning to filter triangles with BVH method...")
    indices = parallel_triangle_filtering_with_bvh(triangles, box_min, box_max, cpu_limit)

    print("Building BVH...")
    build_start = time.time()
    with tqdm(total=len(indices), desc="Building BVH") as pbar:
        def bvh_progress_callback(progress):
            pbar.n = progress
            pbar.refresh()
        bvh_root = build_bvh(triangles, indices, 0, len(indices))
    print(f"BVH built in {time.time() - build_start:.2f} seconds.")

    if cpu_limit is None:
        num_processes = mp.cpu_count()
    else:
        num_processes = max(1, min(cpu_limit, mp.cpu_count()))

    chunk_size = max(1, len(sampled_points) // (num_processes * 2))
    chunks = [sampled_points[i:i + chunk_size] for i in range(0, len(sampled_points), chunk_size)]

    print(f"Using {num_processes} CPU cores to process {len(chunks)} chunks...")

    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, [(chunk, bvh_root, triangles, indices, light_dir) for chunk in chunks]),
            total=len(chunks),
            desc="Processing chunks",
            mininterval=0.1,
            smoothing=0.1
        ))

    print("All chunks processed. Calculating final result...")
    shadowed = np.concatenate(results)
    shaded_percentage = np.mean(shadowed) * 100
    return shaded_percentage

def write_output(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Check if we're dealing with polygon results
        if 'polygon_results' in results[0]:
            writer.writerow(['Plot Name', 'Polygon ID', 'Center X', 'Center Y', 'Center Z', 'Shaded Percentage', 'Illuminated Percentage'])
            for result in results:
                for polygon in result['polygon_results']:
                    writer.writerow([
                        result['plot_name'],
                        polygon['polygon_id'],
                        f"{polygon['center_x']:.6f}",
                        f"{polygon['center_y']:.6f}",
                        f"{polygon['center_z']:.6f}",
                        f"{polygon['shaded_percentage']:.2f}",
                        f"{polygon['illuminated_percentage']:.2f}"
                    ])
        else:
            writer.writerow(['Plot Name', 'Shaded Percentage', 'Illuminated Percentage', 'Calculation Time (s)'])
            for result in results:
                writer.writerow([
                    result['plot_name'], 
                    f"{result['shaded_percentage']:.2f}", 
                    f"{result['illuminated_percentage']:.2f}", 
                    f"{result['calculation_time']:.2f}"
                ])
            
def visualize_mesh_with_polygons(mesh, projected_polygons, expansion_percentage=0, opacity=1.0):
    print("Displaying 3D model with projected polygons and expanded bounding boxes")
    print("1. Use the mouse to rotate, zoom, and pan the view.")
    print("2. Press 'q' to close the window and continue processing.")

    plotter = pv.Plotter()
    
    # Check if the mesh has color data
    if 'RGB' in mesh.array_names:
        print("Using original color data...")
        color_data = mesh['RGB']
        # Normalize RGB values to 0-1 range if they're in 0-255 range
        if color_data.max() > 1:
            color_data = color_data / 255.0
        mesh['colors'] = color_data
        print(f"Adding mesh to plotter with original colors and opacity {opacity}...")
        plotter.add_mesh(mesh, scalars='colors', rgb=True, opacity=opacity, show_edges=False, smooth_shading=True)
    else:
        print(f"No color data found. Displaying mesh with default color and opacity {opacity}...")
        plotter.add_mesh(mesh, opacity=opacity, show_edges=False, smooth_shading=True)

    # Add projected polygons and their bounding boxes
    for poly_id, points in projected_polygons:
        poly_points = np.array(points)
        if len(poly_points) > 1:  # Ensure we have at least two points to create a line
            # Create a closed loop by adding the first point at the end
            closed_poly_points = np.vstack((poly_points, poly_points[0]))
            poly_line = pv.PolyData(closed_poly_points)
            
            # Create line connectivity
            line = np.arange(len(closed_poly_points))
            line = np.insert(line, 0, len(closed_poly_points))  # Insert the length at the beginning
            poly_line.lines = line
            
            plotter.add_mesh(poly_line, color='red', line_width=2, render_lines_as_tubes=True, label='Polygon')

            box_min, box_max = calculate_expanded_bounding_box(poly_points, expansion_percentage)
            box = pv.Box(bounds=(box_min[0], box_max[0], box_min[1], box_max[1], box_min[2], box_max[2]))
            plotter.add_mesh(box, color='blue', style='wireframe', opacity=0.5, line_width=2, label='Expanded Bounding Box')

    # Add a legend
    plotter.add_legend()

    plotter.reset_camera()

    print("Showing plotter...")
    plotter.add_text("Press 'q' to close the window and continue processing.", position='upper_left')
    plotter.show()

def calculate_expanded_bounding_box(points, expansion_percentage):
    """Calculate an expanded bounding box for a set of points."""
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Calculate the current size and center
    size = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    
    # Calculate the expanded size
    expanded_size = size * (1 + expansion_percentage / 100)
    
    # Calculate the new min and max coordinates
    expanded_min = center - expanded_size / 2
    expanded_max = center + expanded_size / 2
    
    return expanded_min, expanded_max

def calculate_expanded_bounding_box(points, expansion_percentage):
    """calculate an expanded bounding box for a set of points."""
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Calculate the current size and center
    size = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    
    # Calculate the expanded size
    expanded_size = size * (1 + expansion_percentage / 100)
    
    # Calculate the new min and max coordinates
    expanded_min = center - expanded_size / 2
    expanded_max = center + expanded_size / 2
    
    return expanded_min, expanded_max

def process_single_plot(plot_info, use_bounding_box, cpu_limit, shapefile_path=None, expansion_percentage=0):
    mesh_path, plot_name = plot_info
    
    if not os.path.exists(mesh_path):
        print(f"3D model file not found: {mesh_path}")
        return None

    print(f"Processing plot: {plot_name}")
    print("Loading 3D mesh...")
    start_time = time.time()
    try:
        mesh = pv.read(mesh_path)
    except Exception as e:
        print(f"Failed to load 3D model: {e}")
        return None

    print(f"Mesh loaded in {time.time() - start_time:.2f} seconds")
    print(f"Number of points: {mesh.n_points}")
    print(f"Number of faces: {mesh.n_cells}")

    light_dir = np.array([0, 0, -1])  # Negative z-direction for top-down light you can change this if you want

    if use_bounding_box:
        point_of_interest, window_size = interactive_bounding_box(mesh)
        print(f"Selected point of interest: {point_of_interest}")
        print(f"Selected window size: {window_size}")
        shaded_percentage = calculate_structure_shading(mesh, light_dir, point_of_interest, window_size, cpu_limit=cpu_limit)
        return {
            'plot_name': plot_name,
            'shaded_percentage': shaded_percentage,
            'illuminated_percentage': 100 - shaded_percentage,
            'calculation_time': time.time() - start_time
        }
    elif shapefile_path and os.path.exists(shapefile_path):
        print("\nLoading and projecting polygons...")
        projected_polygons = load_and_project_polygons(shapefile_path, mesh)
        visualize_mesh_with_polygons(mesh, projected_polygons, expansion_percentage)
        
        print("Calculating shading for each polygon...")
        polygon_results = []
        for poly_id, points in projected_polygons:
            print(f"Processing polygon {poly_id}")
            
            # Calculate the expanded bounding box
            box_min, box_max = calculate_expanded_bounding_box(points, expansion_percentage)
            center = (box_min + box_max) / 2
            window_size = box_max - box_min
            
            shaded_percentage = calculate_structure_shading(mesh, light_dir, center, window_size, cpu_limit=cpu_limit)
            polygon_results.append({
                'polygon_id': poly_id,
                'center_x': center[0],
                'center_y': center[1],
                'center_z': center[2],
                'box_size_x': window_size[0],
                'box_size_y': window_size[1],
                'box_size_z': window_size[2],
                'shaded_percentage': shaded_percentage,
                'illuminated_percentage': 100 - shaded_percentage
            })
            print(f'  Polygon {poly_id}: Shaded {shaded_percentage:.2f}%, Illuminated {100 - shaded_percentage:.2f}%')
        
        avg_shaded = np.mean([r['shaded_percentage'] for r in polygon_results])
        print(f'\nAverage Shaded Percentage: {avg_shaded:.2f}%')
        print(f'Average Illuminated Percentage: {100 - avg_shaded:.2f}%')
        
        return {
            'plot_name': plot_name,
            'polygon_results': polygon_results,
            'shaded_percentage': avg_shaded,
            'illuminated_percentage': 100 - avg_shaded,
            'calculation_time': time.time() - start_time
        }
    else:
        print("No valid bounding box or shapefile provided. Processing entire mesh...")
        point_of_interest = np.mean(mesh.points, axis=0)
        window_size = np.ptp(mesh.points, axis=0)  # Range of points in each dimension

    print("Calculating shading percentage based on coral structure...")
    calculation_start_time = time.time()
    shaded_percentage = calculate_structure_shading(mesh, light_dir, point_of_interest, window_size, cpu_limit=cpu_limit)
    calculation_time = time.time() - calculation_start_time

    print(f'Shaded Percentage: {shaded_percentage:.2f}%')
    print(f'Illuminated Percentage: {100 - shaded_percentage:.2f}%')
    print(f'Calculation Time: {calculation_time:.2f} seconds')

    return {
        'plot_name': plot_name,
        'shaded_percentage': shaded_percentage,
        'illuminated_percentage': 100 - shaded_percentage,
        'calculation_time': calculation_time
    }

def collect_bounding_boxes(plots):
    bounding_boxes = {}
    for mesh_path, plot_name in plots:
        print(f"\nCollecting bounding box for plot: {plot_name}")
        mesh = pv.read(mesh_path)
        point_of_interest, window_size = interactive_bounding_box(mesh)
        bounding_boxes[plot_name] = (point_of_interest, window_size)
        print(f"Bounding box for {plot_name}: POI={point_of_interest}, Size={window_size}")
    return bounding_boxes

def load_and_project_polygons(shapefile_path, mesh):
    print(f"Loading shapefile from: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    print(f"Shapefile CRS: {gdf.crs}")
    print(f"Number of polygons: {len(gdf)}")
    
    if gdf.crs is None:
        print("Warning: Shapefile has no defined CRS. Assuming it matches the mesh coordinates.")
    
    # Create a KDTree for efficient nearest neighbor search
    points = mesh.points
    kdtree = scipy.spatial.cKDTree(points[:, :2])  # Only use x and y for initial matching
    
    projected_polygons = []
    for _, row in gdf.iterrows():
        # We access the 'TL_id' column's value for this row using the column name as a key cuz this was what was in the shp file
        poly_id = row['TL_id']  
        polygon = row['geometry']
        projected_points = []
        for point in polygon.exterior.coords:
            x, y = point[:2]  # Extract x and y coordinates
            
            # Find the nearest point on the mesh surface
            _, index = kdtree.query([x, y])
            closest_point = points[index]
            
            projected_points.append(closest_point)
        projected_polygons.append((poly_id, projected_points))
    
    print(f"Projected {len(projected_polygons)} polygons onto the mesh")
    return projected_polygons


def extract_center_points(projected_polygons):
    center_points = []
    for poly_id, points in projected_polygons:
        center = np.mean(points, axis=0)
        center_points.append((poly_id, center[0], center[1], center[2]))
    return center_points

def create_bounding_box(center, size):
    half_size = size / 2
    return (
        center - [half_size, half_size, half_size],
        center + [half_size, half_size, half_size]
    )

def calculate_polygon_shading(mesh, light_dir, center_points, box_size, cpu_limit):
    results = []
    for poly_id, x, y, z in tqdm(center_points, desc="Processing polygons"):
        center = np.array([x, y, z])
        window_size = np.array([box_size, box_size, box_size])
        
        shaded_percentage = calculate_structure_shading(
            mesh, light_dir, center, window_size, cpu_limit=cpu_limit
        )
        
        results.append({
            'polygon_id': poly_id,
            'x': x,
            'y': y,
            'z': z,
            'shaded_percentage': shaded_percentage,
            'illuminated_percentage': 100 - shaded_percentage
        })
    
    return results


def main(use_bounding_box=False, use_shapefile=True, cpu_limit=5, expansion_percentage=20):
    # Directory containing the 3D model files
    model_directory = '/Users/srikanthsamy1/Desktop/AIMS/testply'
    
    # Directory containing the shapefile files (if using shapefile analysis)
    shapefile_directory = '/Users/srikanthsamy1/Desktop/AIMS/testshp'

    # Get all .ply files in the model directory
    mesh_files = [f for f in os.listdir(model_directory) if f.endswith('.ply')]

    results = []
    for mesh_file in mesh_files:
        mesh_path = os.path.join(model_directory, mesh_file)
        plot_name = os.path.splitext(mesh_file)[0]  # Use filename without extension as plot name

        if not os.path.exists(mesh_path):
            print(f"3D model file not found: {mesh_path}")
            continue

        shapefile_path = None
        if use_shapefile:
            # assuming shapefile has the same name as the mesh file but with .shp extension
            shapefile_name = f"{plot_name}.shp"
            shapefile_path = os.path.join(shapefile_directory, shapefile_name)
            if not os.path.exists(shapefile_path):
                print(f"Shapefile not found: {shapefile_path}")
                shapefile_path = None

        result = process_single_plot((mesh_path, plot_name), use_bounding_box, cpu_limit, shapefile_path, expansion_percentage)
        if result:
            results.append(result)

        print("\n" + "="*50 + "\n")

    output_file = 'shading_results.csv'
    write_output(results, output_file)
    print(f"Results written to {output_file}")

    print("All plots processed successfully.")

if __name__ == "__main__":
    use_bounding_box = False  # Set to True for bounding box mode, False for shapefile mode
    use_shapefile = True  # Set to True for shapefile mode, False for bounding box mode
    cpu_limit = 5
    expansion_percentage = 10  # Percentage to expand the bounding box in shapefile mode (e.g., 20 for 20% expansion)
    main(use_bounding_box, use_shapefile, cpu_limit, expansion_percentage)
