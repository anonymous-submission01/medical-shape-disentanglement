import os
import trimesh

#Directory containing the .ply files
#input_directory = '../../../torus_bump_5000_two_scale_binary_bump_variable_noise_fixed_angle'
#output_directory = '../../../torus_bump_5000_two_scale_binary_bump_variable_noise_fixed_angle/obj_files'

input_directory = '../../../OAI-ZIB/mesh_minimal'
output_directory = '../../../OAI-ZIB/mesh_minimal_obj'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate through all files in the directory
for filename in os.listdir(input_directory):
    # Check if the file is a .ply file
    if filename.endswith('.ply'):
        # Full path of the .ply file
        ply_path = os.path.join(input_directory, filename)
        
        # Load the .ply file
        mesh = trimesh.load(ply_path)
        
        # Output file path (change the extension to .obj)
        obj_filename = filename.replace('.ply', '.obj')
        obj_path = os.path.join(output_directory, obj_filename)
        
        # Export the file to .obj format
        mesh.export(obj_path)
        
        print(f'Converted: {filename} -> {obj_filename}')
