"""Our implementation of the Worldsheet method for novel view synthesis.

Based on [Hu et al. 2021].
"""

grid_mesh_size = [33, 33]
# grid_mesh_size = [65, 65]
input_image_size = [256, 256]

# 1. Generate a Worldsheet for a given view

# 1.1. Build the scene mesh by warping a grid sheet via grid offset and depth

# 1.1.1. Extract visual feature map using ResNet-50 pretrained on ImageNet with dilation

# 1.1.2. Predict the grid offset in -1 to 1 space

# 1.1.3. Predict depth

# 1.1.4. Build the mesh

# 1.2. Reconstruct the mesh texture with a differentiable texture sampler

# 1.2.1. Project image texture from input view to target view

# 1.2.2. Sample the mesh texture as a UV texture map from the input view

# 1.2.2.1. Project the mesh faces onto the image plane to build aascending z-buffer
# containing depth and 2D euclidean distance of points on the closest K mesh faces
# whose projection overlaps image pixel, as in PyTorch3D

# 1.2.2.2. Splat the RGB pixel intensities from the image onto the UV texture map

# 2. Render a novel view by moving the camera with a differentiable mesh renderer

# 3. Apply inpainting network

# Losses


def loss():

    lambda1 = 8
    lambda2 = 2
    lambda3 = 8
    lambda4 = 2
    lambda5 = 0.2
    lambda6 = 0.001

    L_out_rgb = None
    L_out_pc = None
    L_paint_rgb = None
    L_paint_pc = None
    L_g = None
    L_m = None

    return (lambda1 * L_out_rgb
            + lambda2 * L_out_pc
            + lambda3 * L_paint_rgb
            + lambda4 * L_paint_pc
            + lambda5 * L_g
            + lambda6 * L_m)
