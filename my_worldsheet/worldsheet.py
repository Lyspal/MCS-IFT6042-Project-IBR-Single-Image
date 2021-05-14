"""Our implementation of the Worldsheet method for novel view synthesis.

Based on [Hu et al. 2021].
"""

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
