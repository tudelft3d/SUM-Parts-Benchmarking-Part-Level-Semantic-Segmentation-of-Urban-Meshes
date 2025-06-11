from PIL import Image
import numpy as np

# Load the image
image_path = 'C:/wendang/TUDelft/work/papers_writing/paper3/pics/raw_edit_fig/fig19/low_vegetation_1_sam.png'
image = Image.open(image_path)

# Convert the image to a numpy array
image_array = np.array(image)

# Create a mask for the black pixels (i.e., pixels where all RGB values are 0)
mask = (image_array[:, :, 0] == 0) & (image_array[:, :, 1] == 0) & (image_array[:, :, 2] == 0)

# Change black pixels to white
image_array[mask] = [255, 255, 255]

# Convert the numpy array back to an image
new_image = Image.fromarray(image_array)

# Save the new image
new_image_path = 'C:/wendang/TUDelft/work/papers_writing/paper3/pics/raw_edit_fig/fig19/low_vegetation_1_sam.png'
new_image.save(new_image_path)

print(f"Image saved to {new_image_path}")
