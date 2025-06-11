import cv2
import numpy as np

# Load the image
image = cv2.imread('C:/wendang/TUDelft/work/papers_writing/paper3/pics/raw_edit_fig/fig25/Tile_+1985_+2688_0.jpg')

# Generate Gaussian noise
mean = 0
std_dev = 3  # Standard deviation of the noise
gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype('uint8')

# Add the Gaussian noise to the image
noisy_image = cv2.add(image, gaussian_noise)

cv2.imwrite('C:/wendang/TUDelft/work/papers_writing/paper3/pics/raw_edit_fig/fig25/Tile_+1985_+2688_0_noise_3.jpg', noisy_image)
# Display the noisy image
# cv2.imshow('Noisy Image', noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
