# You will use hist() to display the 256 different intensities of the red color. And ravel() to make these color values an array of one flat dimension.

# Matplotlib is preloaded as plt and Numpy as np.

# Remember that if we want to obtain the green color of an image we would do the following:

# green = image[:, :, 1]
# Instructions
# 100 XP
# Obtain the red channel using slicing.
# Plot the histogram and bins in a range of 256. Don't forget .ravel() for the color channel.
# Obtain the red channel
red_channel = image[:, :, 0]

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(), bins=256)

# Set title and show
plt.title('Red Histogram')
plt.show()
