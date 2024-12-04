import matplotlib
matplotlib.use('TkAgg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
# # Create a simple plot
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]

# plt.plot(x, y, label="y = 2x")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Simple Line Plot")
# plt.legend()
# plt.show()

# # Save the plot to a file
# plt.savefig("test_plot.png")
# print("Plot saved as test_plot.png")

loaded_rgb = np.loadtxt("geekfile.txt")

load_original_rgb = loaded_rgb.reshape(
    loaded_rgb.shape[0], loaded_rgb.shape[1] // 3, 3)

print(load_original_rgb)

# seg = np.moveaxis(np.squeeze(seg, axis=0), 0, -1).argmax(axis=-1)
# rgb_seg = LABEL_COLORMAP[seg]

plt.figure(figsize=(18, 7))
plt.subplot(1, 3, 1)
plt.imshow(load_original_rgb)
plt.gca().set_title("RGB image")
plt.axis("off")
plt.savefig("test_plot.png")
plt.show()