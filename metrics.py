import matplotlib.pyplot as plt

# Data points extracted from the logs
epochs = [0.1, 0.19, 0.29, 0.39, 0.49, 0.58, 0.68, 0.78, 0.88, 0.97, 1.07, 1.17, 1.26, 1.36, 1.46, 1.56, 1.65, 1.75, 1.85, 1.95]
loss = [9.8979, 5.3552, 3.017, 2.3606, 2.2296, 1.8959, 2.1702, 1.8527, 1.7289, 1.8234, 1.6494, 1.5614, 1.5913, 1.715, 1.4723, 1.5408, 1.5051, 1.5812, 1.5794, 1.5839]
grad_norm = [7.82246732711792, 2.0048828125, 1.6196556091308594, 1.0519940853118896, 1.5161809921264648, 0.9111020565032959, 1.849332571029663, 0.8743117451667786, 0.951323926448822, 0.8853943943977356, 
             1.1392110586166382, 1.3274261951446533, 1.72735595703125, 1.9319404363632202, 1.4973491430282593, 0.9786012172698975, 1.9257068634033203, 1.357616901397705, 1.5174765586853027, 2.0738258361816406]
learning_rate = [9.828793774319066e-05, 9.634241245136188e-05, 9.439688715953307e-05, 9.245136186770429e-05, 9.05058365758755e-05, 8.85603112840467e-05, 8.66147859922179e-05, 8.46692607003891e-05, 
                 8.272373540856032e-05, 8.077821011673153e-05, 7.883268482490273e-05, 7.688715953307394e-05, 7.494163424124513e-05, 7.299610894941635e-05, 7.105058365758756e-05, 6.910505836575876e-05, 
                 6.715953307392995e-05, 6.521400778210117e-05, 6.326848249027238e-05, 6.132295719844359e-05]

# Plot loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label="Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.legend()

# Plot gradient norm over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, grad_norm, label="Gradient Norm", marker="o", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm Over Epochs")
plt.grid(True)
plt.legend()

# Plot learning rate over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, learning_rate, label="Learning Rate", marker="o", color="green")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Over Epochs")
plt.grid(True)
plt.legend()

plt.show()

