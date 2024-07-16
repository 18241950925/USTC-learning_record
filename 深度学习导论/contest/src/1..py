import matplotlib.pyplot as plt
# 数据
epochs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
loss = [0.0452, 0.0241, 0.0143, 0.0087, 0.0052, 0.0028, 0.0016, 0.0010, 0.0008]
# 绘图
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss, marker='o')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.grid()
plt.show()
