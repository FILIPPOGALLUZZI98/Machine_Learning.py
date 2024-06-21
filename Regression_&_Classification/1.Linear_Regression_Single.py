
# A questo punto, per vedere la retta trovata
f_wb = model_single(x_train, w_final, b_final)
plt.plot(x_train, f_wb, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.show()











