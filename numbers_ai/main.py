import numpy as np
import matplotlib.pyplot as plt
import utils


images, answer_labels = utils.load_dataset()

weights_inp_to_hid = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hid_to_out = np.random.uniform(-0.5, 0.5, (10, 20))

bias_inp_to_hid = np.zeros((20, 1))
bias_hid_to_out = np.zeros((10, 1))

epochs = 5
loss = 0
correct = 0
learning_rate = 0.01
alpha = 0.01

for epoch in range(epochs):
    print(f"Epoch number {epoch + 1}")
    for image, answer_label in zip(images, answer_labels):
        #До этого image был просто вектором из 784 чисел.
        # 1 столбец, и столько строк, чтобы подошло по размеру так сказать
        image = np.reshape(image, (-1, 1))
        answer_label = np.reshape(answer_label, (-1, 1))

        hidden_raw = bias_inp_to_hid + weights_inp_to_hid @ image
        hidden = np.where(hidden_raw > 0, hidden_raw, alpha * hidden_raw)

        output_raw = bias_hid_to_out + weights_hid_to_out @ hidden
        exp_vals = np.exp(output_raw - np.max(output_raw))
        output = exp_vals / np.sum(exp_vals)

        loss += - (np.sum(answer_label * np.log(output + 1e-9)))
        correct += int(np.argmax(output) == np.argmax(answer_label))

        #Backpropagation
        delta_output = output - answer_label
        delta_hidden_raw = np.transpose(weights_hid_to_out) @ delta_output
        lrelu = np.where(hidden_raw > 0, 1.0, alpha)
        delta_hidden = delta_hidden_raw * lrelu

        weights_inp_to_hid += - (learning_rate * delta_hidden @ np.transpose(image))
        weights_hid_to_out += - (learning_rate * delta_output @ np.transpose(hidden))
        bias_inp_to_hid += - (learning_rate * delta_hidden)
        bias_hid_to_out += - (learning_rate * delta_output)


    print(f"Correct: {round(correct / images.shape[0] * 100, 5)}%")
    print(f"Loss: {round(loss / images.shape[0], 5)}")

    correct, loss = 0, 0

    print("")


# В ошибках я писал 1 - np.reshape..., но это ошибка!
# 1 - означает полностью перевернуть цвета!!!



# new_test_image = plt.imread("number.jpg", format="jpeg")
# new_test_image = new_test_image.astype("float32") / 255
#
# image = np.reshape(new_test_image, (-1, 1))
#
# hidden_raw = bias_inp_to_hid + weights_inp_to_hid @ image
# hidden = np.where(hidden_raw > 0, hidden_raw, alpha * hidden_raw)
#
# output_raw = bias_hid_to_out + weights_hid_to_out @ hidden
# exp_vals = np.exp(output_raw - np.max(output_raw))
# output = exp_vals / np.sum(exp_vals)
#
# # cmap gray - оригинальное, Grays - перевёрнотое
# plt.imshow(new_test_image.reshape(28, 28), cmap="Grays")
# plt.title(f"NN suggest the number is: {output.argmax()}")
# plt.show()

#Для сохранения нужно запомнить только веса по факт у. Нейросеть = архитектура + веса
# np.savez("20_epoch.npz",
#          w1=weights_inp_to_hid,
#          w2=weights_hid_to_out,
#          b1=bias_inp_to_hid,
#          b2=bias_hid_to_out)
