import numpy as np
import utils
import string

import matplotlib.pyplot as plt


epochs = 5
learning_rate = 0.01
alpha = 0.01


def train_model(images, labels, w_in, w_out, b_in, b_out):
    for epoch in range(epochs):
        loss = 0
        correct = 0
        print(f"Epoch number {epoch + 1}")

        for image, answer_label in zip(images, labels):
            image = np.reshape(image, (-1, 1))
            answer_label = np.reshape(answer_label, (-1, 1))

            hidden_raw = b_in + w_in @ image
            hidden = np.where(hidden_raw > 0, hidden_raw, alpha * hidden_raw)

            output_raw = b_out + w_out @ hidden
            exp_vals = np.exp(output_raw - np.max(output_raw))
            output = exp_vals / np.sum(exp_vals)

            loss += - (np.sum(answer_label * np.log(output + 1e-9)))
            correct += int(np.argmax(output) == np.argmax(answer_label))

            # Backprop
            delta_output = output - answer_label
            delta_hidden_raw = np.transpose(w_out) @ delta_output
            lrelu = np.where(hidden_raw > 0, 1.0, alpha)
            delta_hidden = delta_hidden_raw * lrelu

            w_in += - (learning_rate * delta_hidden @ np.transpose(image))
            w_out += - (learning_rate * delta_output @ np.transpose(hidden))
            b_in += - (learning_rate * delta_hidden)
            b_out += - (learning_rate * delta_output)

        print(f"Correct: {round(correct / images.shape[0] * 100, 5)}%")
        print(f"Loss: {round(loss / images.shape[0], 5)}\n")

    return w_in, w_out, b_in, b_out


#! ЗАПУСК ДЛЯ ЦИФР
print("--- Training Numbers ---")
img_num, lbl_num = utils.load_dataset(10, "mnist.npz")
w_in_n = np.random.uniform(-0.5, 0.5, (20, 784))
w_out_n = np.random.uniform(-0.5, 0.5, (10, 20))
b_in_n = np.zeros((20, 1))
b_out_n = np.zeros((10, 1))

w_in_n, w_out_n, b_in_n, b_out_n = train_model(img_num, lbl_num, w_in_n, w_out_n, b_in_n, b_out_n)

#! ЗАПУСК ДЛЯ БУКВ
# print("--- Training Letters ---")
# img_let, lbl_let = utils.load_dataset(26, "emnist.npz")
# w_in_l = np.random.uniform(-0.5, 0.5, (20, 784))
# w_out_l = np.random.uniform(-0.5, 0.5, (26, 20))
# b_in_l = np.zeros((20, 1))
# b_out_l = np.zeros((26, 1))

# w_in_l, w_out_l, b_in_l, b_out_l = train_model(img_let, lbl_let, w_in_l, w_out_l, b_in_l, b_out_l)




def predict_custom_image(image_path, w_in, w_out, b_in, b_out, mode="numbers"):
    new_test_image = plt.imread(image_path)
    if len(new_test_image.shape) == 3:
        new_test_image = np.mean(new_test_image, axis=2)


    new_test_image = 1.0 - (new_test_image.astype("float32") / 255.0)
    image_vec = np.reshape(new_test_image, (-1, 1))

    hidden_raw = b_in + w_in @ image_vec
    hidden = np.where(hidden_raw > 0, hidden_raw, alpha * hidden_raw)

    output_raw = b_out + w_out @ hidden
    exp_vals = np.exp(output_raw - np.max(output_raw))
    output = exp_vals / np.sum(exp_vals)

    predicted_index = np.argmax(output)

    if mode == "letters":
        alphabet = string.ascii_uppercase
        result = alphabet[predicted_index]
    else:
        result = str(predicted_index)

    plt.imshow(new_test_image, cmap="Grays")
    plt.title(f"NN suggests: {result} ({round(np.max(output) * 100, 2)}%)")
    plt.show()


predict_custom_image("number.jpg", w_in_n, w_out_n, b_in_n, b_out_n, mode="numbers")