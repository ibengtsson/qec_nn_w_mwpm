import torch
import matplotlib.pyplot as plt

saved_attributes = torch.load('../training_outputs/d5_d_t_5_240315-102016.pt')
training_history = saved_attributes["training_history"]
epoch = saved_attributes["training_history"]["epoch"] + 1
model = saved_attributes["model"]
#print(model)
#print(training_history)
#print(epoch)
accuracy = training_history["train_accuracy"]
val_acc = training_history["val_accuracy"]
comb_acc = training_history["comb_accuracy"]
iter_imp = training_history["iter_improvement"]

plt.plot(iter_imp, accuracy)
plt.plot(iter_imp, val_acc)
plt.plot(iter_imp, comb_acc)
plt.show()
