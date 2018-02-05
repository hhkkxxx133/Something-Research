import matplotlib.pyplot as plt

train_loss = []
test_loss = []
train_acc = []
test_acc = []
with open('log_vgg.txt') as f:
    for line in f.readlines():
        if line.startswith(' >>> binary Train'):
            content = line.split()
            train_acc.append(float(content[-3]))
            train_loss.append(float(content[-1]))
        if line.startswith(' >>> binary Validate'):
            content = line.split()
            test_acc.append(float(content[-3]))
            test_loss.append(float(content[-1]))


plt.figure()
plt.plot(list(range(1,8)), train_acc, 'ro', label='training accuracy')
plt.plot(list(range(1,8)), train_acc)
plt.plot(list(range(1,8)), test_acc, 'go', label='test accuracy')
plt.plot(list(range(1,8)), test_acc)
plt.title('Training/Test Accuracy')
plt.legend()
plt.savefig('accuracy.png')

plt.figure()
plt.plot(list(range(1,8)), train_loss, 'ro', label='training loss')
plt.plot(list(range(1,8)), train_loss)
plt.plot(list(range(1,8)), test_loss, 'go', label='test loss')
plt.plot(list(range(1,8)), test_loss)
plt.title('Training/Test Loss')
plt.legend()
plt.savefig('loss.png')
