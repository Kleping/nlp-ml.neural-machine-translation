import matplotlib.pyplot as plt
import json

language_tag = 'en'
model_names = ['nmt_100_128_100_32', 'nmt_100_128_200_32', 'nmt_100_128_300_32']
legend_names = [str(int(100*128)), str(int(200*128)), str(int(300*128))]


def deserialize_and_read_histories():
    histories = list()
    for model_name in model_names:
        with open('models/{}/{}.config'.format(language_tag, model_name), 'r') as config_file:
            histories.append(json.load(config_file)['history'])

    return histories


histories = deserialize_and_read_histories()

fig, (ax_val_accuracy, ax_val_loss) = plt.subplots(1, 2, figsize=(12, 5))

for h in histories:
    ax_val_accuracy.plot(range(len(h['val_accuracy'])), h['val_accuracy'])
    ax_val_loss.plot(range(len(h['val_loss'])), h['val_loss'])

ax_val_accuracy.set_ylabel('val_accuracy')
ax_val_accuracy.set_xlabel('epochs')
ax_val_accuracy.legend(legend_names)

ax_val_loss.set_ylabel('val_loss')
ax_val_loss.set_xlabel('epochs')
ax_val_loss.legend(legend_names)

plt.show()
