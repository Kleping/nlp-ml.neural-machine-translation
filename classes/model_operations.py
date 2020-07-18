import random

from classes.constant import MODEL_PATH, LATENT_DIMENSIONS, EPOCHS
from classes.model_definition import restore_model, create_model
from classes.auxiliary import get_voc, encode_seq, decode_seq
from classes.data_operations import get_raw_data, get_data


def inference_model():
    raw_data = get_raw_data(1., True)
    voc = get_voc(raw_data)
    encoder_model, decoder_model = restore_model(LATENT_DIMENSIONS)

    for sentence in random.choices(sum(raw_data.values(), []), k=10):
        input_data = encode_seq(sentence, voc)
        output = decode_seq(input_data, encoder_model, decoder_model, voc)
        print('\ninput ' + sentence, '\noutput ' + output)


def train_model():
    (train_data, validation_data), (steps_per_epoch, validation_steps), voc = get_data(.05, True, .2)
    model = create_model(len(voc), LATENT_DIMENSIONS)

    model.fit_generator(generator=train_data,
                        validation_data=validation_data,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        epochs=EPOCHS,
                        verbose=2,
                        use_multiprocessing=False,
                        shuffle=True)

    model.save(MODEL_PATH)
