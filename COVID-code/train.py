import VGG16
import Generator
from keras.callbacks import TensorBoard


def train(steps_per_epoch=326, validation_steps=67, epcohs=50):
    train_generator = Generator.train_data()
    valid_generator = Generator.valid_data()
    visualization = TensorBoard(log_dir='./logs', write_graph=True)
    model = VGG16.Create_Vgg16()
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
              epochs=epcohs, validation_data=valid_generator,
              validation_steps=validation_steps, verbose=1,
              callbacks=[visualization])
    model.save('./COVID2019-VGG16')


train()
