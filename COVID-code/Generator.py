from keras.preprocessing.image import ImageDataGenerator
batch_size = 5
width,height = 224,224

def train_data(train_data_dir="../train"):
    train_data_gen = ImageDataGenerator(rescale=1./225,
                                        rotation_range=15,
                                        shear_range=0.5,
                                        zoom_range=0.2,
                                        width_shift_range=0.3,
                                        height_shift_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True)
    train_generator = train_data_gen.flow_from_directory(train_data_dir,
                                                         target_size=(width,height),
                                                         batch_size=batch_size,
                                                         class_mode='categorical')
    return train_generator
def valid_data(valid_data_dir="../test"):
    valid_data_gen = ImageDataGenerator(rescale=1./225)
    valid_generator = valid_data_gen.flow_from_directory(valid_data_dir,
                                                         target_size=(width,height)
                                                         ,batch_size=batch_size,
                                                         class_mode='categorical')
    return valid_generator