from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.optimizers import RMSprop

def Create_Vgg16(classes = 2):
    base_model = VGG16(include_top=False,weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(classes,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=predictions)
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=RMSprop(lr=0.001,decay=0.9,epsilon=0.1),
                  loss='categorical_crossentropy',metrics=['accuracy'])
    return model
