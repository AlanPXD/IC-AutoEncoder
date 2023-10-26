from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Add , Input, BatchNormalization, ReLU, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from os import environ
environ["CUDA_VISIBLE_DEVICES"]="0"

model_name = "ResidualAutoEncoder-2.1-64x64"

class resblock (Layer):

    def __init__(self, kernel_size, filters, activation = 'relu', padding = 'same', trainable=True, name="ResidualBlock", dtype=None, dynamic=False, **kwargs):
        self.kernel_size = kernel_size
        self.filters = filters
        self.activation = activation
        self.padding = padding
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def __call__(self, x, *args, **kwargs):
        
        fx = Conv2D(x.shape[-1], self.kernel_size, activation = self.activation, padding=  self.padding)(x)
        fx = BatchNormalization()(fx)
        fx = Conv2D(x.shape[-1], self.kernel_size, padding = self.padding)(fx)
        out = Add()([x,fx])
        out = ReLU()(out)
        out = BatchNormalization()(out)
        out = Conv2D(self.filters, self.kernel_size, activation = self.activation, padding=  self.padding)(out)
        out = BatchNormalization()(out)
        return out

inputs = Input(shape=(64,64,1))
layer_1 = resblock(filters = 10, kernel_size = 5, padding = 'same', activation = 'relu')(inputs)
layer_2 = resblock(filters = 20, kernel_size = 5, padding = 'same', activation = 'relu')(layer_1)
layer_3 = MaxPooling2D()(layer_2)
layer_4 = resblock(filters = 30, kernel_size = 3, padding = 'same', activation = 'relu')(layer_3)
layer_5 = resblock(filters = 40, kernel_size = 3, padding = 'same', activation = 'relu')(layer_4)
layer_6 = MaxPooling2D()(layer_5)
layer_7 = resblock(filters = 40, kernel_size = 3, padding = 'same', activation = 'relu')(layer_6)
layer_8 = resblock(filters = 30, kernel_size = 3, padding = 'same', activation = 'relu')(layer_7)
layer_9 = UpSampling2D()(layer_8)
layer_10 = resblock(filters = 20, kernel_size = 3, padding = 'same', activation = 'relu')(layer_9)
layer_11 = resblock(filters = 10, kernel_size = 3, padding = 'same', activation = 'relu')(layer_10)
layer_12 = UpSampling2D()(layer_11)
layer_13 = Conv2D(filters = 5, kernel_size = 3, padding = 'same', activation = 'relu')(layer_12)
layer_14 = Conv2D(filters = 1, kernel_size = 3, padding = 'same', activation = 'relu')(layer_13)

model = Model(inputs = inputs, outputs = layer_14, name = model_name)

model_json = model.to_json(indent = 4)

with open("nNet_models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
    json_file.close()

plot_model(model=model, to_file="nNet_models/PNG-Models/" + model_name + '.png', show_shapes=True, rankdir= "TB", expand_nested=True )

print(model.count_params())