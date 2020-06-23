from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

model = load_model('unnormalized_data.h5')
plot_model(model, to_file='model.png', show_shapes=True)
