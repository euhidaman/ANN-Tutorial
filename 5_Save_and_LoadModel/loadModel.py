# %%
from tensorflow.keras.models import load_model

new_model = load_model('models/medical_trial_model.h5')
new_model.summary()

# %%
new_model.get_weights()

# %%
new_model.optimizer

# %%
