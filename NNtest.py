import tensorflow as tf

model = tf.keras.models.load_model("PNN(1)")
predictions = model.predict()
print(predictions)
