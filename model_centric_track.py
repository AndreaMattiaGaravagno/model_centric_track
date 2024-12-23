import tensorflow_model_optimization as tfmot 
import tensorflow_datasets as tfds
import tensorflow as tf

model_name = 'wv_k_8_c_5'

#Some hyper parameters
#Play with them!
input_shape = (50,50,3)
batch_size = 512
learning_rate = 0.001
epochs = 100

#Load the Wake Vision Dataset

#The first execution will require a lot of time, it has to download the whole Wake Vision dataset from Tensorflow Datasets (https://www.tensorflow.org/datasets/catalog/wake_vision) on your machine. The next executions it will simply use the downloaded data. 

#Where to save the downloaded dataset (239.25 GiB)
data_dir = "/path/to/dataset/"

#5,760,428 images, suitable for pre-training (not used in this example)
#train_large_ds = tfds.load('wake_vision', split="train_large", shuffle_files=True, data_dir=data_dir)

#1,322,574 images with high quality labels 
train_quality_ds = tfds.load('wake_vision', split="train_quality", shuffle_files=True, data_dir=data_dir)

validation_ds = tfds.load('wake_vision', split="validation", shuffle_files=True, data_dir=data_dir)
test_ds = tfds.load('wake_vision', split="test", shuffle_files=True, data_dir=data_dir)

#prepare images
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1]),
    #add some data augmentation
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])
    
train_quality_ds = train_quality_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

#Sample Architecture
#Play with it! 

inputs = tf.keras.Input(shape=input_shape)
#
x = tf.keras.layers.Conv2D(8, (3,3), padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
#
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(16, (3,3), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
#
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(24, (3,3), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
#
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(30, (3,3), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
#
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(34, (3,3), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
#
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(37, (3,3), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
#
x = tf.keras.layers.GlobalAveragePooling2D()(x)
#
x = tf.keras.layers.Dense(37)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
#
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#validation based early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".tf",
    monitor='val_accuracy',
    mode='max', save_best_only=True)

#training
model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[model_checkpoint_callback])


#Post Training Quantization (PTQ)
model = tf.keras.models.load_model(model_name + ".tf")

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
interpreter = tf.lite.Interpreter(tflite_quant_model)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy().argmax() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")
