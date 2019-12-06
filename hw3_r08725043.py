from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

def Model(width, height, depth, classes):
# initialize the model along with the input shape to be
# "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
 
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model
 
def main():
    print("[INFO] loading Fashion MNIST...")
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    NUM_EPOCHS = 25
    INIT_LR = 1e-2
    BS = 32

    if K.image_data_format() == "channels_first":
        trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
        testX = testX.reshape((testX.shape[0], 1, 28, 28))
    else:
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # scale data to the range of [0, 1]
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    
    # one-hot encode the training and testing labels
    trainY = np_utils.to_categorical(trainY, 10)
    testY = np_utils.to_categorical(testY, 10)
    
    # initialize the label names
    labelNames = ["top", "trouser", "pullover", "dress", "coat","sandal", "shirt", "sneaker", "bag", "ankle boot"]

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    model = Model(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    # train the network
    print("[INFO] training model...")
    H = model.fit(trainX, trainY,validation_data=(testX, testY), batch_size=BS, epochs=NUM_EPOCHS)

    preds = model.predict(testX)
 
    print("[INFO] evaluating network...")
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),target_names=labelNames))
    
    #learning curve plot
    plt.style.use('ggplot')
    N = NUM_EPOCHS
    plt.figure()
    plt.plot(np.arange(0,N), H.history['loss'], label = 'train_loss')
    plt.plot(np.arange(0,N), H.history['val_loss'], label = 'val_loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')

    #accuracy rate plot
    plt.figure()
    plt.plot(np.arange(0,N), H.history['acc'], label = 'train_acc')
    plt.plot(np.arange(0,N), H.history['val_acc'], label = 'val_acc')
    plt.title('Accuracy Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower left')

    layer_outputs = [layer.output for layer in model.layers[:1]] #first layer
    act_model = models.Model(input = model.input, outputs = layer_outputs)

    #show first test image and its label
    img = testX[0].reshape(1,28,28,1)
    prediction = preds.argmax(axis=1)
    label = labels[prediction[0]]
    print(label)
    plt.imshow(img[0,:,:,0],cmap='binary')
    plt.axis('off')

    #show activations of the first layer
    act = act_model.predict(img)

    img_row = 8
    first_act = act[0] #first layer activation 
    features = first_act.shape[-1]
    size = first_act.shape[1]
    first_act = first_act.reshape(1,28,28,32)
    n_cols = features // img_row
    display_grid = np.zeros((size * n_cols, img_row * size))
    for col in range(n_cols):
        for row in range(img_row):
            chan_img = first_act[0,:, :,col * img_row + row]
            chan_img -= chan_img.mean() 
            chan_img /= chan_img.std()
            chan_img *= 64
            chan_img = np.clip(chan_img, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = chan_img
            scale = 1 / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='binary')

    from imutils import build_montages

    images = []

    for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
        probs = model.predict(testX[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        label = labels[prediction[0]]
        if K.image_data_format() == "channels_first":
            image = (testX[i][0] * 255).astype("uint8")
        else:
            image = (testX[i] * 255).astype("uint8")
        #green text if correct
        color = (0, 255, 0)
        #red text if incorrect
        if prediction[0] != np.argmax(testY[i]):
            color = (0, 0, 255)
    
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        images.append(image)

    montage = build_montages(images, (96, 96), (4, 4))[0]
    
    # show the output
    cv2.imshow("Fashion MNIST", montage)
    cv2.waitKey(0)
    
if __name__=='__main__':
    main() 