# Validation accuracy for the base network given by 83.59
# Final Validation Accuracy 80.14 and best 80.57 at 48 th epoch

# model definition
model1 = Sequential()
model1.add(SeparableConv2D(filters=32,kernel_size=(3,3),padding='same',use_bias=False, input_shape=(32,32,3))) # 32 * 32 * 48
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(SeparableConv2D(filters=64,kernel_size=(3,3), activation='relu', use_bias=False)) #30*30*32
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(SeparableConv2D(filters=128,kernel_size=(3,3),  use_bias=False, activation='relu')) #28*28*64
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(SeparableConv2D(filters=128,kernel_size=(3,3), use_bias=False, activation='relu')) #26*26*128
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(SeparableConv2D(filters=128,kernel_size=(3,3), use_bias=False, activation='relu')) #26*26*128
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(MaxPooling2D(pool_size=(2,2))) #13*13*128
model1.add(Dropout(0.1))

model1.add(SeparableConv2D(filters=32,kernel_size=(1,1), use_bias=False, activation='relu')) # 13*13*16
model1.add(BatchNormalization())
model1.add(Dropout(0.1))

model1.add(SeparableConv2D(filters=64,kernel_size=(3,3), use_bias=False, activation='relu')) # 11*11*32
model1.add(BatchNormalization())
model1.add(Dropout(0.1))


model1.add(SeparableConv2D(filters=64,kernel_size=(3,3), use_bias=False, activation='relu')) #9*9*64
model1.add(BatchNormalization())
model1.add(Dropout(0.1))

model1.add(SeparableConv2D(filters=128,kernel_size=(3,3), use_bias=False, activation='relu')) #9*9*64
model1.add(BatchNormalization())
model1.add(Dropout(0.1))

model1.add(SeparableConv2D(filters=128,kernel_size=(3,3), use_bias=False, activation='relu')) #7*7*128
model1.add(BatchNormalization())
model1.add(Dropout(0.1))



model1.add(SeparableConv2D(filters=10,kernel_size=(1,1), use_bias=False, activation='relu')) #7*7*128
model1.add(BatchNormalization())
model1.add(Dropout(0.1))

#model1.add(SeparableConv2D(filters=10,kernel_size=(3,3), use_bias=False, activation='relu')) #1*1*10
model1.add(GlobalAveragePooling2D())
model1.add(Dense(units=10))
model1.add(Dense(units=100))
model1.add(Dense(units=10))
model1.add(Activation('softmax'))


model1.summary()

# Logs
Epoch 1/50
200/200 [==============================] - 84s 419ms/step - loss: 1.7394 - acc: 0.3499 - val_loss: 1.7267 - val_acc: 0.4128
Epoch 2/50
200/200 [==============================] - 75s 373ms/step - loss: 1.2999 - acc: 0.5297 - val_loss: 1.3655 - val_acc: 0.5274
Epoch 3/50
200/200 [==============================] - 75s 375ms/step - loss: 1.1274 - acc: 0.5957 - val_loss: 1.1318 - val_acc: 0.5997
Epoch 4/50
200/200 [==============================] - 75s 375ms/step - loss: 1.0327 - acc: 0.6321 - val_loss: 1.0653 - val_acc: 0.6368
Epoch 5/50
200/200 [==============================] - 75s 375ms/step - loss: 0.9659 - acc: 0.6574 - val_loss: 1.0080 - val_acc: 0.6538
Epoch 6/50
200/200 [==============================] - 74s 372ms/step - loss: 0.9092 - acc: 0.6786 - val_loss: 0.9245 - val_acc: 0.6838
Epoch 7/50
200/200 [==============================] - 74s 372ms/step - loss: 0.8664 - acc: 0.6944 - val_loss: 0.9121 - val_acc: 0.6926
Epoch 8/50
200/200 [==============================] - 75s 373ms/step - loss: 0.8326 - acc: 0.7057 - val_loss: 1.0777 - val_acc: 0.6499
Epoch 9/50
200/200 [==============================] - 75s 374ms/step - loss: 0.7950 - acc: 0.7201 - val_loss: 0.9388 - val_acc: 0.6874
Epoch 10/50
200/200 [==============================] - 74s 372ms/step - loss: 0.7732 - acc: 0.7270 - val_loss: 0.8306 - val_acc: 0.7197
Epoch 11/50
200/200 [==============================] - 74s 372ms/step - loss: 0.7482 - acc: 0.7351 - val_loss: 0.8895 - val_acc: 0.6967
Epoch 12/50
200/200 [==============================] - 75s 375ms/step - loss: 0.7235 - acc: 0.7447 - val_loss: 0.7974 - val_acc: 0.7310
Epoch 13/50
200/200 [==============================] - 75s 374ms/step - loss: 0.7091 - acc: 0.7501 - val_loss: 0.7713 - val_acc: 0.7396
Epoch 14/50
200/200 [==============================] - 75s 375ms/step - loss: 0.6887 - acc: 0.7570 - val_loss: 0.8632 - val_acc: 0.7165
Epoch 15/50
200/200 [==============================] - 75s 373ms/step - loss: 0.6730 - acc: 0.7617 - val_loss: 0.8548 - val_acc: 0.7118
Epoch 16/50
200/200 [==============================] - 74s 372ms/step - loss: 0.6541 - acc: 0.7710 - val_loss: 0.8991 - val_acc: 0.7049
Epoch 17/50
200/200 [==============================] - 74s 371ms/step - loss: 0.6439 - acc: 0.7743 - val_loss: 0.7471 - val_acc: 0.7456
Epoch 18/50
200/200 [==============================] - 74s 372ms/step - loss: 0.6365 - acc: 0.7765 - val_loss: 0.7457 - val_acc: 0.7468
Epoch 19/50
200/200 [==============================] - 74s 372ms/step - loss: 0.6197 - acc: 0.7828 - val_loss: 0.7165 - val_acc: 0.7573
Epoch 20/50
200/200 [==============================] - 74s 372ms/step - loss: 0.6097 - acc: 0.7846 - val_loss: 0.8184 - val_acc: 0.7311
Epoch 21/50
200/200 [==============================] - 75s 374ms/step - loss: 0.6021 - acc: 0.7886 - val_loss: 0.6916 - val_acc: 0.7663
Epoch 22/50
200/200 [==============================] - 74s 372ms/step - loss: 0.5942 - acc: 0.7905 - val_loss: 0.7070 - val_acc: 0.7630
Epoch 23/50
200/200 [==============================] - 75s 375ms/step - loss: 0.5828 - acc: 0.7946 - val_loss: 0.7187 - val_acc: 0.7601
Epoch 24/50
200/200 [==============================] - 75s 374ms/step - loss: 0.5721 - acc: 0.8006 - val_loss: 0.7230 - val_acc: 0.7609
Epoch 25/50
200/200 [==============================] - 75s 375ms/step - loss: 0.5667 - acc: 0.7991 - val_loss: 0.7244 - val_acc: 0.7626
Epoch 26/50
200/200 [==============================] - 75s 373ms/step - loss: 0.5570 - acc: 0.8043 - val_loss: 0.6644 - val_acc: 0.7766
Epoch 27/50
200/200 [==============================] - 75s 373ms/step - loss: 0.5476 - acc: 0.8076 - val_loss: 0.7253 - val_acc: 0.7611
Epoch 28/50
200/200 [==============================] - 75s 374ms/step - loss: 0.5470 - acc: 0.8071 - val_loss: 0.7826 - val_acc: 0.7395
Epoch 29/50
200/200 [==============================] - 75s 373ms/step - loss: 0.5350 - acc: 0.8132 - val_loss: 0.7639 - val_acc: 0.7477
Epoch 30/50
200/200 [==============================] - 75s 374ms/step - loss: 0.5309 - acc: 0.8126 - val_loss: 0.6845 - val_acc: 0.7755
Epoch 31/50
200/200 [==============================] - 75s 375ms/step - loss: 0.5255 - acc: 0.8167 - val_loss: 0.6467 - val_acc: 0.7806
Epoch 32/50
200/200 [==============================] - 75s 376ms/step - loss: 0.5222 - acc: 0.8180 - val_loss: 0.6791 - val_acc: 0.7785
Epoch 33/50
200/200 [==============================] - 75s 373ms/step - loss: 0.5135 - acc: 0.8192 - val_loss: 0.6547 - val_acc: 0.7882
Epoch 34/50
200/200 [==============================] - 75s 373ms/step - loss: 0.5071 - acc: 0.8206 - val_loss: 0.6392 - val_acc: 0.7892
Epoch 35/50
200/200 [==============================] - 74s 372ms/step - loss: 0.4970 - acc: 0.8258 - val_loss: 0.6963 - val_acc: 0.7699
Epoch 36/50
200/200 [==============================] - 75s 374ms/step - loss: 0.4969 - acc: 0.8270 - val_loss: 0.6956 - val_acc: 0.7751
Epoch 37/50
200/200 [==============================] - 75s 373ms/step - loss: 0.4911 - acc: 0.8281 - val_loss: 0.7761 - val_acc: 0.7496
Epoch 38/50
200/200 [==============================] - 75s 373ms/step - loss: 0.4930 - acc: 0.8271 - val_loss: 0.7152 - val_acc: 0.7707
Epoch 39/50
200/200 [==============================] - 75s 375ms/step - loss: 0.4787 - acc: 0.8307 - val_loss: 0.7008 - val_acc: 0.7742
Epoch 40/50
200/200 [==============================] - 75s 375ms/step - loss: 0.4750 - acc: 0.8319 - val_loss: 0.6026 - val_acc: 0.7962
Epoch 41/50
200/200 [==============================] - 75s 376ms/step - loss: 0.4737 - acc: 0.8333 - val_loss: 0.6352 - val_acc: 0.7890
Epoch 42/50
200/200 [==============================] - 75s 374ms/step - loss: 0.4664 - acc: 0.8364 - val_loss: 0.6322 - val_acc: 0.7942
Epoch 43/50
200/200 [==============================] - 74s 372ms/step - loss: 0.4638 - acc: 0.8365 - val_loss: 0.6254 - val_acc: 0.7888
Epoch 44/50
200/200 [==============================] - 75s 373ms/step - loss: 0.4589 - acc: 0.8388 - val_loss: 0.6630 - val_acc: 0.7886
Epoch 45/50
200/200 [==============================] - 75s 373ms/step - loss: 0.4568 - acc: 0.8400 - val_loss: 0.6607 - val_acc: 0.7823
Epoch 46/50
200/200 [==============================] - 75s 373ms/step - loss: 0.4529 - acc: 0.8408 - val_loss: 0.6133 - val_acc: 0.8003
Epoch 47/50
200/200 [==============================] - 75s 374ms/step - loss: 0.4504 - acc: 0.8413 - val_loss: 0.6562 - val_acc: 0.7849
Epoch 48/50
200/200 [==============================] - 75s 375ms/step - loss: 0.4485 - acc: 0.8431 - val_loss: 0.5834 - val_acc: 0.8057
Epoch 49/50
200/200 [==============================] - 75s 375ms/step - loss: 0.4407 - acc: 0.8429 - val_loss: 0.6523 - val_acc: 0.7908
Epoch 50/50
200/200 [==============================] - 75s 374ms/step - loss: 0.4387 - acc: 0.8459 - val_loss: 0.6204 - val_acc: 0.8014
