# First attempt
## First stage: data uploading and augmentation
- ### params used:
    - size = 64 pixels
    - batch = 32
    - 0-1 scale (instead of 0-255)
    - validation data is 20%
    - rotation range = 10
    - width shift range = 0.1
    - height shift range = 0.1
    - zoom range = 0.1
    - brightness range = 0.8-1.2
    - no horizontal flip 
- ### generating the dataset:
  - one hot encoding
  - shuffle used
## Second stage: attempt a custom CNN first
### First model:
  - learning rate = 0.001
  - optimizer = adam
  - loss = categorical cross entropy
  - metrics = accuracy
  - used dropout layers between hidden layers
  - used batch normalization between layers
  - train first model with 10 epochs
### Results:
  - model didn't learn
### Second model:
  - learning rate = 1e-4
  - softmax as last custom dense layer activation instead of seperate layer (no change)
  - no augmentation applied to validation split
### Results:
  - train accuracy = 90%
  - validation accuracy = 83%
  - train loss = 0.1
  - validation loss = 0.5
  ![alt text](training_curves.png)
### Live test:
  - failed miserabley
  - but it was not expected to be accurate
  - purely done for the training of writing custom dense layers using keras api
# Second attempt
## First stage: augmentation
  - ### params used:
    - size = 128
    - batch = 32
    - preprocessing_function = preprocess_input, scales to [-1, 1]
    - validation_split = 0.2
    - rotation_range = 15
    - width_shift_range = 0.15
    - height_shift_range = 0.15
    - zoom_range = 0.15
    - brightness_range = [0.6, 1.4], more aggressive than before
    - shear_range = 0.1, slight shear for hand tilt
    - horizontal_flip = False
  - ### generating the dataset:
    - one hot encoding
    - shuffle used
## Second stage: attempt transfer learning model (mobilenetV2)
### First model:
  - epochs: 10 1st phase -> frozen / 20 2nd phase -> fine-tuning
  - input_shape = (128, 128, 3),
  - include_top = False, remove ImageNet head
  - weights = 'imagenet', use pretrained weights
  - frozen base for phase 1 and added custom classification head:
    - x = base_model.output
    - x = GlobalAveragePooling2D()(x)
    - x = BatchNormalization()(x)
    - x = Dense(256, activation='relu')(x)
    - x = Dropout(0.5)(x)
    - x = Dense(128, activation='relu')(x)
    - x = Dropout(0.3)(x)
    - output = Dense(NUM_CLASSES, activation='softmax')(x)
  - learning rate = 0.001
  - optimizer: adam
  - metrics: accuracy
  - phase 2 unfreeze bae for fine-tuning
  - learning rate = 1e-5
  - optimizer: adam
  - metrics: accuracy
  - loss: categorical-crossentropy
### Results:
  - model learnt
  - good train to val accuracy and loss
  - no sign of overfitting or underfitting
### Live test:
  - still not accurate
  - noticeable improvement from custom cnn
  - no consistent correct predictions
# Third attempt
## First stage: switch from pixels to landmarks
- ### rationale:
  - CNN/transfer learning models kept learning background, lighting, and session artifacts instead of hand shape (val accuracy didn't reflect live performance)
  - `nothing` class dropped, a no-hand class can't be represented as landmark coordinates, and the live pipeline already handles "no hand detected" natively
- ### extraction:
  - MediaPipe Hands (Tasks API - `HandLandmarker`), 1 hand max, min detection confidence 0.5
  - 21 landmarks per hand, (x, y) only -> 42 values per sample
  - normalization: subtract wrist (landmark 0) for translation invariance, divide by wrist -> middle-finger-MCP distance for scale invariance
  - dataset read unflipped, matching original (non-mirrored) capture
- ### results:
  - usable samples: 63,590
  - skipped (no hand detected): 23,410 (~27%, mostly `nothing` + some ambiguous `del`/`space` frames)

## Second stage: train a small MLP on landmark vectors
- ### params used:
  - input shape = (42,)
  - architecture: Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(28, softmax)
  - optimizer: adam
  - loss: categorical crossentropy
  - metrics: accuracy
  - early stopping on val_loss, patience 5, restore best weights
  - stratified 80/20 split
- ### results:
  - model learnt cleanly, small architecture, low-dimensional input, no capacity bottleneck

### Live test:
- initial result: rarely correct, many letters mistranslated
- root cause found: `cv2.flip(frame, 1)` was applied **before** detection, training images were never mirrored, so the live feed was feeding the model a left-right-reversed hand shape it had never seen. Same class of bug as the MobileNetV2 attempt, independently reintroduced.
- fix: detection and prediction moved to run on the raw (unflipped) frame; flip applied only afterward, for display, with landmark points re-mirrored for the on-screen skeleton overlay
- result after fix: mostly accurate with slight decrease of confidence in tricky letters or awkward hand placement
