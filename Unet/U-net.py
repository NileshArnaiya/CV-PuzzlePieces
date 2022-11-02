# %%
!pip install segmentation-models
!pip install tensorflow==2.1.0
!pip install keras==2.3.1

# %%
#For live loss function updates
#!pip install livelossplot

# %%
%matplotlib inline

# %%
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


# %%
import tensorflow as tf
import segmentation_models as sm
BACKBONE = 'vgg16'
preprocess_input = sm.get_preprocessing(BACKBONE)


# %%
#print(os.listdir("membrane/train"))

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 128 #Resize images (height  = X, width = Y)
SIZE_Y = 128
splits = KFold(n_splits=6,shuffle=True,random_state=42)
#Capture training image info as a list
train_images = []

for directory_path in glob.glob("images-1024x768\images"):
    print("Seee")
    print(directory_path)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing        
train_images = np.array(train_images, dtype='float32')

#Capture mask/label info as a list
train_masks = [] 
print("Nothing???")
for directory_path in glob.glob("images-1024x768/masks"):
    print("Hereeee")
    print(directory_path)
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        train_masks.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks,  dtype='float32')


# %%
train_images.dtype

# %%
train_masks.dtype

# %%
#Use customary x_train and y_train variables
X = train_images
Y = train_masks
Y = np.expand_dims(Y, axis=3) #May not be necessary.. leftover from previous code 


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.18, random_state=1) # 0.25 x 0.8 = 0.2.. 0.2 * 0.85 = 

# x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)

# preprocess input
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_val = preprocess_input(x_val)

# print(x_train)

# %%
# define model
sm.set_framework('tf.keras')
sm.framework()

# with pre-trained backbones and encoder weights from imagenet

model = sm.Unet(BACKBONE, encoder_weights='imagenet', activation='sigmoid')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

print(model.summary())


# %%


# %%
# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
#from livelossplot import PlotLossesKeras
#Include this as callback., but slows the training (callbacks=[PlotLossesKeras()],)
# print(model)
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)


x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.float32)


# k - fold cross validation see

dataset = ConcatDataset([x_train, x_test])

k=6
splits=KFold(n_splits=k,shuffle=True,random_state=42)

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)


print(train_sampler)
print(train_loader)
# for i, j in train_loader:
#    print("I is ", i, " J is ", j)

print("check again")
x_val = tf.cast(x_val, tf.float32)
y_val = tf.cast(y_val, tf.float32)

model.fit(
   x=x_train,
   y=y_train,
   batch_size=2,
   epochs=5,
   verbose=1,
   validation_data=(x_val, y_val),
)
print("should work now")
accuracy = model.evaluate(x_test, y_test)
# possibly something to do here with validation data 
print('accuracy is ', accuracy)


# %%
model.save('/data/file.h5')

# %%
from tensorflow import keras
model = keras.models.load_model('/data/file.h5', compile=False)

# %%
#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread('images_prepped_val/image-26.png', cv2.IMREAD_COLOR)      
print(test_img)
plt.imshow(test_img)
test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
plt.imshow(test_img, cmap='gray')
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)

# %%
#View and Save segmented image
prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='inferno')
plt.axis("off")

plt.show()
#plt.imsave('images/test_images/segmented.jpg', prediction_image, cmap='gray')


# %%
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
print(fpr, tpr)
roc_auc = auc(fpr,tpr)

# %%
y_score1 = clf_tree.predict_proba(X_test)[:,1]
y_score2 = clf_reg.predict_proba(X_test)[:,1]

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)

# %%



