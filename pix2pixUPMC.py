from pathlib import Path
from natsort import natsorted
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import tensorflow as tf
import time
import datetime
from IPython import display
import numpy as np
import albumentations.augmentations.functional as F
import os

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
EPOCHS = 150

pathStringImages = '/home/scratch/cgmorale/task_segmentation labeling-2021_06_30_20_17_00-segmentation mask 1.1/JPEGImages/'
pathStringMasks = '/home/scratch/cgmorale/task_segmentation labeling-2021_06_30_20_17_00-segmentation mask 1.1/SegmentationClass/'

pathStringImagesList = []
pathStringMasksList = []
pMask = Path(pathStringMasks)
pImages = Path(pathStringImages)

for i in pMask.iterdir():
    pathStringMasksList.append(i)

for j in pImages.iterdir():
    pathStringImagesList.append(j)
    
sortedMasks = natsorted(pathStringMasksList)
sortedImages = natsorted(pathStringImagesList)
sortedImages = sortedImages[0:265]

trainingImages = []
trainingMasks = []
testingImages = []
testingMasks= []

trainingI = sortedImages[0:200]
trainingM = sortedMasks[0:200]

testingI = sortedImages[201:265]
testingM = sortedMasks[201:265]


for i in range(len(trainingI)):
    trainingImages.append(sitk.ReadImage(str(trainingI[i])))
    trainingMasks.append(sitk.ReadImage(str(trainingM[i])))
    
for j in range(len(testingI)):
    testingImages.append(sitk.ReadImage(str(testingI[j])))
    testingMasks.append(sitk.ReadImage(str(testingM[j])))

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image


def get_training_augmentation():
    train_transform = [
        albu.ColorJitter(brightness=0.2, contrast=0.8, saturation=0.7, hue=0.2, always_apply=False, p=0.8),
        albu.RandomRotate90(p=0.3),
        albu.Flip(p=0.3),
        albu.Transpose(p=0.3),
#         albu.OneOf(
#         [
#             albu.OpticalDistortion(p=0.5),
#             albu.IAAPiecewiseAffine(p=0.5)
#         ], p = 0.2),
        albu.RandomContrast(p = 0.4),
#         albu.RandomGamma(p=0.2),
        albu.RandomBrightness(p=0.4),
#         albu.OneOf(
#             [albu.ElasticTransform(alpha = 120,
#                     sigma = 120 * 0.05,
#                     alpha_affine = 120 * 0.03), albu.GridDistortion()],p = 0.5),
        
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=IMG_HEIGHT, min_width=IMG_WIDTH, always_apply=True, border_mode=0),
        albu.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.1,
        ),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.1,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class Dataset(BaseDataset):
    """
    Args:
        images (SimpleITK Images): dataset images 
        masks (SimpleITK Images): ground truth masks for the corresponding images 
    """
    
    def __init__(self, images, masks,preprocessing=None, augmentation = None):
        #self.ids = os.listdir('/zfsauton/project/upmc_echo/CAMUS/training/')
        self.ids = images
        self.images_fps = images
        self.masks_fps = masks
        self.preprocessing = preprocessing
        self.augmentation = augmentation
    def __getitem__(self, i):
        
        # read data
        imgArray = sitk.GetArrayFromImage(self.images_fps[i])
        # this will give an image of shape [1 x W x H]
        # we need to change it into a size of [W x H x 3] for this model
        img3 = np.zeros([imgArray.shape[0],imgArray.shape[1],3])
        img3[:,:,0] = imgArray[:,:]
        img3[:,:,1] = imgArray[:,:]
        img3[:,:,2] = imgArray[:,:]
        img3 = img3.astype(np.uint8)
        resized_image = albu.resize(img3, height=256, width=256)
        
        maskArray = sitk.GetArrayFromImage(self.masks_fps[i])
        #maskimg = np.zeros([maskArray.shape[0], maskArray.shape[1],1])
        #maskimg[:,:,0] = maskArray[:,:]
        resized_mask = albu.resize(maskArray, height=256, width = 256)
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=resized_image, mask=resized_mask)
            resized_image, resized_mask= sample['image'], sample['mask']
            
        #apply augmentation pipeline
        if self.augmentation:
            sample = self.augmentation(image = resized_image, mask = resized_mask)
            resized_image, resized_mask = sample['image'], sample['mask']
        

        return resized_image, resized_mask
        
    def __len__(self):
        return len(self.ids)

augmented_dataset = Dataset(
    trainingImages, 
    trainingMasks, 
    augmentation=get_training_augmentation()
)

testing_dataset= Dataset(testingImages,testingMasks)

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  target = tf.image.convert_image_dtype(target, dtype=tf.float32)
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image[tf.newaxis, ...], training=True)

    disc_real_output = discriminator([input_image[tf.newaxis, ...], target[tf.newaxis, ...]], training=True)
    
    disc_generated_output = discriminator([input_image[tf.newaxis, ...], gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    #example_input, example_target = test_ds[2]
    #generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Training step
    n = 0
    for i in range(len(train_ds)):
        input_image, target = train_ds[i]
        input_image, real_image = resize(target, input_image, 256, 256)
        train_step(real_image, input_image, epoch)

    # Saving (checkpointing) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix=checkpoint_prefix)

fit(augmented_dataset, EPOCHS, testing_dataset)



    