"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
from numpy import newaxis
import scipy.misc as misc
import nibabel as nib
import random
import cv2

class BatchDatset:
    BACKGROUD_CLASS = 0
    NECROSIS_CLASS = 1
    EDEMA_CLASS = 2
    UNKNOWN_CLASS = 3
    ENHANCING_TUMOR_CLASS = 4
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    layers = 155
    brain = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        # self._read_images()

    def _read_images(self):
        # self.__channels = True
        # self.images = np.array(
        #     [self._transform(file_list['input']) for file_list in self.files]
        #     )
        # self.__channels = False
        # self.annotations = np.array(
        #     [np.expand_dims(self._transform(filename['annotation']), axis=2) for filename in self.files])
        self.images, self.annotations = self._match_image_and_mask()
        # print (self.images.shape)      # (, 240, 240, 4)
        # print (self.annotations.shape) # (, 240, 240, 1)

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def _match_image_and_mask(self):
        # match image and mask for single brain case
        # result image shape will be (155, 240, 240, 4)
        # result annot shape will be (155, 240, 240, 1)

        # images = np.zeros((self.layers, 240, 240, 4))
        # masks = np.empty((self.layers, 240, 240, 1))
        images = np.zeros((self.layers, 224, 224, 4))
        masks = np.empty((self.layers, 224, 224, 1))

        t1_file    = [file for file in self.files[self.brain]["input"] if file.endswith("t1.nii.gz")][0]
        t1ce_file  = [file for file in self.files[self.brain]["input"] if file.endswith("t1ce.nii.gz")][0]
        t2_file    = [file for file in self.files[self.brain]["input"] if file.endswith("t2.nii.gz")][0]
        flair_file = [file for file in self.files[self.brain]["input"] if file.endswith("flair.nii.gz")][0]
        mask_file   = self.files[self.brain]["mask"][0]

        t1_img_array    = nib.load(t1_file).get_fdata()
        t1ce_img_array  = nib.load(t1ce_file).get_fdata()
        t2_img_array    = nib.load(t2_file).get_fdata()
        flair_img_array = nib.load(flair_file).get_fdata()
        mask_array      = nib.load(mask_file).get_fdata()

        mask_regrouped   = self._regroup_mask_array(mask_array)
        
        for z in range(self.layers):
            tmp1 = np.concatenate((t1_img_array[:, :, z][:, :, newaxis], t1ce_img_array[:, :, z][:, :, newaxis]), axis = 2)
            tmp2 = np.concatenate((t2_img_array[:, :, z][:, :, newaxis], flair_img_array[:, :, z][:, :, newaxis]), axis = 2)
            concatenate_img_data = np.concatenate((tmp1, tmp2), axis=2)
            
            if self.image_options.get("resize", False) and self.image_options["resize"]:
                resize_size = int(self.image_options["resize_size"])
                resize_image = cv2.resize(concatenate_img_data, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
                resize_mask = cv2.resize(mask_regrouped[:, :, z][:, :, newaxis], (resize_size, resize_size), interpolation=cv2.INTER_AREA)
                resize_mask = resize_mask[:, :, newaxis]
            else:
                resize_image = concatenate_img_data
                resize_mask = mask_regrouped[:, :, z][:, :, newaxis]

            images[z] = resize_image
            masks[z] = resize_mask

        # num_dataset = len(self.files) * self.layers
        # images = np.zeros((num_dataset, 240, 240, 4))
        # masks = np.empty((num_dataset, 240, 240, 1))
        # count = 0
    
        # for brain in records:
        #     t1_file    = [file for file in brain["input"] if file.endswith("t1.nii.gz")][0]
        #     t1ce_file  = [file for file in brain["input"] if file.endswith("t1ce.nii.gz")][0]
        #     t2_file    = [file for file in brain["input"] if file.endswith("t2.nii.gz")][0]
        #     flair_file = [file for file in brain["input"] if file.endswith("flair.nii.gz")][0]
        #     mask_file   = brain["mask"][0]

        #     t1_img_array    = nib.load(t1_file).get_fdata()
        #     t1ce_img_array  = nib.load(t1ce_file).get_fdata()
        #     t2_img_array    = nib.load(t2_file).get_fdata()
        #     flair_img_array = nib.load(flair_file).get_fdata()
        #     mask_array  = nib.load(mask_file).get_fdata()

        #     mask_regrouped   = regroup_mask_array(mask_array)
        
        #     for z in range(self.layers):
        #         tmp1 = np.concatenate((t1_img_array[:, :, z][:, :, newaxis], t1ce_img_array[:, :, z][:, :, newaxis]), axis = 2)
        #         tmp2 = np.concatenate((t2_img_array[:, :, z][:, :, newaxis], flair_img_array[:, :, z][:, :, newaxis]), axis = 2)
        #         concatenate_img_data = np.concatenate((tmp1, tmp2), axis=2)
        #         images[count * self.layers + z] = concatenate_img_data
        #         masks[count * self.layers + z] = mask_regrouped[:, :, z][:, :, newaxis]
            
        #     count += 1

        return images, masks

    def _set_random_brain(self):
        random_brain = random.choice(list(range(len(self.files))))
        print(f"set random brain: {random_brain}")
        self.brain = random_brain

    def _regroup_mask_array(self, mask_array):
        regrouped = mask_array
        regrouped[regrouped == self.UNKNOWN_CLASS] = 0
        regrouped[regrouped != self.BACKGROUD_CLASS] = 1
        return regrouped

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        self._set_random_brain()
        self._read_images()
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
