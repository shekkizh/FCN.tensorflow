import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

class DatasetReader:
    filenames = []
    tf_filenames = tf.convert_to_tensor([])
    label_filenames = []
    image_options = {}

    def __init__(self, records_list, image_options={}, batch_size=1):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        image_augmentation=True/False
        :param predict_dataset: boolean stating whether dataset is for predictions (does not include annotations)
            True/False (default False)
        """
        self.batch_size = batch_size
        self.image_options = image_options
        self.records = {}
        self.records["image"] =  [record['image'] for record in records_list]
        self.records["filename"] =  [record['filename'] for record in records_list]
        if self.image_options.get("predict_dataset", False):
            self.records["annotation"] = [record['annotation'] for record in records_list]
        #tf_records_placeholder = tf.placeholder(self.records)
        if 'annotation' in self.records:
            self.dataset = Dataset.from_tensor_slices((self.records['image'], self.records['filename'],
                                                      self.records['annotation']))
        else:
            self.dataset = Dataset.from_tensor_slices((self.records['image'], self.records['filename']))

        self.dataset = self.dataset.map(self._input_parser)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.repeat()

    def _input_parser(self, image_filename, name, annotation_filename=None):
        image = tf.image.decode_image(tf.read_file(image_filename))
        annotation = None
        if annotation_filename is not None:
            annotation = tf.image.decode_image(tf.read_file(annotation_filename))
        if self.image_options.get("image_augmentation", False):
            return self._augment_image(image, annotation)
        elif self.image_options.get("predict_dataset", False):
            return image
        else:
            return image, annotation

    def _augment_image(self, image, annotation_file=None):
        if annotation_file is not None:
            combined_image_label = tf.concat((image, annotation_file), axis=2)
        else:
            combined_image_label = image
        combined_image_label = tf.image.random_flip_left_right(combined_image_label)
        combined_image_label = tf.image.random_flip_up_down(combined_image_label)
        if annotation_file is not None:
            distorted_image = combined_image_label[:, :, :3]
            distorted_annotation = combined_image_label[:, :, :3]
        else:
            distorted_image = combined_image_label
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        if annotation_file is not None:
            # IDE may not think so, but distorted_annotation is always created before returned
            return distorted_image, distorted_annotation
        else:
            return distorted_image


class TrainVal:
    def __init__(self):
        pass

    @classmethod
    def from_DatasetReaders(cls, train_reader, val_reader):
        train_val = cls()
        train_val.train = train_reader
        train_val.validation = val_reader
        train_val._create_iterator()
        train_val._create_ops()
        return train_val

    @classmethod
    def from_records(cls, train_records, val_records, train_image_options, val_image_options, train_batch_size=1, val_batch_size=1):
        train_reader = DatasetReader(train_records, train_image_options, train_batch_size)
        val_reader = DatasetReader(val_records, val_image_options, val_batch_size)
        return cls.from_DatasetReaders(train_reader, val_reader)

    def _create_iterator(self):
        if not self.train:
            return
        self.iterator = Iterator.from_structure(self.train.output_types,
                                                self.train.output_shapes)

    def _create_ops(self):
        if not self.train or not self.validation:
            return
        self.training_init_op = self.iterator.make_initializer(self.train)
        self.validation_init_op = self.iterator.make_initializer(self.validation)

    def get_iterator(self):
        if not self.iterator:
            self._create_iterator()
        return self.iterator

    def get_ops(self):
        if not self.self.training_init_op or self.validation_init_op:
            self._create_ops()
        return self.self.training_init_op, self.validation_init_op
class SingleDataset:
    def __init__(self):
        pass
    @classmethod
    def from_DatasetReaders(cls, reader):
        dataset = cls()
        dataset.reader = reader
        dataset._create_iterator()
        dataset._create_ops()
        return dataset

    @classmethod
    def from_records(cls, records, image_options, batch_size=1):
        reader = DatasetReader(records, image_options, batch_size)
        return cls.from_DatasetReaders(reader)

    def _create_iterator(self):
        self.iterator = Iterator.from_structure(self.dataset.output_types,
                                                self.dataset.output_shapes)
    def _create_ops(self):
        self.init_op = self.iterator.make_initializer(self.dataset)


