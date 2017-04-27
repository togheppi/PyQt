import keras

if keras.backend.backend() == 'tensorflow':
    import tensorflow as tf

class TensorBoard(keras.callbacks.Callback):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):
        super(TensorBoard, self).__init__()
        if keras.backend.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.global_step = 0

    def set_model(self, model):
        self.model = model
        self.sess = keras.backend.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)
                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def on_train_begin(self, logs=None):
        self.global_step = 0

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.validation_data and self.histogram_freq:
            if batch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [keras.backend.learning_phase()]
                else:
                    val_data = self.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, self.global_step)

        for k in self.params['metrics']:
            if k in logs:
                # self.log_values.append((k, logs[k]))
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = logs[k]
                summary_value.tag = k
                self.writer.add_summary(summary, self.global_step)

        self.writer.flush()
        self.global_step += 1

    def on_train_end(self, _):
        self.writer.close()