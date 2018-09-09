import random, cv2, time, threading, sys, Queue

import numpy as np
#from joblib import Parallel, delayed
from multiprocessing import Process, Pool
from multiprocessing import Queue as mQueue
import tensorflow as tf
from codification import *
from codification import encode, check_distance

class Dataset(object):
    def __init__(self, splited_keys, images, datasets, config_input, augmenter, perception_interface):
        # sample inputs
        # splited_keys: _splited_keys_train[i_labels_per_division][i_steering_bins_perc][a list of keys]
        # images: [i_sensor][i_file_number] = (lastidx, lastidx + x.shape[0], x)
        # datasets: [i_target_name] = dim*batch matrix, where batch=#all_samples
        # config_input: configInputs
        # augmenter: config_input.augment

        # save the inputs
        self._splited_keys = splited_keys
        self._images = images
        self._targets = np.concatenate(tuple(datasets), axis=1)  # Cat the datasets, The shape is totalnum*totaldim
        self._config = config_input
        self._augmenter = augmenter

        self._batch_size = config_input.batch_size

        # prepare all the placeholders: 3 sources: _queue_image_input, _queue_targets, _queue_inputs
        self._queue_image_input = tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                    config_input.feature_input_size[0],
                                                                    config_input.feature_input_size[1],
                                                                    config_input.feature_input_size[2]])

        self._queue_shapes = [self._queue_image_input.shape]

        # config.targets_names: ['wp1_angle', 'wp2_angle', 'Steer', 'Gas', 'Brake', 'Speed']
        self._queue_targets = []
        for i in range(len(self._config.targets_names)):
            self._queue_targets.append(tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                         self._config.targets_sizes[i]]))
            self._queue_shapes.append(self._queue_targets[-1].shape)

        # self.inputs_names = ['Control', 'Speed']
        self._queue_inputs = []
        for i in range(len(self._config.inputs_names)):
            self._queue_inputs.append(tf.placeholder(tf.float32, shape=[config_input.batch_size,
                                                                        self._config.inputs_sizes[i]]))
            self._queue_shapes.append(self._queue_inputs[-1].shape)

        self._queue = tf.FIFOQueue(capacity=config_input.queue_capacity,
                                   dtypes=[tf.float32] + [tf.float32] * (len(self._config.targets_names) + len(self._config.inputs_names)),
                                   shapes=self._queue_shapes)
        self._enqueue_op = self._queue.enqueue([self._queue_image_input] + self._queue_targets + self._queue_inputs)
        self._dequeue_op = self._queue.dequeue()

        #self.parallel_workers = Parallel(n_jobs=8, backend="threading")
        self.input_queue = mQueue(5)
        self.output_queue = mQueue(5)

        self.perception_interface = perception_interface


    def get_batch_tensor(self):
        return self._dequeue_op

    def sample_positions_to_train(self, number_of_samples, splited_keys):
        out_splited_keys = []
        for sp in splited_keys:
            if len(sp)>0:
                out_splited_keys.append(sp)

        return np.random.choice(range(len(out_splited_keys)),
                                size=number_of_samples,
                                replace=True), \
               out_splited_keys

    # Used by next_batch, for each of the control block,
    def datagen(self, batch_size, number_control_divisions):
        # typical input: batch_size, number_control_divisions=3, since 3 blocks
        # Goal: uniformly select from different control signals (group), different steering percentiles.
        generated_ids = np.zeros((batch_size, ), dtype='int32')

        count = 0
        to_be_decoded = [[] for _ in range(len(self._images))]
        for control_part in range(0, number_control_divisions):
            num_to_sample = int(batch_size // number_control_divisions)
            if control_part == (number_control_divisions - 1):
                num_to_sample = batch_size - (number_control_divisions - 1) * num_to_sample

            sampled_positions, non_empty_split_keys = self.sample_positions_to_train(num_to_sample,
                                                               self._splited_keys[control_part])

            for outer_n in sampled_positions:
                i = random.choice(non_empty_split_keys[outer_n])
                for isensor in range(len(self._images)):
                    # fetch the image from the h5 files
                    per_h5_len = self._images[isensor][0].shape[0]
                    ibatch = i // per_h5_len
                    iinbatch = i % per_h5_len
                    imencoded = self._images[isensor][ibatch][iinbatch]
                    to_be_decoded[isensor].append(imencoded)

                generated_ids[count] = i
                count += 1

        return to_be_decoded, generated_ids

    """Return the next `batch_size` examples from this data set."""

    # Used by enqueue
    def next_batch(self, sensors, generated_ids):
        # generate unbiased samples;
        # apply augmentation on sensors and segmentation labels
        # normalize images
        # fill in targets and inputs. with reasonable valid condition checking

        batch_size = self._batch_size

        # Get the images -- Perform Augmentation!!!
        for i in range(len(sensors)):
            # decode each of the sensor in parallel
            func = lambda x: cv2.imdecode(x, 1)
            if hasattr(self._config, "hack_resize_image"):
                height, width = self._config.hack_resize_image
                func_previous = func
                func = lambda x: cv2.resize(func_previous(x), (width, height))

            if hasattr(self._config, "hack_faster_aug"):
                func_previous = func
                func = lambda x: func_previous(x)[::2, ::2, :]

            # func = delayed(func)
            # results = self.parallel_workers(func(x) for x in to_be_decoded[isensor])
            results = []
            for x in sensors[i]:
                results.append(func(x))
            sensors[i] = np.stack(results, 0)

            # from bgr to rgb
            sensors[i] = sensors[i][:, :, :, ::-1]

            if self._augmenter[i] != None:
                sensors[i] = self._augmenter[i].augment_images(sensors[i])

            if self._config.image_as_float[i]:
                sensors[i] = sensors[i].astype(np.float32)
            if self._config.sensors_normalize[i]:
                sensors[i] /= 255.0

        # self._targets is the targets variables concatenated
        # Get the targets
        target_selected = self._targets[generated_ids, :]
        target_selected = target_selected.T

        # prepare the output targets, and inputs
        targets = []
        for i in range(len(self._config.targets_names)):
            targets.append(np.zeros((batch_size, self._config.targets_sizes[i])))
        inputs = []
        for i in range(len(self._config.inputs_names)):
            inputs.append(np.zeros((batch_size, self._config.inputs_sizes[i])))

        for ibatch in range(0, batch_size):
            for itarget in range(len(self._config.targets_names)):
                # Yang: This is assuming that all target names has size 1
                k = self._config.variable_names.index(self._config.targets_names[itarget])
                targets[itarget][ibatch] = target_selected[k, ibatch]
                this_name = self._config.targets_names[itarget]

                if this_name == "Speed":
                    # Yang: speed_factor is normalizing the speed
                    targets[itarget][ibatch] /= self._config.speed_factor / 3.6
                elif this_name == "Gas":
                    # Yang: require Gas >=0
                    targets[itarget][ibatch] = max(0, targets[itarget][ibatch])
                elif this_name == "Brake":
                    # Yang: require 0<=Brake<=1
                    targets[itarget][ibatch] = min(1.0, max(0, targets[itarget][ibatch]))

            for iinput in range(len(self._config.inputs_names)):
                k = self._config.variable_names.index(self._config.inputs_names[iinput])
                this_name = self._config.inputs_names[iinput]

                if this_name == "Control":
                    inputs[iinput][ibatch] = encode(target_selected[k, ibatch])
                elif this_name == "Speed":
                    inputs[iinput][ibatch] = target_selected[k, ibatch] / self._config.speed_factor * 3.6
                elif this_name == "Distance":
                    inputs[iinput][ibatch] = check_distance(target_selected[k, ibatch])
                else:
                    raise ValueError()

        # change the output sensors variable
        sensors = np.concatenate(sensors, axis=0)

        return sensors, targets, inputs

    # Used by enqueue
    def process_run(self, sess, data_loaded):
        reshaped = data_loaded[0]
        nB, nH, nW, nC = reshaped.shape
        num_sensors = len(self._config.sensor_names)
        reshaped = np.reshape(reshaped, (num_sensors, nB//num_sensors, nH, nW, nC))
        reshaped = np.transpose(reshaped, (1, 2, 0, 3, 4))
        # now has shape nB//num_sensors, nH, num_sensors, nW, nC
        reshaped = np.reshape(reshaped, (nB//num_sensors, nH, num_sensors*nW, nC))

        if hasattr(self._config, "add_gaussian_noise"):
            std = self._config.add_gaussian_noise
            print("!!!!!!!!!!!!!!!!!!adding gaussian noise", std)
            reshaped += np.random.normal(0, std, reshaped.shape)

        queue_feed_dict = {self._queue_image_input: reshaped}  # images we already put by default

        for i in range(len(self._config.targets_names)):
            queue_feed_dict.update({self._queue_targets[i]: data_loaded[1][i]})

        for i in range(len(self._config.inputs_names)):
            queue_feed_dict.update({self._queue_inputs[i]: data_loaded[2][i]})

        sess.run(self._enqueue_op, feed_dict=queue_feed_dict)

    def __getstate__(self):
        """Return state values to be pickled."""
        print("pickling")
        return (self._splited_keys, self._targets, self._config, self._augmenter, self._batch_size)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        print("unpickling")
        self._splited_keys, self._targets, self._config, self._augmenter, self._batch_size = state

    def _thread_disk_reader(self):
        while True:
            #start = time.time()
            sensors, generated_ids = self.datagen(self._batch_size, len(self._splited_keys))
            self.input_queue.put((sensors, generated_ids))
            #print('putting ele into input queue, cost', time.time()-start)

    @staticmethod
    def _thread_decode_augment(dataset, input_queue, output_queue):
        while True:
            sensors, generated_ids = input_queue.get()
            out = dataset.next_batch(sensors, generated_ids)
            output_queue.put(out)

    def start_multiple_decoders_augmenters(self):
        n_jobs = 6
        for i in range(n_jobs):
            p = Process(target=self._thread_decode_augment, args=(self, self.input_queue, self.output_queue))
            #p = threading.Thread(target=self._thread_decode_augment, args=(self, self.input_queue, self.output_queue))
            p.start()

    def _thread_perception_splitting(self, input_queue):
        while True:
            one_batch = input_queue.get()
            self.output_remaining_queue.put(one_batch[1:])
            self.output_image_queue.put(one_batch[0])

    def _thread_perception_concat(self, perception_output):
        while True:
            remain = self.output_remaining_queue.get()
            image_feature = perception_output.get()
            self.final_output_queue.put([image_feature, remain[0], remain[1]])

    def _thread_feed_dict(self, sess, output_queue):
        while True:
            #start = time.time()
            one_batch = output_queue.get()
            self.process_run(sess, one_batch)
            #print("fetched one output, cost ", time.time()-start)

    def start_all_threads(self, sess):
        t = threading.Thread(target=self._thread_disk_reader)
        t.isDaemon()
        t.start()

        self.start_multiple_decoders_augmenters()

        if self._config.use_perception_stack:
            self.output_image_queue = Queue.Queue(5)
            self.output_remaining_queue = Queue.Queue(5)
            t = threading.Thread(target=self._thread_perception_splitting, args=(self.output_queue,))
            t.start()

            perception_output = self.perception_interface.compute_async_thread_channel(self.output_image_queue)

            self.final_output_queue = Queue.Queue(5)
            t = threading.Thread(target=self._thread_perception_concat, args=(perception_output,))
            t.start()
            output_queue = self.final_output_queue
        else:
            output_queue = self.output_queue

        t = threading.Thread(target=self._thread_feed_dict, args=(sess, output_queue))
        t.isDaemon()
        t.start()
