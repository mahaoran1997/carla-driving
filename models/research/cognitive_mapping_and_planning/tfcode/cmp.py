# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for setting up the network for CMP.

Sets up the mapper and the planner.
"""

import sys, os, numpy as np
import matplotlib.pyplot as plt
import copy
import argparse, pprint
import time


import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim import arg_scope

import logging
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from src import utils 
import src.file_utils as fu
import tfcode.nav_utils as nu 
import tfcode.cmp_utils as cu 
import tfcode.cmp_summary as cmp_s
from tfcode import tf_utils

value_iteration_network = cu.value_iteration_network
rotate_preds            = cu.rotate_preds
deconv                  = cu.deconv
get_visual_frustum      = cu.get_visual_frustum
fr_v2                   = cu.fr_v2

setup_train_step_kwargs = nu.default_train_step_kwargs
compute_losses_multi_or = nu.compute_losses_multi_or

get_repr_from_image     = nu.get_repr_from_image

_save_d_at_t            = nu.save_d_at_t
_save_all               = nu.save_all
_eval_ap                = nu.eval_ap
_eval_dist              = nu.eval_dist
_plot_trajectories      = nu.plot_trajectories

_vis_readout_maps       = cmp_s._vis_readout_maps
_vis                    = cmp_s._vis
_summary_vis            = cmp_s._summary_vis
_summary_readout_maps   = cmp_s._summary_readout_maps
_add_summaries          = cmp_s._add_summaries

def _inputs(problem):
  # Set up inputs.
  with tf.name_scope('inputs'):
    inputs = []
    if problem.input_type == 'vision':
      # Multiple images from an array of cameras.
      #print(problem.aux_delta_thetas)
      inputs.append(('imgs', tf.float32, 
                     (None, len(problem.aux_delta_thetas)+1,
                      problem.img_height, problem.img_width,
                      problem.img_channels)))
    elif problem.input_type == 'analytical_counts':
      for i in range(len(problem.map_crop_sizes)):
        inputs.append(('analytical_counts_{:d}'.format(i), tf.float32, 
                      (None, problem.map_crop_sizes[i],
                       problem.map_crop_sizes[i], problem.map_channels)))

    if problem.outputs.readout_maps: 
      for i in range(len(problem.readout_maps_crop_sizes)):
        inputs.append(('readout_maps_{:d}'.format(i), tf.float32, 
                      (None,
                       problem.readout_maps_crop_sizes[i],
                       problem.readout_maps_crop_sizes[i],
                       problem.readout_maps_channels)))

    inputs.append(('running_sum_num', tf.float32, (None, problem.map_crop_sizes[0],
                      problem.map_crop_sizes[0], problem.map_channels)))

    inputs.append(('incremental_locs', tf.float32, 
                   (None, 2)))
    inputs.append(('incremental_thetas', tf.float32, 
                   (None, 2)))   #!!!mhr: 1->2


    inputs.append(('command', tf.float32, (None, 4)))
    inputs.append(('speed', tf.float32, (None, 1)))  #!!!!mhr: new adding
    #inputs.append(('measurements', tf.float32, (None, 3)))  #!!!!mhr: new adding
    
    #print(inputs)
    step_input_data, _ = tf_utils.setup_inputs(inputs)
    inputs = []
    inputs.append(('action', tf.float32, (None, problem.num_actions)))
    train_data, _ = tf_utils.setup_inputs(inputs)
    train_data.update(step_input_data)
  return step_input_data, train_data 

def readout_general(multi_scale_belief, num_neurons, strides, layers_per_block,
                    kernel_size, batch_norm_is_training_op, wt_decay):
  multi_scale_belief = tf.stop_gradient(multi_scale_belief)
  with tf.variable_scope('readout_maps_deconv'):
    x, outs = deconv(multi_scale_belief, batch_norm_is_training_op,
                     wt_decay=wt_decay, neurons=num_neurons, strides=strides,
                     layers_per_block=layers_per_block, kernel_size=kernel_size,
                     conv_fn=slim.conv2d_transpose, offset=0,
                     name='readout_maps_deconv')
    probs = tf.sigmoid(x)
  return x, probs


def running_combine(incremental_locs, incremental_thetas, previous_sum_num, map_size, batch_size):
  # incremental_locs is B x 2
  # incremental_thetas is B x 2  #1->2
  # previous_sum_num etc is B x H x W x C    #1 -> N

  with tf.name_scope('combine_{:d}'.format(batch_size)):
    running_sum_nums_ = []
    incremental_locs_ = tf.unstack(incremental_locs, axis=0, num=batch_size)
    incremental_thetas_ = tf.unstack(incremental_thetas, axis=0, num=batch_size)
    incremental_thetas_2 = []
    for thetas in incremental_thetas_:
      incremental_thetas_2.append(tf.unstack(thetas, axis=0))
    running_sum_num = tf.unstack(previous_sum_num, axis=0, num=batch_size)
    #print(num_steps)

    for i in range(batch_size):
      # Rotate the previous running_num and running_denom
      inc_theta_0 = tf.reshape(incremental_thetas_[i][0], shape = [-1, 1])
      inc_theta_1 = tf.reshape(incremental_thetas_[i][1], shape = [-1, 1])
      inc_loc = tf.reshape(incremental_locs_[i], shape = [-1, 2])
      #print incremental_locs_[i]
      #print inc_theta_0
      tmp_running_sum_num = rotate_preds(
          inc_loc, inc_theta_0, map_size,
          [running_sum_num[i]],
          output_valid_mask=False)[0][0]
      tmp_running_sum_num = rotate_preds(
          tf.constant(0.0, shape=[inc_loc.shape[0], 2]), inc_theta_1, map_size,
          [tmp_running_sum_num],
          output_valid_mask=False)[0][0]
      print(i)
      print(tmp_running_sum_num.get_shape().as_list())
      running_sum_nums_.append(tmp_running_sum_num)

    running_sum_nums = tf.stack(running_sum_nums_, axis=0)
    return running_sum_nums

def get_map_from_images(imgs, mapper_arch, task_params, freeze_conv, wt_decay,
                        is_training, batch_norm_is_training_op, num_maps,
                        split_maps=True):
  # Hit image with a resnet.
  n_views = len(task_params.aux_delta_thetas) + 1
  print(n_views)
  print(task_params.aux_delta_thetas)
  print(task_params.data_augment)

  out = utils.Foo()

  images_reshaped = tf.reshape(imgs, 
      shape=[-1, task_params.img_height,
             task_params.img_width,
             task_params.img_channels], name='re_image')

  x, out.vars_to_restore = get_repr_from_image(
      images_reshaped, task_params.modalities, task_params.data_augment,
      mapper_arch.encoder, freeze_conv, wt_decay, is_training)

  # Reshape into nice things so that these can be accumulated over time steps
  # for faster backprop.
  sh_before = x.get_shape().as_list()
  print('------------------------------------------')
  print(sh_before)
  out.encoder_output = tf.reshape(x, shape=[-1, n_views] + sh_before[1:])
  print(out.encoder_output.get_shape().as_list())
  
  x = tf.reshape(out.encoder_output, shape=[-1] + sh_before[1:])
  print(x.get_shape().as_list())

  # Add a layer to reduce dimensions for a fc layer.
  if mapper_arch.dim_reduce_neurons > 0:
    ks = 1
    neurons = mapper_arch.dim_reduce_neurons
    init_var = np.sqrt(2.0/(ks**2)/neurons)
    batch_norm_param = mapper_arch.batch_norm_param
    batch_norm_param['is_training'] = batch_norm_is_training_op
    batch_norm_param['decay'] = 0.95 #mhr: 0.999
    print('adfasdf')
    print(x.get_shape().as_list())
    out.conv_feat = slim.conv2d(x, neurons, kernel_size=ks, stride=1,
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_param,
                    padding='SAME', scope='dim_reduce',
                    weights_regularizer=slim.l2_regularizer(wt_decay),
                    weights_initializer=tf.random_normal_initializer(stddev=init_var))
    #print(out.conv_feat.get_shape().as_list())
    reshape_conv_feat = slim.flatten(out.conv_feat)
    sh = reshape_conv_feat.get_shape().as_list()
    out.reshape_conv_feat = tf.reshape(reshape_conv_feat, shape=[-1, sh[1]*n_views])

  with tf.variable_scope('fc'):
    # Fully connected layers to compute the representation in top-view space.
    fc_batch_norm_param = {'center': True, 'scale': True, 
                           'activation_fn':tf.nn.relu,
                           'is_training': batch_norm_is_training_op,
                           'decay' : 0.95} #mhr: 0.9
    f = out.reshape_conv_feat
    out_neurons = (mapper_arch.fc_out_size**2)*mapper_arch.fc_out_neurons
    neurons = mapper_arch.fc_neurons + [out_neurons]
    f, _ = tf_utils.fc_network(f, neurons=neurons, wt_decay=wt_decay,
                               name='fc', offset=0,
                               batch_norm_param=fc_batch_norm_param,
                               is_training=is_training,
                               dropout_ratio=mapper_arch.fc_dropout)
    f = tf.reshape(f, shape=[-1, mapper_arch.fc_out_size,
                             mapper_arch.fc_out_size,
                             mapper_arch.fc_out_neurons], name='re_fc')

  # Use pool5 to predict the free space map via deconv layers.
  with tf.variable_scope('deconv'):
    x, outs = deconv(f, batch_norm_is_training_op, wt_decay=wt_decay,
                     neurons=mapper_arch.deconv_neurons,
                     strides=mapper_arch.deconv_strides,
                     layers_per_block=mapper_arch.deconv_layers_per_block,
                     kernel_size=mapper_arch.deconv_kernel_size,
                     conv_fn=slim.conv2d_transpose, offset=0, name='deconv')

  # Reshape x the right way.
  out.deconv_output = x
  return out









#-----------------------------CIL---------------------------------------
def weight_ones(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    return tf.Variable(initial)


def weight_xavi_init(shape, name):
    initial = tf.get_variable(name=name, shape=shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    return initial


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Network(object):

    def __init__(self, dropout, image_shape):
        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_' + str(self._count_conv))
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv))

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool' + str(self._count_pool))

    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=False,
                                            updates_collections=None,
                                            scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x):
        print("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout' + str(self._count_dropouts))

        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc))

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)

            return self.activation(x)

    def fc_block(self, x, output_size):
        print(" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x)
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features







#-----------------------------CIL---------------------------------------












def setup_to_run(m, args, is_training, batch_norm_is_training, summary_mode):
  assert(args.arch.multi_scale), 'removed support for old single scale code.'
  # Set up the model.
  tf.set_random_seed(args.solver.seed)
  task_params = args.navtask.task_params

  batch_norm_is_training_op = \
      tf.placeholder_with_default(batch_norm_is_training, shape=[],
                                  name='batch_norm_is_training_op') 

  # Setup the inputs
  m.input_tensors = {}
  m.train_ops = {}
  m.input_tensors['step'], m.input_tensors['train'] = \
      _inputs(task_params)
  
  m.init_fn = None

  if task_params.input_type == 'vision':
    m.vision_ops = get_map_from_images(
        m.input_tensors['step']['imgs'], args.mapper_arch,
        task_params, args.solver.freeze_conv,
        args.solver.wt_decay, is_training, batch_norm_is_training_op,
        num_maps=len(task_params.map_crop_sizes))

    # Load variables from snapshot if needed.
    if args.solver.pretrained_path is not None:
      m.init_fn = slim.assign_from_checkpoint_fn(args.solver.pretrained_path,
                                                 m.vision_ops.vars_to_restore)

    # Set up caching of vision features if needed.
    if args.solver.freeze_conv:
      m.train_ops['step_data_cache'] = [m.vision_ops.encoder_output]
    else:
      m.train_ops['step_data_cache'] = []

    # Set up blobs that are needed for the computation in rest of the graph.
    m.ego_map_ops = m.vision_ops.deconv_output
    print('ego_maps:')
    print(m.ego_map_ops.get_shape().as_list())

  num_steps = task_params.num_steps
  num_goals = task_params.num_goals

  map_crop_size_ops = []
  for map_crop_size in task_params.map_crop_sizes:
    map_crop_size_ops.append(tf.constant(map_crop_size, dtype=tf.int32, shape=(2,)))

  with tf.name_scope('check_size'):
    is_single_step = tf.equal(tf.unstack(tf.shape(m.ego_map_ops), num=4)[0], task_params.batch_size)

  
  
  updated_state = []; state_names = []

  #for i in range(len(task_params.map_crop_sizes)):
  map_crop_size = task_params.map_crop_sizes[0]
  #with tf.variable_scope('scale_{:d}'.format(i)): 
  # Accumulate the map.
  fn = lambda ns: running_combine(
          m.input_tensors['step']['incremental_locs'] * task_params.map_scales[0],
          m.input_tensors['step']['incremental_thetas'],
          m.input_tensors['step']['running_sum_num'],
          map_crop_size, ns)

  running_sum_num_pre = tf.cond(is_single_step, lambda: fn(task_params.batch_size), lambda: fn(task_params.batch_size*task_params.num_steps))
  

  with tf.name_scope('fusion'):
    with tf.name_scope('concat'):
      pre_sh = running_sum_num_pre.get_shape().as_list()
      sh = [-1]+pre_sh[-3:]
      reshape_sum_num = tf.reshape(running_sum_num_pre, shape=sh)
      after_concat = tf.concat([reshape_sum_num, m.ego_map_ops], 3)
    
    init_var = np.sqrt(2.0/(args.mapper_arch.conv_ks**2)/args.mapper_arch.conv_neurons)
    batch_norm_param = args.mapper_arch.batch_norm_param
    batch_norm_param['is_training'] = batch_norm_is_training_op
    batch_norm_param['decay'] = 0.95  #mhr: 0.999
    running_sum_num = slim.conv2d(after_concat, args.mapper_arch.conv_neurons, 
                      kernel_size=args.mapper_arch.conv_ks, stride=args.mapper_arch.conv_strides,
                      normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_param,
                      padding='SAME', scope='fusion_conv',
                      weights_regularizer=slim.l2_regularizer(args.solver.wt_decay),
                      weights_initializer=tf.random_normal_initializer(stddev=init_var))

    updated_state += [running_sum_num]
    state_names += ['running_sum_num']
    occupancy = running_sum_num

  # print occupancy.get_shape().as_list()

  # Concat occupancy, how much occupied and goal.

  network_manager = Network(task_params.dropout_vec, tf.shape(occupancy))

  """conv1"""  # kernel sz, stride, num feature maps
  xc = network_manager.conv_block(occupancy, 5, 2, 32, padding_in='VALID')
  print(xc)
  xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')
  print(xc)

  """conv2"""
  xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
  print(xc)
  xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')
  print(xc)

  """conv3"""
  xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
  print(xc)
  xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
  print(xc)

  """conv4"""
  xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
  print(xc)
  xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
  print(xc)
  """mp3 (default values)"""

  """ reshape """
  x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
  print(x)

  """ fc1 """
  x = network_manager.fc_block(x, 512)
  print(x)
  """ fc2 """
  x = network_manager.fc_block(x, 512)

  """Process Control"""

  """ Speed (measurements)"""
  with tf.name_scope("Speed"):
      speed = tf.reshape(m.input_tensors['step']['speed'], [-1,1])#input_data[1]  # get the speed from input data
      speed = network_manager.fc_block(speed, 128)
      speed = network_manager.fc_block(speed, 128)

  """ Joint sensory """
  j = tf.concat([x, speed], 1)
  j = network_manager.fc_block(j, 512)

  """Start BRANCHING"""
  branch_config = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], 
                    ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]]#, ["Speed"]]
  
  branches = []
  for i in range(0, len(branch_config)):
      with tf.name_scope("Branch_" + str(i)):
          if branch_config[i][0] == "Speed":
              # we only use the image as input to speed prediction
              branch_output = network_manager.fc_block(x, 256)
              branch_output = network_manager.fc_block(branch_output, 256)
          else:
              branch_output = network_manager.fc_block(j, 256)
              branch_output = network_manager.fc_block(branch_output, 256)

          branches.append(network_manager.fc(branch_output, len(branch_config[i])))

      print(branch_output)

  m.action_logits_op_vec = branches

  init_state = tf.constant(0., dtype=tf.float32, shape=[
      task_params.batch_size, 1, map_crop_size, map_crop_size,
      task_params.map_channels])

  m.train_ops['state_names'] = state_names
  m.train_ops['updated_state'] = updated_state
  m.train_ops['init_state'] = [init_state for _ in updated_state]
  
  
  #m.train_ops['common'] = [m.input_tensors['common']['orig_maps'], m.input_tensors['common']['goal_loc']]
  m.train_ops['batch_norm_is_training_op'] = batch_norm_is_training_op
  m.loss_ops = []; m.loss_ops_names = []

  '''if args.arch.readout_maps:
    with tf.name_scope('readout_maps'):
      all_occupancys = tf.concat(m.occupancys + m.confs, 3)
      readout_maps, probs = readout_general(
          all_occupancys, num_neurons=args.arch.rom_arch.num_neurons,
          strides=args.arch.rom_arch.strides, 
          layers_per_block=args.arch.rom_arch.layers_per_block, 
          kernel_size=args.arch.rom_arch.kernel_size,
          batch_norm_is_training_op=batch_norm_is_training_op,
          wt_decay=args.solver.wt_decay)

      gt_ego_maps = [m.input_tensors['step']['readout_maps_{:d}'.format(i)]
                     for i in range(len(task_params.readout_maps_crop_sizes))]
      m.readout_maps_gt = tf.concat(gt_ego_maps, 4)
      gt_shape = tf.shape(m.readout_maps_gt)
      m.readout_maps_logits = tf.reshape(readout_maps, gt_shape)
      m.readout_maps_probs = tf.reshape(probs, gt_shape)

      print('----------readout--------------')
      print(m.readout_maps_logits.get_shape().as_list())
      # Add a loss op
      m.readout_maps_loss_op = tf.losses.sigmoid_cross_entropy(
          tf.reshape(m.readout_maps_gt, [-1, len(task_params.readout_maps_crop_sizes)]), 
          tf.reshape(readout_maps, [-1, len(task_params.readout_maps_crop_sizes)]),
          scope='loss')
      m.readout_maps_loss_op = 1000.*m.readout_maps_loss_op'''


  ewma_decay = 0.99 if is_training else 0.0


  weight_one = tf.ones_like(m.input_tensors['train']['action'], dtype=tf.float32, name='weight')
  ones_old, ones = tf.split(weight_one, [1,2], 1)
  
  tens = ones_old * 20.0
  weight = tf.concat([tens, ones], 1)
  sh_list = weight.get_shape().as_list()
  print(sh_list)
  m.reg_loss_op, m.data_loss_op, m.total_loss_op, m.acc_ops, m.action_logits_op = \
    compute_losses_multi_or(m.action_logits_op_vec,
                            m.input_tensors['train']['action'], 
                            m.input_tensors['train']['command'], 
                            weights=weight,
                            num_actions=task_params.num_actions,
                            data_loss_wt=100 ,#args.solver.data_loss_wt,
                            reg_loss_wt= 0.05,#args.solver.reg_loss_wt,
                            ewma_decay=ewma_decay)
  #print(z)
  m.train_ops['step'] = m.action_logits_op
  
  '''if args.arch.readout_maps:
    m.total_loss_op = m.total_loss_op + m.readout_maps_loss_op
    m.loss_ops += [m.readout_maps_loss_op]
    m.loss_ops_names += ['readout_maps_loss']'''

  m.loss_ops += [m.reg_loss_op, m.data_loss_op, m.total_loss_op]
  m.loss_ops_names += ['reg_loss', 'data_loss', 'total_loss']

  if args.solver.freeze_conv:
    vars_to_optimize = list(set(tf.trainable_variables()) -
                            set(m.vision_ops.vars_to_restore))
  else:
    vars_to_optimize = None

  m.lr_op, m.global_step_op, m.train_op, m.should_stop_op, m.optimizer, \
  m.sync_optimizer = tf_utils.setup_training(
      m.total_loss_op, 
      args.solver.initial_learning_rate, 
      args.solver.steps_per_decay,
      args.solver.learning_rate_decay, 
      args.solver.momentum,
      args.solver.max_steps, 
      args.solver.sync, 
      args.solver.adjust_lr_sync,
      args.solver.num_workers, 
      args.solver.task,
      vars_to_optimize=vars_to_optimize,
      clip_gradient_norm=args.solver.clip_gradient_norm,
      typ=args.solver.typ, momentum2=args.solver.momentum2,
      adam_eps=args.solver.adam_eps)

  if args.arch.sample_gt_prob_type == 'subsection':
    m.sample_gt_prob_op = tf_utils.step_subsection(args.arch.isd_k,
                                                         m.global_step_op)

  elif args.arch.sample_gt_prob_type == 'inverse_sigmoid_decay':
    m.sample_gt_prob_op = tf_utils.inverse_sigmoid_decay(args.arch.isd_k,
                                                         m.global_step_op)
  elif args.arch.sample_gt_prob_type == 'zero':
    m.sample_gt_prob_op = tf.constant(-1.0, dtype=tf.float32)

  elif args.arch.sample_gt_prob_type.split('_')[0] == 'step':
    step = int(args.arch.sample_gt_prob_type.split('_')[1])
    m.sample_gt_prob_op = tf_utils.step_gt_prob(
        step, m.input_tensors['step']['step_number'][0,0,0])

  m.sample_action_type = args.arch.action_sample_type
  m.sample_action_combine_type = args.arch.action_sample_combine_type

  m.summary_ops = {
      summary_mode: _add_summaries(m, args, summary_mode,
                                   args.summary.arop_full_summary_iters)}

  m.init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
  m.saver_op = tf.train.Saver(keep_checkpoint_every_n_hours=4,
                              write_version=tf.train.SaverDef.V2)
  return m
