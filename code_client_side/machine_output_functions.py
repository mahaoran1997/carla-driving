import numpy as np
import math

from codification import *

from PIL import Image



# This is used to mount the vector just for the early case
def mount_vec_image(img,  sequence_resample, network_input_size):
    clip_input = np.empty((sequence_resample, network_input_size[0] , network_input_size[1] , network_input_size[2]))



    for i in range(0, sequence_resample):
        clip_input[i] = img

    return clip_input  # clip_input.reshape((sequence_resample,network_input_size[0]*network_input_size[1]*network_input_size[2]))


def fuse_frames(image_clip, fused_size, network_input_size):

    fused_clip = np.empty((1, network_input_size[0] ,
                           network_input_size[1] , network_input_size[2]*fused_size ))



    for i in range(image_clip.shape[0]):


        fused_clip[0,:,:,(i*3):(i+1)*3] = image_clip[i,:,:,:]


    return fused_clip


def mount_vec(img, speed, sequence_resample, network_input_size):
    clip_input = np.empty((sequence_resample, network_input_size[0] , network_input_size[1] , network_input_size[2]))
    speed_vector = np.empty((sequence_resample, 1))


    for i in range(0, sequence_resample):
        clip_input[i] = img
        speed_vector[i] = speed

    return clip_input,speed_vector  # clip_input.reshape((sequence_resample,network_input_size[0]*network_input_size[1]*network_input_size[2]))



def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(clip_input=[],speed_vector=[],first_time=True)

def branched_dynamic(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]


    if control_input == 2 or control_input == 0.0:
        all_net = branches[0]
    elif control_input == 3:
        all_net = branches[2]
    elif control_input == 4:
        all_net = branches[3]
    elif control_input == 5:
        all_net = branches[1]
    else:
        all_net = branches[0]


    # LETS START FIRST JUST TRYING JUST ONE BRANCH EVERY TIME

    if branched_dynamic.first_time:
        branched_dynamic.clip_input, branched_dynamic.speed_vector = mount_vec(image_input, speed / config.speed_factor,
                               config.number_frames_sequenced,config.image_size)
        branched_dynamic.first_time = False
    else:


        # print input_vec.shape
        # print image_input.shape

        """ The zero position receives 1"""
        # for  i in range(0,config.sequence_resample-1):


        branched_dynamic.clip_input = np.delete(branched_dynamic.clip_input, 0, axis=0)

        branched_dynamic.clip_input = np.append(branched_dynamic.clip_input, image_input[np.newaxis,:,:,:], axis=0)


        branched_dynamic.speed_vector = np.delete(branched_dynamic.speed_vector, 0, axis=0)

        branched_dynamic.speed_vector = np.append(branched_dynamic.speed_vector,
                                                  np.array(speed)[np.newaxis][:, np.newaxis], axis=0)

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    #image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    #image_result.save(str('saida_res.jpg'))


    feedDict = {x: branched_dynamic.clip_input, input_speed: branched_dynamic.speed_vector,
                dout: [1] *  len(config.dropout)}

    #start_time = time.time()
    output = sess.run(all_net, feed_dict=feedDict)
    #duration = time.time() - start_time

    # print "DURATION"
    # print duration
    # print output
    # print output[0][4]

    predicted_steers = 0
    predicted_gas = 0
    predicted_brake = 0
    for j in range(0, config.number_frames_sequenced):
        predicted_steers += (output[j][0]) / config.number_frames_sequenced
        predicted_gas += (output[j][1]) / config.number_frames_sequenced
        predicted_brake += (output[j][2]) / config.number_frames_sequenced

        #print '     ',output[j][0]



    return predicted_steers,predicted_gas,predicted_brake


@static_vars(clip_input=[],first_time=True)
def branched_early(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

    speed = np.array(speed / config.speed_factor)


    speed = speed.reshape((1, 1))


    if control_input == 2 or control_input == 0.0:
        all_net = branches[0]
    elif control_input == 3:
        all_net = branches[2]
    elif control_input == 4:
        all_net = branches[3]
    elif control_input == 5:
        all_net = branches[1]
    else:
        all_net = branches[0]


    # LETS START FIRST JUST TRYING JUST ONE BRANCH EVERY TIME

    if branched_dynamic.first_time:
        branched_dynamic.clip_input = mount_vec_image(image_input,
                               config.number_frames_fused, config.image_size)
        branched_dynamic.first_time = False
    else:


        # print input_vec.shape
        # print image_input.shape

        """ The zero position receives 1"""
        # for  i in range(0,config.sequence_resample-1):


        branched_dynamic.clip_input = np.delete(branched_dynamic.clip_input, 0, axis=0)

        branched_dynamic.clip_input = np.append(branched_dynamic.clip_input, image_input[np.newaxis,:,:,:], axis=0)




    fused_input = fuse_frames(branched_dynamic.clip_input, config.number_frames_fused, config.image_size)

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    # image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    # image_result.save(str('saida_res.jpg'))






    feedDict = {x: fused_input, input_speed:speed, dout: [1] *  len(config.dropout)}

    #start_time = time.time()
    output = sess.run(all_net, feed_dict=feedDict)
    #duration = time.time() - start_time

    # print "DURATION"
    # print duration
    # print output
    # print output[0][4]

    predicted_steers = output[0][0]
    predicted_gas = output[0][1]
    predicted_brake = output[0][2]

    return predicted_steers,predicted_gas,predicted_brake





def input(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    control = np.array(encode(control_input))
    control = control.reshape((1, 4))
    speed = speed.reshape((1, 1))

    net = branches[0]

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    # image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    # image_result.save(str('saida_res.jpg'))

    feedDict = {x: image_input, input_speed: speed, input_control: control, dout: [1] * len(config.dropout)}

    output_net = sess.run(net, feed_dict=feedDict)

    predicted_steers = (output_net[0][0])

    predicted_acc = (output_net[0][1])

    predicted_brake = (output_net[0][2])

    return predicted_steers, predicted_acc, predicted_brake


def input_drc(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    control = np.array(encode(control_input))
    control = control.reshape((1, 4))
    speed = speed.reshape((1, 1))

    net = branches[0]

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    # image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    # image_result.save(str('saida_res.jpg'))

    feedDict = {x: image_input, input_speed: speed, input_control: control, dout: [1] * len(config.dropout)}

    output_net = sess.run(net, feed_dict=feedDict)

    predicted_steers = (output_net[0][0])

    predicted_acc = (output_net[0][1])

    return predicted_steers, predicted_acc, 0


def goal(image_input, speed, goal_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_goal = train_manager._input_data[config.inputs_names.index("Goal")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)
    # aux =goal_input[0]
    # goal_input[0]  = goal_input[1]
    # goal_input[1] =aux
    module = math.sqrt(goal_input[0] * goal_input[0] + goal_input[1] * goal_input[1])
    goal_input = np.array(goal_input)
    goal_input = goal_input.reshape((1, 2)) / module
    speed = speed.reshape((1, 1))

    net = branches[0]
    print " Inputing ", goal_input

    feedDict = {x: image_input, input_speed: speed, input_goal: goal_input, dout: [1] * len(config.dropout)}

    output_net = sess.run(net, feed_dict=feedDict)

    predicted_steers = (output_net[0][0])

    predicted_acc = (output_net[0][1])

    predicted_brake = (output_net[0][2])

    return predicted_steers, predicted_acc, predicted_brake


def base_no_speed(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    net = branches[0]

    feedDict = {x: image_input, dout: [1] * len(config.dropout)}

    output_net = sess.run(net, feed_dict=feedDict)

    predicted_steers = (output_net[0][0])

    return predicted_steers, None, None


def base(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    net = branches[0]

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_net = sess.run(net, feed_dict=feedDict)

    predicted_steers = (output_net[0][0])

    predicted_acc = (output_net[0][1])

    predicted_brake = (output_net[0][2])

    return predicted_steers, predicted_acc, predicted_brake


def controller( wp_angle):


    steer = 0.8 * wp_angle


    if steer > 0:
        steer = min(steer, 1)
    else:
        steer = max(steer, -1)


    return steer






def single_branch_waypoints(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]
    # image_result = Image.fromarray((image_input*255).astype(np.uint8))
    # image_result.save('image.png')

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    if control_input == 2 or control_input == 0.0:
        all_net = branches[0]
    elif control_input == 3:
        all_net = branches[2]
    elif control_input == 4:
        all_net = branches[3]
    elif control_input == 5:
        all_net = branches[1]

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    # image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    # image_result.save(str('saida_res.jpg'))

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_all = sess.run(all_net, feed_dict=feedDict)


    predicted_acc = (output_all[0][1])

    predicted_brake = (output_all[0][2])

    predicted_waypoint1 = (output_all[0][3])

    predicted_waypoint2 = (output_all[0][4])

    predicted_steers = controller((predicted_waypoint1+predicted_waypoint2)/2.0)
    print 'Steers'
    print (predicted_waypoint1+predicted_waypoint2)/2.0
    print predicted_steers

    print ' Ghost steer ', output_all[0][1]



    predicted_speed = sess.run(branches[4], feed_dict=feedDict)
    predicted_speed = predicted_speed[0][0]
    real_speed = speed * config.speed_factor
    print ' REAL PREDICTED ',predicted_speed*config.speed_factor

    print ' REAL SPEED ',real_speed
    real_predicted = predicted_speed * config.speed_factor
    if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
        # print 'BOOSTING'
        predicted_acc = 1 * (20.0 / config.speed_factor - speed) + predicted_acc  # print "DURATION"

        predicted_brake = 0.0

        predicted_acc = predicted_acc[0][0]

    return predicted_steers, predicted_acc, predicted_brake




def base_drc(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    net = branches[0]

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_net = sess.run(net, feed_dict=feedDict)

    predicted_steers = (output_net[0][0])

    predicted_acc = (output_net[0][1])

    return predicted_steers, predicted_acc, 0


def branched_speed(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    if control_input == 2 or control_input == 0.0:
        steer_net = branches[0]
    elif control_input == 3:
        steer_net = branches[2]
    elif control_input == 4:
        steer_net = branches[3]
    elif control_input == 5:
        steer_net = branches[1]

    acc_net = branches[4]
    brake_net = branches[5]
    speed_net = branches[6]  # This is hardcoded !!!!!!

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    # image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    # image_result.save(str('saida_res.jpg'))

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_steer = sess.run(steer_net, feed_dict=feedDict)
    output_acc = sess.run(acc_net, feed_dict=feedDict)
    output_speed = sess.run(speed_net, feed_dict=feedDict)
    output_brake = sess.run(brake_net, feed_dict=feedDict)

    if config.use_speed_trick:
        if speed < (4.0 / config.speed_factor) and output_speed[0][0] > (
                4.0 / config.speed_factor):  # If (Car Stooped) and ( It should not have stoped)
            output_acc[0][0] = 0.3 * (4.0 / config.speed_factor - speed) + output_acc[0][0]  # print "DURATION"

    predicted_steers = (output_steer[0][0])

    predicted_acc = (output_acc[0][0])

    predicted_brake = (output_brake[0][0])

    return predicted_steers, predicted_acc, predicted_brake


def single_branch_seg(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]
    # image_result = Image.fromarray((image_input*255).astype(np.uint8))
    # image_result.save('image.png')

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    if control_input == 2 or control_input == 0.0:
        all_net = branches[0]
    elif control_input == 3:
        all_net = branches[2]
    elif control_input == 4:
        all_net = branches[3]
    elif control_input == 5:
        all_net = branches[1]

    # image_result.save(str('saida_res.jpg'))

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_all = sess.run(all_net, feed_dict=feedDict)

    predicted_steers = (output_all[0][0])

    predicted_acc = (output_all[0][1])

    predicted_brake = (output_all[0][2])

    predicted_speed = sess.run(branches[4], feed_dict=feedDict)
    predicted_speed = predicted_speed[0][0]
    real_speed = speed * config.speed_factor

    real_predicted = predicted_speed * config.speed_factor
    if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)

        predicted_acc = 1 * (20.0 / config.speed_factor - speed) + predicted_acc  # print "DURATION"

        predicted_brake = 0.0
        predicted_acc = predicted_acc[0][0]

    print predicted_steers, predicted_acc, predicted_brake

    return predicted_steers, predicted_acc, predicted_brake


def single_branch(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]
    # image_result = Image.fromarray((image_input*255).astype(np.uint8))
    # image_result.save('image.png')

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    if control_input == 2 or control_input == 0.0:
        all_net = branches[0]
    elif control_input == 3:
        all_net = branches[2]
    elif control_input == 4:
        all_net = branches[3]
    elif control_input == 5:
        all_net = branches[1]

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    # image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    # image_result.save(str('saida_res.jpg'))

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_all = sess.run(all_net, feed_dict=feedDict)

    predicted_steers = (output_all[0][0])

    predicted_acc = (output_all[0][1])

    predicted_brake = (output_all[0][2])

    predicted_speed = sess.run(branches[4], feed_dict=feedDict)
    predicted_speed = predicted_speed[0][0]
    real_speed = speed * config.speed_factor

    real_predicted = predicted_speed * config.speed_factor
    if real_speed < 5.0 and real_predicted > 6.0:  # If (Car Stooped) and ( It should not have stoped)
        # print 'BOOSTING'
        predicted_acc = 1 * (20.0 / config.speed_factor - speed) + predicted_acc  # print "DURATION"

        predicted_brake = 0.0

        predicted_acc = predicted_acc[0][0]

    return predicted_steers, predicted_acc, predicted_brake


def single_branch_drc(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    if control_input == 2 or control_input == 0.0:
        all_net = branches[0]
    elif control_input == 3:
        all_net = branches[2]
    elif control_input == 4:
        all_net = branches[3]
    elif control_input == 5:
        all_net = branches[1]

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    # image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    # image_result.save(str('saida_res.jpg'))

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_all = sess.run(all_net, feed_dict=feedDict)

    predicted_steers = (output_all[0][0])

    predicted_acc = (output_all[0][1])

    return predicted_steers, predicted_acc, 0


def branched_speed_4cmd(image_input, speed, control_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    input_control = train_manager._input_data[config.inputs_names.index("Control")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    if control_input == 5:
        steer_net = branches[1]
    elif control_input == 3 or control_input == 6:
        steer_net = branches[2]
    elif control_input == 4 or control_input == 7 or control_input == 8:
        steer_net = branches[3]
    else:
        steer_net = branches[0]

    acc_net = branches[4]
    brake_net = branches[5]
    speed_net = branches[6]  # This is hardcoded !!!!!!

    # print clip_input.shape

    # input_vec = input_vec.reshape((1,config.input_size[0]*config.input_size[1]*config.input_size[2]))

    # image_result = Image.fromarray((scipy.misc.imresize(image_input[0],(210,280,3))).astype(np.uint8))

    # image_result.save(str('saida_res.jpg'))

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_steer = sess.run(steer_net, feed_dict=feedDict)
    output_acc = sess.run(acc_net, feed_dict=feedDict)
    output_speed = sess.run(speed_net, feed_dict=feedDict)
    output_brake = sess.run(brake_net, feed_dict=feedDict)

    if config.use_speed_trick:
        if speed < (4.0 / config.speed_factor) and output_speed[0][0] > (
                4.0 / config.speed_factor):  # If (Car Stooped) and ( It should not have stoped)
            output_acc[0][0] = 0.3 * (4.0 / config.speed_factor - speed) + output_acc[0][0]  # print "DURATION"

    predicted_steers = (output_steer[0][0])

    predicted_acc = (output_acc[0][0])

    predicted_brake = (output_brake[0][0])

    return predicted_steers, predicted_acc, predicted_brake


def get_intermediate_rep(image_input, speed, config, sess, train_manager):
    seg_network = train_manager._gray
    x = train_manager._input_images
    dout = train_manager._dout
    input_control = train_manager._input_data[config.inputs_names.index("Control")]

    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]
    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))
    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    output_image = sess.run(seg_network, feed_dict=feedDict)

    return output_image[0]


def vbp(image_input, speed, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout
    input_speed = train_manager._input_data[config.inputs_names.index("Speed")]

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    speed = np.array(speed / config.speed_factor)

    speed = speed.reshape((1, 1))

    vbp_images_tensor = train_manager._vis_images

    feedDict = {x: image_input, input_speed: speed, dout: [1] * len(config.dropout)}

    vbp_images = sess.run(vbp_images_tensor, feed_dict=feedDict)

    return vbp_images[0]


def vbp_nospeed(image_input, config, sess, train_manager):
    branches = train_manager._output_network
    x = train_manager._input_images
    dout = train_manager._dout

    image_input = image_input.reshape((1, config.image_size[0], config.image_size[1], config.image_size[2]))

    vbp_images_tensor = train_manager._vis_images

    feedDict = {x: image_input, dout: [1] * len(config.dropout)}

    vbp_images = sess.run(vbp_images_tensor, feed_dict=feedDict)

    return vbp_images[0]
