import tensorflow as tf

def mse_branched(network_outputs, ground_truths, control_input, config):
    # typical input: network_outputs: output from network,
    #                ground_truths: _targets_data
    #                control_input: _input_data[self._config.inputs_names.index("Control")]
    '''
    branch_config = [["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"],
                     ["Steer", "Gas", "Brake"], ["Speed"]]
    '''
    branches_configuration = config.branch_config  # Get the branched configuration

    error_vec = []
    energy_vec = []
    loss_function = 0.0
    for ibranch in range(len(branches_configuration)):
        energy_branch = []
        error_branch = []

        print("network output ", ibranch, network_outputs[ibranch])
        network_outputs_split = tf.split(network_outputs[ibranch], network_outputs[ibranch].get_shape()[1], 1)

        print("branch configuration ", ibranch, branches_configuration[ibranch])
        for i_within_branch in range(len(branches_configuration[ibranch])):
            # Get the name of the target data to be taken
            target_name = branches_configuration[ibranch][i_within_branch]  # name of the target
            target_gt = ground_truths[config.targets_names.index(target_name)]
            # Yang: control branch is in the front, and the latter targets are not branched
            if ibranch < config.inputs_sizes[config.inputs_names.index('Control')]:
                branch_selection = tf.reshape(control_input[:, ibranch], tf.shape(ground_truths[0]))
            else:
                branch_selection = tf.ones(tf.shape(target_gt))

            square_dist = tf.pow(tf.subtract(target_gt, network_outputs_split[i_within_branch]), 2) * branch_selection
            dist = tf.abs(tf.subtract(target_gt, network_outputs_split[i_within_branch])) * branch_selection
            error_branch.append(dist)

            energy_branch.append(square_dist)
            loss_function = loss_function + square_dist * \
                                            config.branch_loss_weight[ibranch] * \
                                            config.variable_weight[target_name]

        energy_vec.append(energy_branch)
        error_vec.append(error_branch)

    if hasattr(config, "weight_decay"):
        wd = config.weight_decay
        if wd > 1e-8:
            print("using weight decay ", wd)
            decay_set = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
            l2_loss = wd * tf.add_n(decay_set)
            loss_function = loss_function + l2_loss

    return loss_function, error_vec, energy_vec, None, branch_selection
