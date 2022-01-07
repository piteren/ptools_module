"""

 2020 (c) piteren

"""
from ptools.neuralmess.base_elements import tf
from ptools.neuralmess.layers import lay_dense


def pgnn_graph(
        name=           'pgnn',
        state_size=     4,
        num_actions=    2,
        hidden_layers=  (20,),
        seed=           121,
        **kwargs):

    with tf.variable_scope(name):

        states_PH = tf.placeholder( # environment state representation (prepared by PolicyGradientsEnvironment.encode_state())
            shape=  (None, state_size),
            dtype=  tf.float32,
            name=   'input_states')
        acc_rew_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.float32,
            name=   'accumulated_rewards')
        actions_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.int32,
            name=   'actions')

        layer = states_PH
        for i in range(len(hidden_layers)):
            layer = lay_dense(
                input=      layer,
                name=       f'hidden_layer_{i + 1}',
                units=      hidden_layers[i],
                activation= tf.nn.relu,
                seed=       seed)
        logits = lay_dense(
            input=      layer,
            name=       'logits',
            units=      num_actions,
            activation= None,
            seed=       seed)

        action_prob = tf.nn.softmax(logits)
        log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions_PH)
        loss = tf.reduce_mean(acc_rew_PH * log_policy)

    return {
        'states_PH':    states_PH,
        'acc_rew_PH':   acc_rew_PH,
        'actions_PH':   actions_PH,
        'action_prob':  action_prob,
        'loss':         loss}