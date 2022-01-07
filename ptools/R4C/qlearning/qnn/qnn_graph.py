"""

 2020 (c) piteren

 baseline qnn graph (TF)

"""
from ptools.neuralmess.base_elements import tf
from ptools.neuralmess.layers import lay_dense


def qnn_graph(
        name=               'qnn',
        num_actions: int=   4,
        num_states: int=    16,
        state_emb_width=    4,
        hidden_layers_size= (12,),
        gamma=              0.9,
        seed=               121,
        **kwargs):

    with tf.variable_scope(name):

        qv_target_PH = tf.placeholder(      # qv next state placeholder
            shape=  [None,num_actions],
            dtype=  tf.float32)
        reward_PH = tf.placeholder(         # reward
            shape=  [None],
            dtype=  tf.float32)
        state_PH = tf.placeholder(          # state
            shape=  [None],
            dtype=  tf.int32)
        enum_actions_PH = tf.placeholder(   # enumerated action indexes (0,1),(1,3),(2,0),..
            shape=  [None,2],
            dtype=  tf.int32)

        state_emb = tf.get_variable(
            name=   'state_emb',
            shape=  [num_states,state_emb_width],
            dtype=  tf.float32)

        input = tf.nn.embedding_lookup(state_emb, state_PH)
        print('input:', input)

        for l in hidden_layers_size:
            input = lay_dense(
                input=      input,
                units=      l,
                activation= tf.nn.relu,
                seed=       seed)
        output = lay_dense( # QV for all actions (for given input(state))
            input=      input,
            units=      num_actions,
            activation= None,
            seed=       seed)

        pred_qv = tf.gather_nd(output, indices=enum_actions_PH)
        gold_qv = reward_PH + gamma * tf.reduce_max(qv_target_PH, axis=-1) # gold is predicted by same network

        loss = tf.losses.mean_squared_error(labels=gold_qv, predictions=pred_qv) # loss on predicted vs next, we want predicted to match next
        loss = tf.reduce_mean(loss)

    return {
        'qv_target_PH':     qv_target_PH,
        'reward_PH':        reward_PH,
        'state_PH':         state_PH,
        'enum_actions_PH':  enum_actions_PH,
        'output':           output,
        'loss':             loss}