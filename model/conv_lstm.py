# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:47:10 2018

@author: Administrator
"""
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
class BasicConvLSTMCell(object):
  """Basic Conv LSTM recurrent network cell.
  """

  def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
               state_is_tuple=False, activation=tf.nn.tanh,time_size=3):
    """Initialize the basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      num_features: int thats the depth of the cell 
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
      the `c_state` and `m_state`.  If False, they are concatenated
      along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    #if not state_is_tuple:
      #logging.warn("%s: Using a concatenated state is slower and will soon be "
      #             "deprecated.  Use state_is_tuple=True.", self)
    self.shape = shape 
    self.filter_size = filter_size
    self.num_features = num_features 
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self.time_size=time_size
    self.ALL_SIZE_ONE=256
    self.ALL_SIZE_TWO=1
#  @property
#  def state_size(self):
#    return (LSTMStateTuple(self._num_units, self._num_units)
#            if self._state_is_tuple else 2 * self._num_units)

#  @property
#  def output_size(self):
#    return self._num_units
#  def zero_state(self, batch_size, dtype):
  def zero_state(self, batch_size):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
      filled with zeros
    """
    
    shape = self.shape 
    num_features = self.num_features
    zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2]) 
    return zeros

  def C_LSTM_cell(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__,reuse=tf.AUTO_REUSE):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
      concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

      new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.nn.sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat(axis=3, values=[new_c, new_h])
      return new_h, new_state
  def State_Result(self, X, state):
      outputs = []
      self.state = state
#    LSTM层的运算过程
      with tf.variable_scope('CV_LSTM'):
          for timestep in range(self.time_size):
              if timestep > 0:
                  tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
              (cell_output, state) = self.C_LSTM_cell(X[:, timestep,: ,: , :],state)
              outputs.append(cell_output)
#            print(state)
#LSTM的最后输出结果
      h_state = outputs[-1]
      return h_state
  def Full_connect(self, X, state):
      state_result=self.State_Result(X, state)
      shape=state_result.get_shape().as_list()
      print(shape)
      nodes=shape[1]*shape[2]*shape[3]
      reshaped=tf.reshape(state_result,[shape[0],nodes])
      #第一个全连接层
      with tf.variable_scope('F_Layer_one',reuse=tf.AUTO_REUSE):
          weight_three=tf.get_variable("weight",
                                   [nodes,self.ALL_SIZE_ONE],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
          bias_three=tf.get_variable("bias",[self.ALL_SIZE_ONE],
                                   initializer=tf.constant_initializer(0.1))
          layer1=tf.nn.relu(tf.matmul(reshaped,weight_three)+bias_three)
        #        如果keep_out不等于None，则使用dropout函数，任何一个给定单元的留存率
#        if(KeepProb!=1):
#          layer1 = tf.nn.dropout(layer1, keep_prob=KeepProb)
#第二个全连接层
      with tf.variable_scope('F_Layer_two',reuse=tf.AUTO_REUSE):
          weight_four=tf.get_variable("weight",
                                   [self.ALL_SIZE_ONE,self.ALL_SIZE_TWO],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
          bias_four=tf.get_variable("bias",[self.ALL_SIZE_TWO],
                                   initializer=tf.constant_initializer(0.1))
          layer2=tf.nn.relu(tf.matmul(layer1,weight_four)+bias_four)
#        如果keep_out不等于None，则使用dropout函数，任何一个给定单元的留存率
#        if(KeepProb!=1):
#          layer2 = tf.nn.dropout(layer6, keep_prob=KeepProb)
      return layer2

def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
  """convolution:
  Args:
    args: a 4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 4D Tensor with shape [batch h w num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
#  结果是[[32, 14, 7, 3], [32, 14, 7, 3]]
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[3]
#      结果是6
  dtype = [a.dtype for a in args][0]
#  获取元素的类型
  # Now the computation.
  with tf.variable_scope(scope or "Conv"):
    matrix = tf.get_variable(
        "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
#    bias_one=tf.get_variable("bias",[num_features],
#                                  initializer=tf.constant_initializer(0))
#    matrix1 = tf.get_variable(
#        "Matrix1", [filter_size[0]+1, filter_size[1]+1, num_features, num_features], dtype=dtype)
    if len(args) == 1:
      res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
    else:
      args=tf.concat(axis=3, values=args)
#      结果是(32, 14, 7, 6)
      res = tf.nn.conv2d(args, matrix, strides=[1, 1, 1, 1], padding='SAME')
#      res=tf.nn.relu(tf.nn.bias_add(res, bias_one))
#      res = tf.nn.conv2d(res, matrix1, strides=[1, 1, 1, 1], padding='SAME')
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [num_features],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term