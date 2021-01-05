import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras import regularizers
from keras import models
from keras.layers import Layer
from keras.layers.core import Lambda

# Model input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
# 
#     V: num_of_vertices
#     T: num_of_timesteps
#     F: num_of_features
#
# Model output: (*, 5)


class TemporalAttention(Layer):
    '''
    compute temporal attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.U_1 = self.add_weight(name='U_1',
                                      shape=(num_of_vertices, 1),
                                      initializer='uniform',
                                      trainable=True)
        self.U_2 = self.add_weight(name='U_2',
                                      shape=(num_of_features, num_of_vertices),
                                      initializer='uniform',
                                      trainable=True)
        self.U_3 = self.add_weight(name='U_3',
                                      shape=(num_of_features, ),
                                      initializer='uniform',
                                      trainable=True)
        self.b_e = self.add_weight(name='b_e',
                                      shape=(1, num_of_timesteps, num_of_timesteps),
                                      initializer='uniform',
                                      trainable=True)
        self.V_e = self.add_weight(name='V_e',
                                      shape=(num_of_timesteps, num_of_timesteps),
                                      initializer='uniform',
                                      trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        
        # shape of lhs is (batch_size, T, V)
        lhs=K.dot(tf.transpose(x,perm=[0,1,3,2]), self.U_1)
        lhs=tf.reshape(lhs,[tf.shape(x)[0],num_of_timesteps,num_of_features])
        lhs = K.dot(lhs, self.U_2)
        
        # shape of rhs is (batch_size, V, T)
        rhs = K.dot(self.U_3, tf.transpose(x,perm=[2,0,3,1])) # K.dot((F),(V,batch_size,F,T))=(V,batch_size,T)
        rhs=tf.transpose(rhs,perm=[1,0,2])
        
        # shape of product is (batch_size, T, T)
        product = K.batch_dot(lhs, rhs)
        
        S = tf.transpose(K.dot(self.V_e, tf.transpose(K.sigmoid(product + self.b_e),perm=[1, 2, 0])),perm=[2, 0, 1])
        
        # normalization
        S = S - K.max(S, axis = 1, keepdims = True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis = 1, keepdims = True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[1])


class SpatialAttention(Layer):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.W_1 = self.add_weight(name='W_1',
                                      shape=(num_of_timesteps, 1),
                                      initializer='uniform',
                                      trainable=True)
        self.W_2 = self.add_weight(name='W_2',
                                      shape=(num_of_features, num_of_timesteps),
                                      initializer='uniform',
                                      trainable=True)
        self.W_3 = self.add_weight(name='W_3',
                                      shape=(num_of_features, ),
                                      initializer='uniform',
                                      trainable=True)
        self.b_s = self.add_weight(name='b_s',
                                      shape=(1, num_of_vertices, num_of_vertices),
                                      initializer='uniform',
                                      trainable=True)
        self.V_s = self.add_weight(name='V_s',
                                      shape=(num_of_vertices, num_of_vertices),
                                      initializer='uniform',
                                      trainable=True)
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        
        # shape of lhs is (batch_size, V, T)
        lhs=K.dot(tf.transpose(x,perm=[0,2,3,1]), self.W_1)
        lhs=tf.reshape(lhs,[tf.shape(x)[0],num_of_vertices,num_of_features])
        lhs = K.dot(lhs, self.W_2)
        
        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.W_3, tf.transpose(x,perm=[1,0,3,2])) # K.dot((F),(V,batch_size,F,T))=(V,batch_size,T)
        rhs=tf.transpose(rhs,perm=[1,0,2])
        
        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)
        
        S = tf.transpose(K.dot(self.V_s, tf.transpose(K.sigmoid(product + self.b_s),perm=[1, 2, 0])),perm=[2, 0, 1])
        
        # normalization
        S = S - K.max(S, axis = 1, keepdims = True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis = 1, keepdims = True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2],input_shape[2])


def F_norm(weight_matrix, Falpha):
    '''
    compute F Norm
    '''
    return Falpha * K.sum(weight_matrix ** 2)


class Graph_Learn(Layer):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S = tf.convert_to_tensor(0.0)
        super(Graph_Learn, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a = self.add_weight(name='a',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        self.add_loss(F_norm(self.S,self.alpha))
        super(Graph_Learn, self).build(input_shape)

    def call(self, x):
        #Input:  [N, timesteps, vertices, features]
        _, T, V, F = x.shape
        N = tf.shape(x)[0]
        
        # shape: (N,V,F) use the current slice
        x = x[:,int(int(x.shape[1])/2),:,:]
        # shape: (N,V,V)
        diff = tf.transpose(tf.broadcast_to(x,[V,N,V,F]), perm=[2,1,0,3]) - x
        # shape: (N,V,V)
        tmpS = K.exp(K.reshape(K.dot(tf.transpose(K.abs(diff), perm=[1,0,2,3]), self.a), [N,V,V]))
        # normalization
        S = tmpS / tf.transpose(tf.broadcast_to(K.sum(tmpS,axis=1),[V,N,V]), perm=[1,2,0])
        
        self.S = K.mean(S,axis=0)
        return S

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices, num_of_vertices)
        return (input_shape[0],input_shape[2],input_shape[2])


class cheb_conv_with_SAt_GL(Layer):
    '''
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             SAtt(batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(cheb_conv_with_SAt_GL, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape,SAtt_shape,S_shape=input_shape
        _, num_of_timesteps, num_of_vertices, num_of_features = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, num_of_features, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_SAt_GL, self).build(input_shape)

    def call(self, x):
        #Input:  [x,SAtt,S]
        assert isinstance(x, list)
        assert len(x)==3,'cheb_gcn error'
        x,spatial_attention,W=x
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        #Calculating Chebyshev polynomials
        D = tf.matrix_diag(K.sum(W,axis=1))
        L = D - W
        '''
        Here may report an error which may due to TensorFlow's version, consider to change the size of batch_size or directly edit the following line to "lambda_max = 2".
        '''
        lambda_max = K.max(tf.self_adjoint_eigvals(L),axis=1)
        L_t = (2 * L) / tf.reshape(lambda_max,[-1,1,1]) - [tf.eye(int(num_of_vertices))]
        cheb_polynomials = [tf.eye(int(num_of_vertices)), L_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        #GCN
        outputs=[]
        for time_step in range(num_of_timesteps):
            # shape of x is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            output = K.zeros(shape = (tf.shape(x)[0], num_of_vertices, self.num_of_filters))
            
            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[kk]
                    
                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at,perm=[0, 2, 1]), graph_signal)

                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output,1))
            
        return K.relu(K.concatenate(outputs, axis = 1))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2],self.num_of_filters)

    
class cheb_conv_with_SAt_static(Layer):
    '''
    K-order chebyshev graph convolution with static graph structure
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             SAtt(batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, cheb_polynomials, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        super(cheb_conv_with_SAt_static, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape,SAtt_shape=input_shape
        _, num_of_timesteps, num_of_vertices, num_of_features = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, num_of_features, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_SAt_static, self).build(input_shape)

    def call(self, x):
        #Input:  [x,SAtt]
        assert isinstance(x, list)
        assert len(x)==2,'cheb_gcn error'
        x,spatial_attention = x
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        
        outputs=[]
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            output = K.zeros(shape = (tf.shape(x)[0], num_of_vertices, self.num_of_filters))
            
            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[kk]
                    
                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at,perm=[0, 2, 1]), graph_signal)

                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output,1))
            
        return K.relu(K.concatenate(outputs, axis = 1))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2],self.num_of_filters)


def reshape_dot(x):
    #Input:  [x,TAtt]
    x,temporal_At = x
    return tf.reshape(
        K.batch_dot(
            tf.reshape(tf.transpose(x,perm=[0,2,3,1]),(tf.shape(x)[0], -1, tf.shape(x)[1])), 
            temporal_At
        ),
        [-1, x.shape[1],x.shape[2],x.shape[3]]
    )


def LayerNorm(x):
    # do the layer normalization
    x_residual,time_conv_output=x
    relu_x=K.relu(x_residual+time_conv_output)
    ln=tf.contrib.layers.layer_norm(relu_x,begin_norm_axis=3)
    return ln


def Transpose(x, perm):
    # packaged Transpose
    return tf.transpose(x,perm = perm)


def GraphSleepBlock(x, k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel, useGL, GLalpha, i=0):
    '''
    packaged Spatial-temporal convolution Block
    -------
    x: input
    '''
        
    # shape is (batch_size, T, T)   
    temporal_At = TemporalAttention()(x)
    x_TAt = Lambda(reshape_dot,name='reshape_dot'+str(i))([x,temporal_At])

    # cheb gcn with spatial attention
    spatial_At = SpatialAttention()(x_TAt)
    
    if useGL:
        S = Graph_Learn(alpha=GLalpha)(x)
        spatial_gcn = cheb_conv_with_SAt_GL(num_of_filters=num_of_chev_filters, k=k)([x, spatial_At, S])
    else:
        spatial_gcn = cheb_conv_with_SAt_static(num_of_filters=num_of_chev_filters, k=k, cheb_polynomials=cheb_polynomials)([x, spatial_At])
    
    # convolution along time axis
    ''' Output: batch_size, num_of_timesteps, num_of_vertices, num_of_time_filters)'''
    time_conv_output = layers.Conv2D(
        filters = num_of_time_filters, 
        kernel_size = (time_conv_kernel, 1), 
        padding = 'same', 
        strides = (time_conv_strides, 1)
    )(spatial_gcn)

    # residual shortcut
    x_residual = layers.Conv2D(
        filters = num_of_time_filters, 
        kernel_size = (1, 1), 
        strides = (1, time_conv_strides)
    )(x)
    
    # LayerNorm
    end_output = Lambda(LayerNorm,
                        name='layer_norm'+str(i))([x_residual,time_conv_output])
    return end_output


def build_GraphSleepNet(k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel, 
                sample_shape, num_block, dense_size, opt, useGL, GLalpha, regularizer, dropout):
    
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = layers.Input(shape=sample_shape, name='Input-Data')
    
    # GraphSleepBlock
    block_out = GraphSleepBlock(data_layer,k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel,useGL,GLalpha)
    for i in range(1,num_block):
        block_out = GraphSleepBlock(block_out,k,num_of_chev_filters,num_of_time_filters,1,cheb_polynomials,time_conv_kernel,useGL,GLalpha,i)
        
    # Global dense layer
    block_out = layers.Flatten()(block_out)
    for size in dense_size:
        block_out=layers.Dense(size)(block_out)
    
    # dropout
    if dropout!=0:
        block_out = layers.Dropout(dropout)(block_out)
    
    # softmax classification
    softmax = layers.Dense(5,activation='softmax',kernel_regularizer=regularizer)(block_out)
    
    model = models.Model(inputs = data_layer, outputs = softmax)
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    return model


def build_GraphSleepNet_test():
    
    # an example to test
    cheb_k = 3
    num_of_chev_filters = 10
    num_of_time_filters = 10
    time_conv_strides = 1
    time_conv_kernel = 3
    dense_size=np.array([64,32])
    cheb_polynomials = [np.random.rand(26,26),np.random.rand(26,26),np.random.rand(26,26)]

    opt='adam'

    model=build_GraphSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters,time_conv_strides, cheb_polynomials, time_conv_kernel, 
                      sample_shape=(5,26,9),num_block=1, dense_size=dense_size, opt=opt, useGL=True, 
                      GLalpha=0.0001, regularizer=None, dropout = 0.0)
    model.summary()
    model.save('GraphSleepNet_build_test.h5')
    print("save ok")
    return model


# build_GraphSleepNet_test()

