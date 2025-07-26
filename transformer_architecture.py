import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer,Embedding,LayerNormalization,Dense,Input
from tensorflow.keras import Model

def softmax(arr):
    exp_arr=tf.math.exp(arr-tf.reduce_max(arr))
    return exp_arr/tf.reduce_sum(exp_arr)

class Positional_Encoding(Layer):
    def __init__(self,dim,seq_len):
        super().__init__()
        pos=np.arange(seq_len)[:,np.newaxis] #seqlen,1
        i=np.arange(dim)[np.newaxis,:] #1,dim
        den=1/np.power(10000,2*i/np.float32(dim))
        self.postion_array=pos*den 
        self.postion_array[:,0::2]=np.sin(self.postion_array[:,0::2])
        self.postion_array[:,1::2]=np.sin(self.postion_array[:,1::2])
    def call(self,inputs):
        return inputs+tf.convert_to_tensor(self.postion_array[np.newaxis,:,:],dtype=tf.float32)
    
class MultiHeadAttention(Layer):
    def __init__(self,n_dim,no_of_heads,mask=False):
        super().__init__()
        self.WQ=self.add_weight(shape=(n_dim,n_dim),initializer="random_normal",trainable=True)
        self.WK=self.add_weight(shape=(n_dim,n_dim),initializer="random_normal",trainable=True)
        self.WV=self.add_weight(shape=(n_dim,n_dim),initializer="random_normal",trainable=True)
        self.ndim=n_dim
        self.mask=mask
        self.no_of_heads=no_of_heads
    def call(self,q,k,v):
        #In this step we projected the embed of word to ndim equal to n_dim/no_of_heads the batches has dimention=batch_size,seqlen,no_of_heads,n_dim/no_of_heads as we have to divide between multiple heads
        batch_size=tf.shape(q)[0]
        seq_len=tf.shape(q)[1]
        Q=tf.reshape(tf.matmul(q,self.WQ),(batch_size,self.no_of_heads,seq_len,self.ndim//self.no_of_heads))
        K=tf.reshape(tf.matmul(k,self.WK),(batch_size,self.no_of_heads,seq_len,self.ndim//self.no_of_heads))
        V=tf.reshape(tf.matmul(v,self.WV),(batch_size,self.no_of_heads,seq_len,self.ndim//self.no_of_heads))
        S=softmax(tf.matmul(Q,tf.transpose(K,perm=[0,1,3,2]))/tf.sqrt(tf.constant(self.ndim//self.no_of_heads,dtype=tf.float32))) #batch_size,self.no_of_heads,seq_len,seq_len
        if self.mask:
            mask_matrix=tf.linalg.band_part(tf.ones(shape=(1,1,seq_len,seq_len)),-1,0)
            S=softmax(S-(1.0-mask_matrix)*-1e9)
        x=tf.transpose(tf.matmul(S,V),perm=[0,2,1,3]) #batch_size,self.no_of_heads,seq_len,self.ndim//self.no_of_heads (before transpose)
        return tf.reshape(x,(batch_size,seq_len,self.ndim))

class FeedForward(Layer):
    def __init__(self,n_dim,seq_len):
        super().__init__()
        self.dense1=Dense(2048,activation="relu")
        self.dense2=Dense(n_dim,activation="sigmoid")
    def call(self,inputs):
        x=self.dense1(inputs)
        x=self.dense2(x)
        return x

class Encoder_Layer(Layer):
    def __init__(self,n_dim,seq_len,no_of_heads,vocab_size):
        super().__init__()
        self.embeding=Embedding(input_dim=vocab_size,output_dim=n_dim)
        self.position=Positional_Encoding(n_dim,seq_len)
        self.attention=MultiHeadAttention(n_dim,no_of_heads)
        self.norm1=LayerNormalization()
        self.feed=FeedForward(n_dim,seq_len)
        self.norm2=LayerNormalization()
    def call(self,inputs):
        embeded_vector=self.embeding(inputs)
        print(tf.shape(embeded_vector))
        input_of_multihead=self.position(embeded_vector)
        output=self.attention(input_of_multihead,input_of_multihead,input_of_multihead)
        added_attention=tf.add(input_of_multihead,output)
        input_to_feedforward=self.norm1(added_attention)
        ouput_to_feed_forward=self.feed(input_to_feedforward)
        x=self.norm2(tf.add(ouput_to_feed_forward,input_to_feedforward))
        return x
    
class Decoder_Layer(Layer):
    def __init__(self,n_dim,seq_len,no_of_heads,vocab_size):
        super().__init__()
        self.embeding=Embedding(input_dim=vocab_size,output_dim=n_dim)
        self.position=Positional_Encoding(n_dim,seq_len)
        self.attention=MultiHeadAttention(n_dim,no_of_heads,mask=True)
        self.norm1=LayerNormalization()
        self.crossattention=MultiHeadAttention(n_dim,no_of_heads)
        self.norm2=LayerNormalization()
        self.feed=FeedForward(n_dim,seq_len)
        self.norm3=LayerNormalization()
    def call(self,inputs,encoder_outputs):
        embeded_vector=self.embeding(inputs)
        input_of_multihead=self.position(embeded_vector)
        output_of_masked_att=self.attention(input_of_multihead,input_of_multihead,input_of_multihead)
        input_to_cross=self.norm1(tf.add(input_of_multihead,output_of_masked_att))
        output_of_cross=self.crossattention(encoder_outputs,encoder_outputs,input_to_cross)
        input_to_ff=self.norm2(tf.add(input_to_cross,output_of_cross))
        ouput_to_feed_forward=self.feed(input_to_ff)
        x=self.norm3(tf.add(ouput_to_feed_forward,input_to_ff))
        return softmax(x)
