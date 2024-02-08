from tf.keras.layers inport Layer,LSTM,LayerNormalization,Add


class LSTMTransformerLayer(Layer):
  def __init__(self, height, dropout=0):
    super().__init__()
    self.LSTMb=tf.keras.layers.LSTM(units=height, return_sequences=True, 
                                    go_backwards=True,dropout=dropout)
    self.LSTMf=tf.keras.layers.LSTM(units=height, return_sequences=True,
                                   dropout=dropout)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
  def call(self, x):
    lstmb=self.LSTMb(x)
    lstmf=self.LSTMf(x)
    x = self.add([x, lstmb, lstmf])
    x = self.layernorm(x)
    return x
