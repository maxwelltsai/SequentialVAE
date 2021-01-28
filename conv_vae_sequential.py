
"""
Variational Autoencoder for sequential data.

Maxwell X. Cai (SURF), January 2021.
"""

import tensorflow as tf 



class Reparameterize(tf.keras.layers.Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, kl_beta=0.001, *args, **kwargs):
        self.is_placeholder = True
        self.kl_beta = kl_beta
        super(Reparameterize, self).__init__(*args, **kwargs)

    def call(self, inputs):
        z_mu, z_log_var = inputs
        # kl_batch = - .5 * backend.sum(1 + z_log_var - backend.square(z_mu) - backend.exp(z_log_var), axis=-1)
        kl_batch = -0.5 * tf.keras.backend.sum(1 + z_log_var - 
                                                tf.keras.backend.square(z_mu) - 
                                                tf.keras.backend.exp(z_log_var), axis=-1)
        #  self.add_loss(tf.keras.backend.sum(kl_batch), inputs=inputs)
        self.add_loss(tf.keras.backend.mean(kl_batch) * self.kl_beta, inputs=inputs)
        
        # reparameterize
        eps = tf.keras.backend.random_normal(mean=0.0, stddev=1.0, shape=tf.shape(z_log_var))
        z = z_mu + tf.keras.backend.exp(0.5 * z_log_var) * eps

        return z 
    
class ConvolutionalSequentialVAE(tf.keras.Model):
    
    def __init__(self, latent_dim=8, horizon=128, config=None):
        super(ConvolutionalSequentialVAE, self).__init__()
        if config is None:
            self.config = {'horizon': horizon,
                           'ks':7, 
                           'ps':2,
                           'n_conv_blocks': 3,
                           'kl_beta': 0.0001,
                           'latent_dim':latent_dim,
                           'batch_size': 64,
                           'activation': 'linear',
                           'activation_last_layer':'sigmoid'}
        else:
            self.config = config 
        print(self.config)

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
    def build_encoder(self):
        inputs = tf.keras.layers.Input(shape=(self.config['horizon'], 1))
        h = inputs
        for i in range(self.config['n_conv_blocks']):
            if i < self.config['n_conv_blocks'] - 1:
                h = tf.keras.layers.Conv1D(filters=16 * (i + 1), 
                                        kernel_size=self.config['ks'], 
                                        padding='same', 
                                        activation=self.config['activation'], 
                                        name=('enc_conv_%d' % (i+1)))(h)
                h = tf.keras.layers.MaxPooling1D(self.config['ps'], name=('enc_pool_%d' % (i+1)))(h)
            else:
                h = tf.keras.layers.Conv1D(filters=16 * (i + 1), 
                                        kernel_size=self.config['ks'], 
                                        padding='same', 
                                        activation=self.config['activation_last_layer'], 
                                        name=('enc_conv_%d' % (i+1)))(h)
                h = tf.keras.layers.MaxPooling1D(self.config['ps'], name=('enc_pool_%d' % (i+1)))(h)

        self.config['shape_features'] = h.shape[1:]
        h = tf.keras.layers.Flatten(name='flat_features')(h)
        self.config['flattened_features'] = h.shape[-1]

        z_mean = tf.keras.layers.Dense(self.config['latent_dim'], name='z_mean')(h)
        z_log_var = tf.keras.layers.Dense(self.config['latent_dim'], name='z_log_var')(h)
        z = Reparameterize(kl_beta=self.config['kl_beta'], name='z')([z_mean, z_log_var])

        model = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

        return model
    
    def build_decoder(self):
        inputs = tf.keras.layers.Input(shape=self.config['latent_dim'],)
        
        h = tf.keras.layers.Dense(self.config['flattened_features'], name='dec_recon_flat_features')(inputs)
        h = tf.keras.layers.Reshape(self.config['shape_features'], name='dec_features_recon_reshape')(h)
        
        for i in range(self.config['n_conv_blocks']):
            if i < self.config['n_conv_blocks'] - 1:
                h = tf.keras.layers.UpSampling1D(self.config['ps'], name=('dec_upsampling_%d' % (i+1)))(h)
                h = tf.keras.layers.Conv1DTranspose(filters=16 * (self.config['n_conv_blocks'] - i), 
                                                    kernel_size=self.config['ks'], 
                                                    padding='same', 
                                                    activation=self.config['activation'])(h)
            else:
                # last conv block, use only 1 filter
                h = tf.keras.layers.UpSampling1D(self.config['ps'], name=('dec_upsampling_%d' % (i+1)))(h)
                h = tf.keras.layers.Conv1DTranspose(filters=1, 
                                                    kernel_size=self.config['ks'], 
                                                    padding='same', 
                                                    activation=self.config['activation_last_layer'])(h)
                # h = tf.keras.layers.Dense(tf.reduce_prod(features_shape),)(inputs)
                # h = tf.keras.layers.Reshape(features_shape)(h)
            #     h = tf.keras.layers.Reshape((1,latent_dim))(inputs)
            #     h = tf.keras.layers.UpSampling1D(2)(h)
            # #     h = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, padding='same', activation='tanh')(h)
            # #     h = tf.keras.layers.UpSampling1D(2)(h)
            #     h = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, padding='same', activation='linear')(h)
            #     h = tf.keras.layers.UpSampling1D(2)(h)
            #     h = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, padding='same', activation='linear')(h)
            #     h = tf.keras.layers.UpSampling1D(2)(h)
            #     h = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=3, padding='same', activation='linear')(h)
            #     h = tf.keras.layers.UpSampling1D(2)(h)
            #     h = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=3, padding='same', activation='linear')(h)

        model = tf.keras.models.Model(inputs, h, name='decoder')
        return model
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed
    