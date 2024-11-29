import jax
import equinox
import einops

class Layer(equinox.Module):
    """
    All possible of transformer layers
    """
    layer_norm_1: equinox.nn.LayerNorm = equinox.field(static=False)
    attn: equinox.nn.MultiheadAttention = equinox.field(static=False)
    layer_norm_2: equinox.nn.LayerNorm = equinox.field(static=False)
    mlp: equinox.nn.MLP = equinox.field(static=False)    
    def __init__(self, dimension: int, rng: jax.Array):
        self.layer_norm_1 = equinox.nn.LayerNorm(dimension, use_bias=False, use_weight=False)
        rng, key = jax.random.split(rng)
        self.attn = equinox.nn.MultiheadAttention(num_heads=8, query_size=dimension, key=key)
        self.layer_norm_2 = equinox.nn.LayerNorm(dimension, use_bias=False, use_weight=False)
        rng, key = jax.random.split(rng)
        self.mlp = equinox.nn.MLP(dimension, dimension, dimension*10, depth=1, key=key)
    def __call__(self, x: jax.Array, mask: jax.Array):
        """
        x: variable length list of tokens
        mask: jax array with shape (len(x), len(x))
        """
        y = equinox.filter_vmap(self.layer_norm_1)(x)
        x += self.attn(y, y, y, mask)
        y = equinox.filter_vmap(self.layer_norm_2)(x)
        x += equinox.filter_vmap(self.mlp)(y)
        return x
    
def generate_io_mask(context_timesteps: jax.Array, latent_timesteps: jax.Array) -> jax.Array:
    """
    Generates a casual mask using timesteps for each token. This allows you to have out of order tokens. We can either be encoding things into the latent or decoding things into the context
    TODO: this current doesn't take into account padding. This should not be impossible. It'll manifest as the time_in having padded tokens be uint max.
    token_timesteps: array of unsigned integers with shape (T1,)
    latent_timesteps: array of unsigned integers with shape (T2,)
    output: array of bools equal with shape (T1 + T2, T1 + T2)
    """
    mask_dimension = context_timesteps.shape[0] + latent_timesteps.shape[0]
    time_in = einops.pack([context_timesteps, latent_timesteps], "*")[0] # we look at both latents and tokens        
    time_out = einops.pack([jax.numpy.full_like(context_timesteps, -1), latent_timesteps], "*")[0] # we don't generate context, we only generate latents
    return einops.repeat(time_out, "t1 -> t1 t2", t2=mask_dimension) >= einops.repeat(time_in, "t1 -> t1 t2", t2=mask_dimension).T        

def generate_backbone_mask(timesteps: jax.Array) -> jax.Array:
    return einops.repeat(timesteps, "t1 -> t1 t2", t2=timesteps.shape[0]) >= einops.repeat(timesteps, "t1 -> t1 t2", t2=timesteps.shape[0]).T

import unittest

class TestTransformerLayer(unittest.TestCase):
    def test_generate_io_mask(self):
        context_timesteps = jax.numpy.array([0, 2, 1])
        latent_timesteps = jax.numpy.array([0, 0, 2, 2, 1, 1])
        expected_mask = jax.numpy.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],            
            [1, 0, 1, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 0, 0, 1, 1],    
        ], dtype=bool)
        mask = generate_io_mask(context_timesteps, latent_timesteps)
        self.assertTrue(jax.numpy.array_equal(mask, expected_mask), f"Expected {expected_mask}, but got {mask}")

    def test_generate_backbone_mask(self):
        timesteps = jax.numpy.array([0, 2, 1])
        expected_mask = jax.numpy.array([
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 1],
        ], dtype=bool)
        mask = generate_backbone_mask(timesteps)
        self.assertTrue(jax.numpy.array_equal(mask, expected_mask), f"Expected {expected_mask}, but got {mask}")

    def test_transformer_layer_shape(self):
        # Define some dummy input data
        embedding_size = 128
        feature_size = 64
        rng = jax.random.PRNGKey(0)
        
        # Create a transformer Layer instance
        layer = Layer(embedding_size + feature_size, rng=rng)
        
        # Create dummy input data
        input_data = jax.random.normal(rng, (10, embedding_size + feature_size))
        
        # Create a dummy mask
        mask = jax.numpy.ones((10, 10), dtype=bool)
        
        # Call the transformer layer
        output = layer(input_data, mask)
        
        # Check the shape of the output
        expected_shape = input_data.shape
        self.assertEqual(output.shape, expected_shape, f"Expected shape {expected_shape}, but got {output.shape}")

    def test_transformer_layer_overfit(self):
        import optax
        embedding_size = 128
        feature_size = 64
        rng = jax.random.PRNGKey(0)
        layer = Layer(embedding_size + feature_size, rng=rng)
        num_batches = 2
        num_timesteps = 2

        rng, key = jax.random.split(rng)
        input_data = jax.random.normal(key, (num_timesteps, embedding_size + feature_size))
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, num_batches)
        input_data_batches = [input_data + (jax.random.normal(key, input_data.shape) * 1e-2) for key in keys]

        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, num_batches)
        output_data_batches = [jax.random.normal(key, (num_timesteps, embedding_size + feature_size)) for key in keys]
        masks = [jax.numpy.tril(jax.numpy.ones((num_timesteps, num_timesteps), dtype=bool)) for _ in range(num_batches)]
        
        def loss_fn(layer, input_data, mask, target):
            output = layer(input_data, mask)
            return jax.numpy.mean((output - target) ** 2)
        
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(equinox.filter(layer, equinox.is_array_like))
        
        num_epochs = 1000
        for _ in range(num_epochs):
            for input_data, output_data, mask in zip(input_data_batches, output_data_batches, masks):
                params = layer
                grads = equinox.filter_grad(loss_fn)(params, input_data, mask, output_data)
                updates, opt_state = optimizer.update(grads, opt_state, equinox.filter(params, equinox.is_array_like))
                layer = equinox.apply_updates(params, updates)
        
        for input_data, output_data, mask in zip(input_data_batches, output_data_batches, masks):
            output = layer(input_data, mask)
            loss = jax.numpy.mean((output - output_data) ** 2)
            self.assertTrue(loss < 1e-3, f"Model did not overfit as expected. Loss: {loss}")
        
if __name__ == '__main__':
    unittest.main()
