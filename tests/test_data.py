from efficient_frontier import data


def test_get_rand_weights_shape():
    assert data.get_rand_weights(100, 3).shape == (100, 3)
