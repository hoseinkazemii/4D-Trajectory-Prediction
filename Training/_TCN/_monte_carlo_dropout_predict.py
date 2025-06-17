import tensorflow as tf

def _monte_carlo_dropout_predict(model, test_X, **params):
    mc_dropout_passes = params.get("mc_dropout_passes")

    y_preds_list = []
    for _ in range(mc_dropout_passes):
        y_preds_list.append(model(test_X, training=True))

    return tf.stack(y_preds_list, axis=0)
