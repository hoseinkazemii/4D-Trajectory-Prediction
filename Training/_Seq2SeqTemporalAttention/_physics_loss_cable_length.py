import tensorflow as tf
from tensorflow.keras import backend as K

def piml_loss_factory(alpha=0.1, max_cable_length=10.0):
    """
    Returns a custom 'physics-informed' loss function that penalizes predictions violating:
      - For Y model: Y < 0 (assuming Y must be above ground).
      - For XZ model: sqrt(X^2 + Z^2) > max_cable_length.
    alpha: weight for the physics penalty
    max_cable_length: your crane cable length or radial constraint
    """
    def piml_loss(y_true, y_pred):
        """
        y_true, y_pred have shape (batch_size, prediction_horizon, D)
          Where D = 1 for Y model, D = 2 for XZ model
        We'll compute MSE + alpha * physics_penalty
        """
        # 1) Standard MSE
        mse = K.mean(K.square(y_pred - y_true), axis=[1,2])  # average over time & dimension
        # shape => (batch_size,)

        # 2) Physics penalty
        # We'll parse dimension (D) from y_pred’s last axis
        output_dim = tf.shape(y_pred)[-1]

        physics_penalty = 0.0

        # We'll do a *per-sample, per-timestep* penalty, then average.
        # physics_penalty should be shape (batch_size,).

        # (A) If D == 1 => Y model
        #     Y >= 0 => penalty for negative Y
        #     penalty = sum(ReLU(-Y))^2 or something simpler like sum(ReLU(-Y))
        def penalty_y(y):
            # We want y >= 0, so negative values are penalized
            # Let's do a simple L1 penalty:
            return K.relu(-y)  # shape => same as y

        # (B) If D == 2 => XZ model
        #     sqrt(X^2 + Z^2) <= max_cable_length
        #     penalty if r = sqrt(X^2 + Z^2) > L, i.e. ReLU(r - L)
        def penalty_xz(xz):
            x = xz[..., 0]
            z = xz[..., 1]
            r = K.sqrt(K.square(x) + K.square(z))
            return K.relu(r - max_cable_length)  # shape => (batch_size, horizon)

        # We can handle both cases in one pass:
        # Gather all timesteps => shape (batch_size, horizon, D)
        # Then compute per-timestep penalty and sum/mean
        # We'll do a dynamic check:
        physics_penalty_per_sample = tf.cond(
            tf.equal(output_dim, 1),  # if shape[-1] == 1 => Y
            lambda: K.mean(penalty_y(y_pred), axis=[1,2]), 
            # if shape[-1] == 2 => XZ
            lambda: K.mean(penalty_xz(y_pred), axis=1) 
            # axis=1 => average over horizon dimension => shape (batch_size,)
        )

        physics_penalty = physics_penalty_per_sample  # shape => (batch_size,)

        # Combine them => final loss per sample
        total_loss_per_sample = mse + alpha * physics_penalty
        # Return average across the batch
        return K.mean(total_loss_per_sample, axis=0)

    return piml_loss
