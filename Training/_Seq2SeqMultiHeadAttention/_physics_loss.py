import tensorflow as tf

def constant_velocity_loss(y_pred, time_step=1.0):
    """
    Penalize changes in velocity (i.e., high acceleration) to encourage smooth motion.
    Assumes uniform time steps.
    """
    # y_pred shape: (batch, time_steps, dims)
    velocity = (y_pred[:, 1:, :] - y_pred[:, :-1, :]) / time_step
    acceleration = (velocity[:, 1:, :] - velocity[:, :-1, :]) / time_step
    accel_penalty = tf.reduce_mean(tf.square(acceleration))
    return accel_penalty

def kinematics_loss(y_pred, time_steps, initial_conditions=None):
    """
    Enforce a simple kinematic relationship on predicted positions.
    For demonstration, this function assumes zero acceleration (constant velocity).
    You can modify it to enforce different equations based on your knowledge.
    """
    # y_pred shape: (batch, time_steps, dims)
    batch_size = tf.shape(y_pred)[0]
    dims = tf.shape(y_pred)[-1]
    
    # For demonstration: assume constant velocity starting from origin.
    # Theoretical position x(t) = v0 * t with v0 assumed 0 for simplicity.
    # Modify initial_conditions and the equation as needed.
    # Here, we assume initial velocity v0=0 and constant acceleration a=0.
    # Therefore, expected position remains 0 (for simplicity).
    expected = tf.zeros_like(y_pred)  # shape matches y_pred
    
    # Compute deviation from the theoretical positions
    kin_penalty = tf.reduce_mean(tf.square(y_pred - expected))
    return kin_penalty

def combined_loss(y_true, y_pred):
    """
    Combined loss: standard MSE + physics-informed penalties.
    """
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Physics loss terms
    const_vel_penalty = constant_velocity_loss(y_pred, time_step=1.0)
    
    # Assume prediction horizon inferred from y_pred shape for time_steps
    pred_horizon = tf.shape(y_pred)[1]
    time_steps = tf.cast(tf.range(pred_horizon), tf.float32)
    kin_penalty = kinematics_loss(y_pred, time_steps)

    # Hyperparameters to balance the contributions of each term
    lambda_const = 0.0  # weight for constant velocity loss
    lambda_kin = 0.1    # weight for kinematics loss

    return mse_loss + lambda_const * const_vel_penalty + lambda_kin * kin_penalty
