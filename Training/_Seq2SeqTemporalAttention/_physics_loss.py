import tensorflow as tf
from tensorflow.keras import backend as K

def piml_loss_factory(alpha=0.1, beta=0.1, v_max=1.0, a_max=0.5, dt=1.0):
    """
    Creates a custom physics-informed loss function that:
      - Minimizes MSE
      - Penalizes velocity > v_max
      - Penalizes acceleration > a_max
    The final loss = MSE + alpha * velocity_penalty + beta * acceleration_penalty.

    alpha, beta: relative weights for velocity and acceleration penalties
    v_max: maximum allowed velocity
    a_max: maximum allowed acceleration
    dt: time step between consecutive predictions
    """

    def piml_loss(y_true, y_pred):
        """
        y_true, y_pred shape: (batch_size, prediction_horizon, 1)
        We'll assume a single output (X or Y or Z) per model.
        """
        # 1) Standard MSE across all timesteps
        mse = K.mean(K.square(y_pred - y_true), axis=[1,2])  # shape => (batch_size,)

        # 2) Approximate velocities and accelerations
        #    We'll define velocity at t as (y_{t+1} - y_{t}) / dt
        #    and acceleration at t as (y_{t+1} - 2y_{t} + y_{t-1}) / dt^2.
        # y_pred => (batch_size, horizon, 1)
        # We'll slice along the time dimension.

        # We'll handle velocity for t in [0 .. horizon-2]
        # So velocity_pred shape => (batch_size, horizon-1)
        y_pred_squeezed = tf.squeeze(y_pred, axis=-1)  # => (batch_size, horizon)

        # velocity t => y_pred[:, t+1] - y_pred[:, t]
        vel_pred = (y_pred_squeezed[:, 1:] - y_pred_squeezed[:, :-1]) / dt
        # => shape (batch_size, horizon-1)

        # acceleration t => y_pred[:, t+1] - 2 y_pred[:, t] + y_pred[:, t-1]
        # We'll compute this for t in [1 .. horizon-2], i.e. skip the edges
        # => shape (batch_size, horizon-2)
        accel_pred = (y_pred_squeezed[:, 2:] - 2.0*y_pred_squeezed[:, 1:-1] + y_pred_squeezed[:, :-2]) / (dt*dt)

        # 3) Velocity penalty
        # We penalize if |vel| > v_max
        # A simple penalty = mean(ReLU(|vel| - v_max)).
        # Alternatively, square it for a smoother gradient, etc.
        vel_excess = tf.nn.relu(tf.abs(vel_pred) - v_max)  # shape (batch_size, horizon-1)
        vel_penalty = K.mean(vel_excess, axis=1)           # shape (batch_size,)

        # 4) Acceleration penalty
        # We penalize if |accel| > a_max
        accel_excess = tf.nn.relu(tf.abs(accel_pred) - a_max)  # shape (batch_size, horizon-2)
        accel_penalty = K.mean(accel_excess, axis=1)           # shape (batch_size,)

        # 5) Combine all
        # total_loss per sample => MSE + alpha * vel_penalty + beta * accel_penalty
        total_loss = mse + alpha*vel_penalty + beta*accel_penalty

        # Return average across batch
        return K.mean(total_loss, axis=0)

    return piml_loss
