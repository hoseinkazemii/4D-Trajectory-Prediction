# Step 1: Convert Time-Series Data to Images
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, UpSampling1D, LeakyReLU, BatchNormalization, Dropout, Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential



# Load dataset
data = pd.read_csv('sample_dataset.csv')  # Replace with your dataset path
time_series = data[['Time', 'X', 'Y', 'Z']].values




# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(time_series[:, 1:])

# Define the sequence length
sequence_length = 5  # Adjust this value as needed
print(len(normalized_data))
raise ValueError
# Reshape to image format
def create_image(data, img_width=5):
    seq_len, num_features = data.shape
    img_height = seq_len // img_width
    data = data[:img_height * img_width]
    img = data.reshape((img_height, img_width, num_features))
    return img

images = [create_image(normalized_data[i:i + sequence_length]) for i in range(0, len(normalized_data) - sequence_length, sequence_length)]
images = np.array(images)

# # Check if images are created correctly
# if len(images) > 0:
#     print(f"Number of images created: {len(images)}")
#     print(f"Shape of first image: {images[0].shape}")
# else:
#     print("No images created.")

# # Visualize the first image and save as PNG
# if len(images) > 0:
#     for i in range(3):
#         plt.figure(figsize=(10, 5))
#         plt.imshow(images[0][:,:,i], cmap='viridis')
#         plt.title(f"Channel {i+1} of the first image")
#         plt.colorbar()
#         # plt.savefig(f"channel_{i+1}_of_first_image.png")
#         plt.show()





###############################################################################################
# Step 2: Implement Conv1D-GAN
num_features = 3  # For (x, y, z)
latent_dim = sequence_length  # Make sure latent_dim matches sequence_length

def build_generator():
    model = Sequential()
    model.add(Dense(sequence_length * num_features, input_dim=sequence_length))
    model.add(Reshape((sequence_length, num_features)))
    model.add(Conv1D(50, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(50, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv1D(num_features, kernel_size=3, padding='same'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv1D(50, kernel_size=3, input_shape=(sequence_length, num_features)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(50, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    
    gan_input = tf.keras.Input(shape=(sequence_length,))
    generated_img = generator(gan_input)
    gan_output = discriminator(generated_img)
    
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    
    gan_input = tf.keras.Input(shape=(sequence_length,))
    generated_img = generator(gan_input)
    gan_output = discriminator(generated_img)
    
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)




###############################################################################################
# Step 3: Train the GAN
def train_gan(gan, generator, discriminator, data, epochs=3, batch_size=32):
    half_batch = batch_size // 2
    
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_imgs = data[idx]
        
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        generated_imgs = generator.predict(noise)


        real_imgs = np.squeeze(real_imgs, axis=1)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))  # Use latent_dim instead of 200
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

train_gan(gan, generator, discriminator, images)



#######################################################################################################
# Step 4: Convert Generated Images Back to Time-Series Data
def generate_time_series(generator, noise_dim=latent_dim, num_samples=1):
    noise = np.random.normal(0, 1, (num_samples, noise_dim))
    generated_imgs = generator.predict(noise)
    
    generated_data = []
    for img in generated_imgs:
        seq_len, num_features = img.shape
        img = img.reshape((seq_len * num_features,))
        img = img.reshape((seq_len, num_features))
        generated_data.append(img)
        
    return np.array(generated_data)


def plot_real_vs_generated(real_data, generated_data, num_samples=3):
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, num_samples * 3))
    if num_samples == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        for j in range(num_features):
            ax.plot(real_data[i, :, j], label=f'Real Feature {j+1}', linestyle='--')
            ax.plot(generated_data[i, :, j], label=f'Generated Feature {j+1}')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.show()



generated_time_series = generate_time_series(generator, num_samples=10)


# Plot generated time series data
plot_real_vs_generated(images[:3], generated_time_series[:3], num_samples=3)