import os
from PIL import Image, ImageOps

# Define the target size for resizing (e.g., 224x224 for most models)
target_size = (224, 224)

# Directory where the images are stored
source_directory = "train"  # Change to your source directory
destination_directory = "train_ready"  # Change to your destination directory

# Function to resize images while preserving aspect ratio and adding padding
def resize_and_pad_image(image_path, output_path, target_size):
    # Open image
    image = Image.open(image_path)
    
    # Pad the image to maintain aspect ratio
    padded_image = ImageOps.pad(image, target_size, color=(255, 255, 255))  # Adding white padding
    
    # Save the image to the destination path
    padded_image.save(output_path)

# Function to traverse folders and resize images
def process_images(source_dir, dest_dir, target_size):
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding destination folder
        relative_path = os.path.relpath(root, source_dir)
        output_folder = os.path.join(dest_dir, relative_path)
        os.makedirs(output_folder, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                # Full file path for source and destination
                source_file = os.path.join(root, file)
                destination_file = os.path.join(output_folder, file)
                
                # Resize and pad the image
                resize_and_pad_image(source_file, destination_file, target_size)

# Start the bulk resizing process
source_directory = "train"  # Set this to your source directory
destination_directory = "train_ready"  # Set this to your destination directory

# Call the function to process the images
process_images(source_directory, destination_directory, target_size)
