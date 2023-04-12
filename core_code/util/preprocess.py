import numpy as np

def preprocess_image(image, percentile_range = [2,98], normalization_range = [0,1]):
    """
    Normalize each channel of an image separately by cropping the pixel intensity range to a given percentile range
    and then scaling the values to a given normalization range.
    """        
    
    # Check if the input is a numpy ndarray
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
        
    normalized_channels = []        
    for channel_idx in range(image.shape[0]):
        channel = image[channel_idx]
        
        # Crop the pixel intensity range of the channel to the desired percentile range
        cropped_channel = crop_to_percentile_range(channel, percentile_range)
        
        # Scale the pixel intensity values of the channel to the desired normalization range
        normalized_channel = normalize_intensity_range(cropped_channel, normalization_range)
        
        normalized_channels.append(normalized_channel)
        
    return np.stack(normalized_channels)
        
        
def crop_to_percentile_range(img_channel, percentile_range):
    """
    Crop the pixel intensity range of an image channel to a given percentile range.
    """
    low_cutoff = np.percentile(img_channel, percentile_range[0])
    high_cutoff = np.percentile(img_channel, percentile_range[1])
    return np.clip(img_channel, low_cutoff, high_cutoff)


def normalize_intensity_range(img_channel, normalization_range=[0, 1]):
    """
    Normalize the pixel intensity values of an image channel to a given range.

    """
    min_val, max_val = normalization_range
        
    return (img_channel - min_val) / (max_val - min_val)
    
    