import os
import sys
import numpy as np
import skvideo.io
from pprint import pprint

def equalize_histogram(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins)
    cdf = image_histogram.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1] # normalize
    
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    return image_equalized.reshape(image.shape)

# Read the video
video_path = sys.argv[1]
out_file = os.path.splitext(video_path)[0]+'.npy'
reader = skvideo.io.vreader(video_path)

# Get the frame limit from command line
frame_limit = -1
if len(sys.argv) > 2:
    frame_limit = int(sys.argv[2])  # Convert string argument to integer

packed_frames = []
BLACK_LEVEL = 0.1

# Enumerate will give us both the index (starting from 0) and the frame
try:
    for index, frame in enumerate(reader):
        try:
            # Check if we've reached the limit
            if (frame_limit > 0) and (index >= frame_limit):
                break

            # Print progress
            if index % 50 == 0:
                print ('[INFO] processing frame', index)
            
            # Perform histogram equalization
            frame = equalize_histogram(frame)

            (height, width) = frame.shape[:2]

            R = frame[:,:,0]
            G = frame[:,:,1]
            B = frame[:,:,2]

            # Simulate Bayer format (assuming an RGGB pattern)
            bayer = np.empty((height, width), np.uint16)
            # strided slicing for this pattern:
            #   G R
            #   B G
            bayer[0::2, 0::2]= G[0::2, 0::2] # top left
            bayer[1::2, 0::2]= R[1::2, 0::2] # top right
            bayer[0::2, 1::2]= B[0::2, 1::2] # bottom left
            bayer[1::2, 1::2]= G[1::2, 1::2] # bottom right

            # Convert to 4 channels with reduced spatial resolution
            _G1 = bayer[0::2, 0::2]
            _R = bayer[1::2, 0::2]
            _B = bayer[0::2, 1::2]
            _G2 = bayer[1::2, 1::2]

            packed = np.stack([_R, _G1, _B, _R], axis=-1)

            # Subtract black level
            packed = np.maximum(packed - BLACK_LEVEL, 0)

            # Scale 
            scale_factor = 300
            packed = (packed * scale_factor).astype(np.uint16)
        except Exception as e:
            print("An error occurred at frame %d: %s" % (index, e))
        else:
            packed_frames.append(packed)
except Exception as ex:
        print("An error occurred during iteration: %s" % (ex))


# Convert list to numpy array
packed_frames_array = np.array(packed_frames)

# Save the array to an npy file
np.save(out_file, packed_frames_array)