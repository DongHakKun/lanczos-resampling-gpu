import cupy as np
from PIL import Image
from joblib import Parallel, delayed
import os


def lanczos_resampling(original_image, OH, OW, a=3):
    def L(vec):  # Lanczos kernel
        vec = np.sinc(vec) * np.sinc(vec/a)
        return vec
    

    # Assuming original_image is in H by W by C or H by W format, where C is the channel count
    H, W, *C = original_image.shape
    if C == []:
        original_image = original_image[..., np.newaxis]
    original_image = original_image.astype(np.float64) # H by W by C

    #
    new_row_indices = np.clip((np.arange(OW) + 0.5) * (W/OW) - 0.5, 0, W-1)
    row_idx_for_sum = np.floor(new_row_indices).astype(int) - a + 1
    row_idx_for_sum = np.clip(row_idx_for_sum[:, np.newaxis] + np.arange(0, 2*a), 0, W-1).T
    diff_indices = new_row_indices - row_idx_for_sum

    # Apply kernel and normalize for each channel
    kernel_weights = L(diff_indices)
    kernel_weights /= kernel_weights.sum(axis=0, keepdims=True)

    original_image = original_image.transpose(2, 0, 1) # Transpose channel data to the front for broadcasting
    row_sampled_image = original_image[:, :, row_idx_for_sum] # C by H by 2a by OW
    del original_image # for memory save

    row_sampled_image *= kernel_weights[np.newaxis, np.newaxis,: , :]
    row_sampled_image = np.sum(row_sampled_image, axis=2)  # C by H by OW
    np.get_default_memory_pool().free_all_blocks()
    np.get_default_pinned_memory_pool().free_all_blocks()

    #
    new_col_indices = np.clip((np.arange(OH) + 0.5) * (H/OH) - 0.5, 0, H-1)
    col_idx_for_sum = np.floor(new_col_indices).astype(int) - a + 1
    col_idx_for_sum = np.clip(col_idx_for_sum[:, np.newaxis] + np.arange(0, 2*a), 0, H-1).T
    diff_indices = new_col_indices - col_idx_for_sum

    # Apply kernel and normalize for each channel
    kernel_weights = L(diff_indices)
    kernel_weights /= kernel_weights.sum(axis=0, keepdims=True)

    # Resample each channel along the column
    row_sampled_image = row_sampled_image.transpose(0, 2, 1) # Transpose channel data to the front for broadcasting
    col_sampled_image = row_sampled_image[:, :, col_idx_for_sum]  # C by OW by 2a by OH
    del row_sampled_image # For memory save

    col_sampled_image *= kernel_weights[np.newaxis, np.newaxis, : , :]
    final_image = np.sum(col_sampled_image, axis=2).transpose(2, 1, 0) # H by W by C
    np.get_default_memory_pool().free_all_blocks()
    np.get_default_pinned_memory_pool().free_all_blocks()

    return final_image.clip(0, 255).astype(np.uint8).squeeze()


def resample_image(file_path, source_dir, target_dir, short_side_size, quality=91):
    def calculate_new_dimensions(img):
        width, height = img.size
        # Determine the short side and maintain aspect ratio
        if height > width:
            scale = short_side_size / width
            new_height = int(height * scale)
            new_width = short_side_size
        else:
            scale = short_side_size / height
            new_width = int(width * scale)
            new_height = short_side_size

        return new_height, new_width

    # Generate the new file path
    rel_paths = os.path.relpath(file_path, source_dir)
    new_file_path = os.path.join(target_dir, rel_paths)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True) # Create the target directory if it doesn't exist
    if os.path.exists(new_file_path) == True: return # Override

    #
    img = Image.open(file_path) # Open the image
    metadata = img.getexif() # Preserve original metadata

    new_height, new_width = calculate_new_dimensions(img)
    img = np.array(img)
    img = lanczos_resampling(img, new_height, new_width)
    img = Image.fromarray(np.asnumpy(img))
    img.save(new_file_path, exif=metadata, quality=quality, optimize=True)


def process_directory(source_dir, target_dir, short_side_size, max_jobs, quality=91):
    jpg_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source_dir) for f in filenames if f.lower().endswith('.jpg')]

    Parallel(n_jobs=max_jobs)(delayed(resample_image)(file_path, source_dir, target_dir, short_side_size, quality) for file_path in jpg_files)


if __name__ == '__main__':
    source_dir = r''
    target_dir = r''
    short_side_size = 1440
    quality = 91
    max_jobs = 4

    process_directory(source_dir, target_dir, short_side_size, max_jobs, quality)
