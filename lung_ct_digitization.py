import os
import argparse
import numpy as np
import pydicom
import matplotlib.pyplot as plt


def load_dicom(file_path):
    """Read a DICOM medical image."""
    dicom_data = pydicom.dcmread(file_path)
    pixel_array = dicom_data.pixel_array.astype(np.float32)

    return dicom_data, pixel_array


def convert_to_hu(dicom_data, pixel_array):
    """Convert pixel values into Hounsfield Units."""
    slope = float(getattr(dicom_data, "RescaleSlope", 1))
    intercept = float(getattr(dicom_data, "RescaleIntercept", 0))

    hu_image = pixel_array * slope + intercept

    return hu_image


def apply_lung_window(hu_image):
    """Apply lung window to the CT image."""
    window_min = -1000
    window_max = 400

    lung_image = np.clip(hu_image, window_min, window_max)

    return lung_image


def normalize_image(image):
    """Normalize image values between 0 and 1."""
    min_value = np.min(image)
    max_value = np.max(image)

    if max_value == min_value:
        return np.zeros_like(image)

    normalized = (image - min_value) / (max_value - min_value)

    return normalized


def convert_to_8bit(image):
    """Convert image into 8-bit format (0-255)."""
    image_8bit = (image * 255).astype(np.uint8)

    return image_8bit


def save_results(hu_image, lung_image, normalized, output_dir):
    """Save processed images and numerical data."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "hu_image.npy"), hu_image)
    np.save(os.path.join(output_dir, "lung_window.npy"), lung_image)
    np.save(os.path.join(output_dir, "normalized_image.npy"), normalized)

    image_8bit = convert_to_8bit(normalized)

    plt.imsave(
        os.path.join(output_dir, "digitized_ct.png"),
        image_8bit,
        cmap="gray"
    )

    print("Results saved successfully!")


def show_results(original, hu_image, lung_image, normalized):
    """Display and save the image processing stages."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original DICOM Image")

    axes[1].imshow(hu_image, cmap="gray")
    axes[1].set_title("Hounsfield Units")

    axes[2].imshow(lung_image, cmap="gray")
    axes[2].set_title("Lung Window")

    axes[3].imshow(normalized, cmap="gray")
    axes[3].set_title("Normalized Image")
    
    image_8bit = convert_to_8bit(normalized)
    axes[4].imshow(image_8bit, cmap="gray")
    axes[4].set_title("8-bit Final Output")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    
    # Աυստեղ ավտոմատ պահպանում ենք նկարը results պանակում
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/all_stages_comparison.png", dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Lung CT Image Digitization"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input DICOM file"
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Output directory"
    )

    args = parser.parse_args()

    print("Loading DICOM image...")

    dicom_data, pixel_array = load_dicom(args.input)

    print("Converting image into Hounsfield Units...")

    hu_image = convert_to_hu(dicom_data, pixel_array)

    print("Applying lung window...")

    lung_image = apply_lung_window(hu_image)

    print("Normalizing image...")

    normalized = normalize_image(lung_image)

    print("Saving results...")

    save_results(
        hu_image,
        lung_image,
        normalized,
        args.output
    )

    print("Image dimensions:", pixel_array.shape)
    print("Minimum HU:", np.min(hu_image))
    print("Maximum HU:", np.max(hu_image))

    show_results(
        pixel_array,
        hu_image,
        lung_image,
        normalized
    )

    print("Digitization completed!")


if __name__ == "__main__":
    main()