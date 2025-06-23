import os
import glob
import numpy as np
from pca_subtraction import run_pca_subtraction

def main():
    # User: Change this pattern to match your data files
    directory_pattern = "/path/to/your/files/*.fits"

    # Output directory for results
    output_dir = "pca_results"
    os.makedirs(output_dir, exist_ok=True)

    # Find all files matching the pattern
    files = sorted(glob.glob(directory_pattern))
    if not files:
        print(f"No files found for pattern: {directory_pattern}")
        return

    print(f"Found {len(files)} files. Running PCA subtraction...")

    for file_path in files:
        try:
            # Example: load your data here (customize as needed)
            # Replace with your actual data loading logic
            fits.getdata(file_path)

            # Run PCA subtraction (customize arguments as needed)
            # Example: run_pca_subtraction(data, wave, start_wav, end_wav, component_count)
            # Here we use dummy arguments for illustration
            wave = np.arange(data.shape[-1])
            start_wav = wave[0]
            end_wav = wave[-1]
            component_count = 2

            tdm, wdm = run_pca_subtraction(data, wave, start_wav, end_wav, component_count)

            # Save output as .npy files
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            np.save(os.path.join(output_dir, f"{base_name}_tdm.npy"), tdm)
            np.save(os.path.join(output_dir, f"{base_name}_wdm.npy"), wdm)

            print(f"Processed and saved: {base_name}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()