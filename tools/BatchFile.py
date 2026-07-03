import os
import re
import pandas as pd
from datetime import datetime
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox


def extract_sample_id(file_path: str) -> str:
    """
    Robustly extract sample id from filenames such as:
    - 12RH26_41-Y_0000.tif.Label-Analysis(2).csv -> 12RH26_41-Y_0000
    - 14RH_7_57um.tif.Label-Analysis(2).csv      -> 14RH_7
    - BG02_4B_39um.Label-Analysis(2).csv         -> BG02_4B
    - CC10_18-74spinelrawvolume.Label-Analysis.csv -> CC10_18-74spinelrawvolume
    - 14RH12... -> 14RH12
    """
    name = os.path.basename(file_path)

    # Remove trailing known analysis suffixes
    name = re.sub(r"\.Label-Analysis.*$", "", name, flags=re.IGNORECASE)

    # If the filename contains an embedded .tif (common in ImageJ exports), drop everything after it
    if ".tif" in name.lower():
        parts = re.split(r"\.tif", name, flags=re.IGNORECASE)
        name = parts[0]

    # Drop extension if still present
    name = re.sub(r"\.(csv|xlsx)$", "", name, flags=re.IGNORECASE)

    # Drop trailing resolution suffix like _39um / _39u
    name = re.sub(r"_(\d+(?:\.\d+)?)(?:um|u)$", "", name, flags=re.IGNORECASE)

    # Trim stray separators
    name = name.strip("_- .")

    # Normalize to "base" id: take substring before the first underscore.
    # This keeps IDs like "BG02-4B" intact, while mapping "12RH26_41-Y_0000" -> "12RH26".
    if "_" in name:
        name = name.split("_", 1)[0].strip("_- .")

    return name or "NoSampleFound"


def extract_sample_id_from_processed_xlsx(file_path: str) -> str:
    """
    Extract sample id from our generated xlsx filenames:
    - totalBG02_4B.xlsx -> BG02_4B
    - Quantity_BG02_4B.xlsx -> BG02_4B
    - EigensBG02_4B.xlsx -> BG02_4B
    - VolumeEigenBG02_4B.xlsx -> BG02_4B
    - BG02_4B.xlsx -> BG02_4B
    """
    name = os.path.basename(file_path)
    name = re.sub(r"\.xlsx$", "", name, flags=re.IGNORECASE)

    for prefix in ("total", "Quantity_", "Eigens", "VolumeEigen"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Same normalization rule as raw filenames
    name = name.strip("_- .")
    if "_" in name:
        name = name.split("_", 1)[0].strip("_- .")

    return name or "NoSampleFound"


def find_first_sample_number(input_files):
    for filename in input_files:
        sample_id = extract_sample_id(filename)
        if sample_id and sample_id != "NoSampleFound":
            return sample_id
    return "NoSampleFound"


def generate_log_filename(output_directory, sample_number):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(output_directory, "WorkLog")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{sample_number}_{current_time}.txt")


def convert_and_clean_csv(input_files, output_directory, log_file):
    xlsx_files = []
    with open(log_file, 'a') as log:
        log.write(f"Conversion session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for file_path in input_files:
            filename = os.path.basename(file_path)
            if filename.lower().endswith('.csv'):
                sample_number = extract_sample_id(file_path)
                xlsx_path = os.path.join(output_directory, f"{sample_number}.xlsx")
                if not os.path.exists(xlsx_path):
                    df = pd.read_csv(file_path, skiprows=1)
                    df.to_excel(xlsx_path, index=False)
                    log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log.write(
                        f"{log_time}, converted {filename} at {file_path} to {sample_number}.xlsx at {xlsx_path}, rows processed: {len(df)}\n")
                    print(f"Converted and cleaned {filename} to {sample_number}.xlsx in {output_directory}")
                xlsx_files.append(xlsx_path)
    return xlsx_files


def process_xlsx_files(input_files, output_directory, log_file, volume_threshold):
    with open(log_file, 'a') as log:
        log.write(f"Processing session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for file_path in input_files:
            filename = os.path.basename(file_path)
            if filename.endswith('.xlsx') and not filename.startswith('Eigens') and not filename.startswith(
                    'VolumeEigen') and not filename.startswith('Quantity'):
                sample_number = extract_sample_id_from_processed_xlsx(file_path)
                if not sample_number or sample_number == "NoSampleFound":
                    continue
                df = pd.read_excel(file_path)

                initial_count = len(df)
                df_filtered = df[(df['EigenVal1'] != 0) & (df['EigenVal2'] != 0) & (df['EigenVal3'] != 0)]
                eigenvalue_filtered_count = len(df_filtered)
                df_filtered = df_filtered[df_filtered['Anisotropy'] != 1]
                final_count = len(df_filtered)

                log.write(f"Initial spinel count: {initial_count}\n")
                log.write(f"Count after removing zero-eigenvalue spinels: {eigenvalue_filtered_count}\n")
                log.write(f"Count after removing spinels with Anisotropy = 1: {final_count}\n")

                # Save the filtered data with all columns
                total_path = os.path.join(output_directory, f"total{sample_number}.xlsx")
                df_filtered.to_excel(total_path, index=False)
                log.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, created total{sample_number} at {total_path}\n")

                # Find the volume column
                volume_column_name = next(
                    (col for col in df.columns if 'volume' in col.lower() and 'mm^3' in col.lower()), None)

                if volume_column_name and volume_threshold > 0:
                    df_filtered = df_filtered[df_filtered[volume_column_name] >= volume_threshold]
                    log.write(f"Applied volume threshold {volume_threshold} mm^3, remaining spinel count: {len(df_filtered)}\n")
                elif volume_threshold > 0:
                    log.write(f"Warning: no volume column found in {filename}. Available columns: {', '.join(df.columns)}\n")
                    messagebox.showwarning("Column not found", f"No volume column found in {filename}. Skipping volume filtering for this file.")

                quantity_path = os.path.join(output_directory, f"Quantity_{sample_number}.xlsx")
                df_filtered.to_excel(quantity_path, index=False)
                log.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, created Quantity_{sample_number} at {quantity_path}\n")

                df_sorted = df_filtered.sort_values(by=df.columns[0], ascending=False)

                columns_to_extract = [
                    'EigenVal1', 'EigenVal2', 'EigenVal3',
                    'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',
                    'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',
                    'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z'
                ]
                df_extracted = df_sorted.loc[:, [col for col in columns_to_extract if col in df_sorted.columns]]
                df_extracted.replace(0, 0.00000001, inplace=True)
                eigens_path = os.path.join(output_directory, f"Eigens{sample_number}.xlsx")
                df_extracted.to_excel(eigens_path, index=False)

                if volume_column_name:
                    volume_eigen_columns = [volume_column_name] + [col for col in columns_to_extract if
                                                                   col in df_sorted.columns]
                    df_volume_eigen = df_sorted.loc[:, volume_eigen_columns]
                    df_volume_eigen.replace(0, 0.00000001, inplace=True)
                    volume_eigen_path = os.path.join(output_directory, f"VolumeEigen{sample_number}.xlsx")
                    df_volume_eigen.to_excel(volume_eigen_path, index=False)
                    log.write(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, created VolumeEigen{sample_number} at {volume_eigen_path}\n")
                    print(f"Created VolumeEigen{sample_number} in {output_directory}")
                else:
                    log.write(f"Warning: no volume column found in {filename}; cannot create VolumeEigen file\n")
                    print(f"Warning: no volume column found in {filename}; cannot create VolumeEigen file")

                log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"{log_time}, created Eigens{sample_number} at {eigens_path}, final rows: {len(df_extracted)}\n")
                print(f"Created Eigens{sample_number} in {output_directory}")


def select_input_files():
    files = filedialog.askopenfilenames(title="Select input files",
                                        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    return list(files)


def select_output_directory():
    directory = filedialog.askdirectory(title="Select output directory")
    return directory


def main():
    root = Tk()
    root.withdraw()

    input_files = select_input_files()
    if not input_files:
        print("No input files selected.")
        return

    output_directory = select_output_directory()
    if not output_directory:
        print("No output directory selected.")
        return

    volume_window = Tk()
    volume_window.title("Enter volume threshold")

    Label(volume_window, text="Volume threshold (mm^3, enter 0 to disable):").grid(row=0)
    volume_entry = Entry(volume_window)
    volume_entry.grid(row=0, column=1)

    def on_submit():
        volume_threshold = volume_entry.get()
        if not volume_threshold:
            print("No volume threshold provided.")
            return
        try:
            volume_threshold = float(volume_threshold)
        except ValueError:
            print("Invalid volume threshold format.")
            return

        volume_window.destroy()

        first_sample = find_first_sample_number(input_files)
        log_file = generate_log_filename(output_directory, first_sample)
        xlsx_files = convert_and_clean_csv(input_files, output_directory, log_file)

        # Process the converted xlsx files
        process_xlsx_files(xlsx_files, output_directory, log_file, volume_threshold)

        print(f"Volume threshold: {volume_threshold} mm^3")

    Button(volume_window, text="Submit", command=on_submit).grid(row=1, columnspan=2)

    volume_window.mainloop()


if __name__ == "__main__":
    main()
