import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os


def convert_dataframe(df):
    eigenvalues = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].astype(float)
    if eigenvalues.isna().any().any() or (eigenvalues <= 0).any().any():
        raise ValueError('EigenVal1-3 must be finite and strictly positive')
    radii = (5.0 * eigenvalues) ** 0.5

    return pd.DataFrame({
        'Number': df['index'],
        'Component': 'spinel',
        'Unique#': df['index'],
        'Volume (mm^3)': df['Volume3d (mm^3) '],
        'PEllipsoid X (mm)': df['BaryCenterX (mm) '],
        'PEllipsoid Y (mm)': df['BaryCenterY (mm) '],
        'PEllipsoid Z (mm)': df['BaryCenterZ (mm) '],
        'PEllipsoid Rad1 (mm)': radii['EigenVal1'],
        'PEllipsoid Rad2 (mm)': radii['EigenVal2'],
        'PEllipsoid Rad3 (mm)': radii['EigenVal3'],
        'PEllipsoid X1 (dmls)': df['EigenVec1X'],
        'PEllipsoid Y1 (dmls)': df['EigenVec1Y'],
        'PEllipsoid Z1 (dmls)': df['EigenVec1Z'],
        'PEllipsoid X2 (dmls)': df['EigenVec2X'],
        'PEllipsoid Y2 (dmls)': df['EigenVec2Y'],
        'PEllipsoid Z2 (dmls)': df['EigenVec2Z'],
        'PEllipsoid X3 (dmls)': df['EigenVec3X'],
        'PEllipsoid Y3 (dmls)': df['EigenVec3Y'],
        'PEllipsoid Z3 (dmls)': df['EigenVec3Z']
    })


def process_file(filepath, output_dir):
    # Determine the file extension
    file_extension = os.path.splitext(filepath)[1]

    # Choose the reader engine based on the extension
    if file_extension == '.xls':
        df = pd.read_excel(filepath, engine='xlrd')
    elif file_extension == '.xlsx':
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        print(f"Unsupported file format: {file_extension}")
        return

    df_new = convert_dataframe(df)

    # Get the sample id from the filename
    sample_number = os.path.splitext(os.path.basename(filepath))[0]

    # Build the output filename (tab-separated; the .xls extension lets
    # TomoFab / Excel open it directly)
    tsv_filename = f'TT_{sample_number}.xls'

    # Full output path
    tsv_filepath = os.path.join(output_dir, tsv_filename)

    # Save as a tab-separated file
    df_new.to_csv(tsv_filepath, sep='\t', index=False)

    print(f'File {tsv_filepath} has been created successfully.')


def batch_process_files():
    # Create and hide the Tk root window
    root = tk.Tk()
    root.withdraw()

    # File dialog to select the input Excel files
    filepaths = filedialog.askopenfilenames(filetypes=[("Excel files", "*.xls *.xlsx")])

    if not filepaths:
        print("No files selected.")
        return

    # File dialog to select the output directory
    output_dir = filedialog.askdirectory(title="Select Output Directory")

    if not output_dir:
        print("No output directory selected.")
        return

    for filepath in filepaths:
        process_file(filepath, output_dir)


if __name__ == "__main__":
    batch_process_files()
