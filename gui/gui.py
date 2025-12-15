import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk

from functions.mosaic_functions_v2 import bayer_mosaic_generator
from functions.synthetic_image_functions import generate_synthetic_images
from functions.nearest_neighbour import nearest_neighbour
from functions.analysis import run_analysis
from functions.noise import add_noise

# --- Placeholder Functions ---

import subprocess
import sys
from pathlib import Path


def bilinear_interpolation(input_path, output_path):
    """Placeholder for Bilinear Interpolation demosaicing."""
    print(f"Running Bilinear Interpolation demosaicing from: {input_path} to: {output_path}")
    pass

def bicubic_interpolation(input_path, output_path):
    """Placeholder for Bicubic Interpolation demosaicing."""
    print(f"Running Bicubic Interpolation demosaicing from: {input_path} to: {output_path}")
    pass

def malvar_he_cutler_mhc(input_path, output_path):
    """Placeholder for Malvar-He-Cutler (MHC) demosaicing."""
    print(f"Running Malvar-He-Cutler (MHC) demosaicing from: {input_path} to: {output_path}")
    pass

def frequency_reconstruction_fourier(input_path, output_path):
    """Placeholder for Frequency Reconstruction (Fourier) demosaicing."""
    print(f"Running Frequency Reconstruction (Fourier) demosaicing from: {input_path} to: {output_path}")
    pass

def total_variation_regularization_tv(input_path, output_path):
    """Placeholder for Total Variation Regularization (TV) demosaicing."""
    print(f"Running Total Variation Regularization (TV) demosaicing from: {input_path} to: {output_path}")
    pass

def cnn_based_reconstruction(input_path, output_path):

    import subprocess, sys
    from pathlib import Path

    cnn_root = Path(__file__).resolve().parents[1] / "functions" / "cnn"

    cmd = [
        sys.executable,
        "-m", "demosaicnet_ours",
        "--input_dir", str(input_path),
        "--output_dir", str(output_path),
        "--noise", "0.0",
    ]

    subprocess.run(cmd, cwd=str(cnn_root), check=True)

    print(f"Running CNN-Based Reconstruction demosaicing from: {input_path} to: {output_path}")

    


# Dictionary to map dropdown names to the actual functions
DEMOSAIC_METHODS = {
    "Nearest Neighbour": nearest_neighbour,
    "Bilinear Interpolation": bilinear_interpolation,
    "Bicubic Interpolation": bicubic_interpolation,
    "Malvar-He-Cutler (MHC)": malvar_he_cutler_mhc,
    "Frequency Reconstruction (Fourier)": frequency_reconstruction_fourier,
    "Total Variation Regularization (TV)": total_variation_regularization_tv,
    "CNN-Based Reconstruction": cnn_based_reconstruction,
}


class ImageProcessingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing Tool")

        # --- Variables to store paths and selections ---
        self.synth_output_folder = tk.StringVar(value="No folder selected.")
        self.noise_input_folder = tk.StringVar(value="No folder selected.")
        self.noise_output_folder = tk.StringVar(value="No folder selected.")
        self.mosaic_input_folder = tk.StringVar(value="No folder selected.")
        self.mosaic_output_folder = tk.StringVar(value="No folder selected.")
        

        self.demosaic_input_folder = tk.StringVar(value="No folder selected.")
        self.demosaic_output_folder = tk.StringVar(value="No folder selected.")
        self.demosaic_method = tk.StringVar()
        
        self.analysis_original_folder = tk.StringVar(value="No folder selected.")
        self.analysis_reconstructed_folder = tk.StringVar(value="No folder selected.")
        self.analysis_output_folder = tk.StringVar(value="No folder selected.")
        self.analysis_method = tk.StringVar()
        
        self.image_list_mosaic = tk.StringVar()
        self.image_list_demosaic = tk.StringVar()
        
        # --- Image display variables (Pillow requires global/instance reference) ---
        self.current_mosaic_image = None
        self.current_demosaic_image = None
        
        # --- Notebook (Tabs) ---
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(pady=10, padx=10, expand=True, fill='both')

        # Create the four tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text='Welcome')
        self.notebook.add(self.tab2, text='Bayer Mosaicing')
        self.notebook.add(self.tab3, text='Demosaicing')
        self.notebook.add(self.tab4, text='Analysis')

        # Initialize content for each tab
        self._setup_welcome_tab(self.tab1)
        self._setup_bayer_mosaicing_tab(self.tab2)
        self._setup_demosaicing_tab(self.tab3)
        self._setup_analysis_tab(self.tab4)

    # --- Helper Functions ---

    def _choose_folder(self, path_var):
        """Opens a directory chooser dialog and updates a StringVar."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            path_var.set(folder_selected)
            # Special logic for image view tabs to refresh the file list
            if path_var == self.mosaic_input_folder:
                self._update_image_list_view(folder_selected, self.image_list_mosaic, self.mosaic_listbox, self.mosaic_image_label)
            elif path_var == self.demosaic_input_folder:
                self._update_image_list_view(folder_selected, self.image_list_demosaic, self.demosaic_listbox, self.demosaic_image_label)

    def _update_image_list_view(self, folder_path, list_var, listbox, image_label):
        """Clears the listbox, lists image files in the folder, and clears the displayed image."""
        try:
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.ppm'))]
            listbox.delete(0, tk.END)
            for file in image_files:
                listbox.insert(tk.END, file)
            
            # Clear currently displayed image
            image_label.config(image='')
            image_label.image = None
            
        except Exception as e:
            listbox.delete(0, tk.END)
            listbox.insert(tk.END, "Error reading folder.")
            print(f"Error updating image list: {e}")
            
    def _display_selected_image(self, event, folder_path_var, listbox, image_label, image_ref_var):
        """Displays the selected image from the listbox."""
        selected_indices = listbox.curselection()
        if not selected_indices:
            return
            
        selected_file = listbox.get(selected_indices[0])
        full_path = os.path.join(folder_path_var.get(), selected_file)
        
        try:
            img = Image.open(full_path)
            # Resize image to fit in the GUI, maintaining aspect ratio
            max_size = (300, 300)
            img.thumbnail(max_size)
            
            # Convert the PIL Image object to a Tkinter PhotoImage object
            # Note: Must keep a reference to the PhotoImage object to prevent garbage collection!
            tk_img = ImageTk.PhotoImage(img)
            
            image_label.config(image=tk_img)
            image_label.image = tk_img # Store a reference
            
            # Update the instance's reference variable
            if image_ref_var == 'mosaic':
                self.current_mosaic_image = tk_img
            elif image_ref_var == 'demosaic':
                self.current_demosaic_image = tk_img
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not open or display image: {e}")
            image_label.config(image='')
            image_label.image = None
            

    # --- Tab 1: Welcome ---
    def _setup_welcome_tab(self, tab):
        """Sets up the content for the Welcome tab."""
        # Title of Project
        title = ttk.Label(tab, text="Image Demosaicing & Analysis Tool", font=("Helvetica", 16, "bold"))
        title.pack(pady=20)

        # List of Names
        names_label = ttk.Label(tab, text="Developers:", font=("Helvetica", 12, "underline"))
        names_label.pack(pady=(10, 5))

        names = [
            "Mate Batinović", "Arjana Ivković", "Nikolas Jarić",
            "Lara Kustić", "Rebeka Naglić", "Gabriela Perković",
            "Vida Trlek", "Dora Zaninović", "Ema Zebić"
        ]
        
        for name in names:
            ttk.Label(tab, text=name).pack()

        # Year
        year_label = ttk.Label(tab, text="\nYear: 2025./2026.", font=("Helvetica", 12, "italic"))
        year_label.pack(pady=20)

    # --- Tab 2: Bayer Mosaicing ---
    def _setup_bayer_mosaicing_tab(self, tab):
        """Sets up the content for the Bayer Mosaicing tab."""
        # Split tab into two main frames (Left/Right)
        main_frame = ttk.Frame(tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='y', padx=(0, 10))

        right_frame = ttk.Frame(main_frame, relief='groove', padding=10)
        right_frame.pack(side='right', expand=True, fill='both')

        # --- Left Section: Synthetic Images ---
        synth_frame = ttk.LabelFrame(left_frame, text="Synthetic Images", padding=10)
        synth_frame.pack(pady=10, padx=5, fill='x')

        # Choose output folder button
        btn_choose_synth_out = ttk.Button(synth_frame, text="Choose output folder for synthetic images", command=lambda: self._choose_folder(self.synth_output_folder))
        btn_choose_synth_out.pack(pady=5, fill='x')

        # Path display
        lbl_synth_out_path = ttk.Label(synth_frame, textvariable=self.synth_output_folder, wraplength=300)
        lbl_synth_out_path.pack(pady=5)

        # Run synthetic image generator button
        btn_run_synth = ttk.Button(synth_frame, text="Run synthetic image generator", command=self._run_synthetic_generator)
        btn_run_synth.pack(pady=10, fill='x')

        # Result message label
        self.lbl_synth_result = ttk.Label(synth_frame, text="", foreground="green")
        self.lbl_synth_result.pack(pady=5)

        # --- Left Section: Add Noise ---
        
        ttk.Label(left_frame, text="Add Noise", font=("Helvetica", 12, "bold")).pack(pady=(20, 5), padx=5, anchor='w')

        # Choose input folder button
        btn_choose_noise_in = ttk.Button(left_frame, text="Choose input folder", command=lambda: self._choose_folder(self.noise_input_folder))
        btn_choose_noise_in.pack(pady=5, fill='x', padx=5)

        # Path display
        lbl_noise_in_path = ttk.Label(left_frame, textvariable=self.noise_input_folder, wraplength=300)
        lbl_noise_in_path.pack(pady=5, padx=5)

        # Choose output folder button
        btn_choose_noise_out = ttk.Button(left_frame, text="Choose output folder for mosaic", command=lambda: self._choose_folder(self.noise_output_folder))
        btn_choose_noise_out.pack(pady=5, fill='x', padx=5)
        
        # Path display
        lbl_noise_out_path = ttk.Label(left_frame, textvariable=self.noise_output_folder, wraplength=300)
        lbl_noise_out_path.pack(pady=5, padx=5)

        # Run Add Noise button
        btn_run_noise = ttk.Button(left_frame, text="Run Add Noise", command=self._run_add_noise)
        btn_run_noise.pack(pady=10, fill='x', padx=5)
        
        # Result message label
        self.lbl_noise_result = ttk.Label(left_frame, text="", foreground="green")
        self.lbl_noise_result.pack(pady=5, padx=5)
        
        # --- Left Section: Image Mosaicing ---
        
        ttk.Label(left_frame, text="Image Mosaicing", font=("Helvetica", 12, "bold")).pack(pady=(20, 5), padx=5, anchor='w')

        # Choose input folder button
        btn_choose_mosaic_in = ttk.Button(left_frame, text="Choose input folder", command=lambda: self._choose_folder(self.mosaic_input_folder))
        btn_choose_mosaic_in.pack(pady=5, fill='x', padx=5)

        # Path display
        lbl_mosaic_in_path = ttk.Label(left_frame, textvariable=self.mosaic_input_folder, wraplength=300)
        lbl_mosaic_in_path.pack(pady=5, padx=5)

        # Choose output folder button
        btn_choose_mosaic_out = ttk.Button(left_frame, text="Choose output folder for mosaic", command=lambda: self._choose_folder(self.mosaic_output_folder))
        btn_choose_mosaic_out.pack(pady=5, fill='x', padx=5)
        
        # Path display
        lbl_mosaic_out_path = ttk.Label(left_frame, textvariable=self.mosaic_output_folder, wraplength=300)
        lbl_mosaic_out_path.pack(pady=5, padx=5)

        # Run Bayer Mosaic Generator button
        btn_run_mosaic = ttk.Button(left_frame, text="Run Bayer Mosaic Generator", command=self._run_bayer_mosaic)
        btn_run_mosaic.pack(pady=10, fill='x', padx=5)
        
        # Result message label
        self.lbl_mosaic_result = ttk.Label(left_frame, text="", foreground="green")
        self.lbl_mosaic_result.pack(pady=5, padx=5)

        # --- Right Section: Image View ---
        ttk.Label(right_frame, text="Input Folder Image View", font=("Helvetica", 12, "bold")).pack(pady=5)
        
        # Listbox for files
        listbox_frame = ttk.Frame(right_frame)
        listbox_frame.pack(fill='x', pady=5)
        self.mosaic_listbox = tk.Listbox(listbox_frame, height=10)
        self.mosaic_listbox.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.mosaic_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.mosaic_listbox.config(yscrollcommand=scrollbar.set)
        
        # Bind click event to display image
        self.mosaic_listbox.bind('<<ListboxSelect>>', lambda e: self._display_selected_image(e, self.mosaic_input_folder, self.mosaic_listbox, self.mosaic_image_label, 'mosaic'))

        # Image display label
        self.mosaic_image_label = ttk.Label(right_frame, text="Click an image name to view it here.")
        self.mosaic_image_label.pack(pady=10)
        
    def _run_synthetic_generator(self):
        """Action for the Run synthetic image generator button."""
        output_path = self.synth_output_folder.get()
        if os.path.isdir(output_path):
            generate_synthetic_images(output_path)
            self.lbl_synth_result.config(text="Synthetic images generated successfully.")
        else:
            self.lbl_synth_result.config(text="Please choose a valid output folder.", foreground="red")

    def _run_add_noise(self):
        """Action for the Run Add Noise button."""
        input_path = self.noise_input_folder.get()
        output_path = self.noise_output_folder.get()
        
        if os.path.isdir(input_path) and os.path.isdir(output_path):
            add_noise(input_path, output_path)
            self.lbl_noise_result.config(text="Noise added", foreground="green")
        else:
            self.lbl_noise_result.config(text="Please choose valid input and output folders.", foreground="red")

    def _run_bayer_mosaic(self):
        """Action for the Run Bayer Mosaic Generator button."""
        input_path = self.mosaic_input_folder.get()
        output_path = self.mosaic_output_folder.get()
        
        if os.path.isdir(input_path) and os.path.isdir(output_path):
            bayer_mosaic_generator(input_path, output_path)
            self.lbl_mosaic_result.config(text="Bayer mosaic generated.", foreground="green")
        else:
            self.lbl_mosaic_result.config(text="Please choose valid input and output folders.", foreground="red")

    # --- Tab 3: Demosaicing ---
    def _setup_demosaicing_tab(self, tab):
        """Sets up the content for the Demosaicing tab."""
        # Split tab into two main frames (Left/Right)
        main_frame = ttk.Frame(tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        left_frame = ttk.Frame(main_frame, padding=10)
        left_frame.pack(side='left', fill='y', padx=(0, 10))

        right_frame = ttk.Frame(main_frame, relief='groove', padding=10)
        right_frame.pack(side='right', expand=True, fill='both')

        # --- Left Section: Controls ---
        
        # Choose input folder button
        btn_choose_demosaic_in = ttk.Button(left_frame, text="Choose input folder with mosaics", command=lambda: self._choose_folder(self.demosaic_input_folder))
        btn_choose_demosaic_in.pack(pady=5, fill='x')

        # Path display
        lbl_demosaic_in_path = ttk.Label(left_frame, textvariable=self.demosaic_input_folder, wraplength=300)
        lbl_demosaic_in_path.pack(pady=5)

        # Choose output folder button
        btn_choose_demosaic_out = ttk.Button(left_frame, text="Choose output folder for reconstructed images", command=lambda: self._choose_folder(self.demosaic_output_folder))
        btn_choose_demosaic_out.pack(pady=5, fill='x')
        
        # Path display
        lbl_demosaic_out_path = ttk.Label(left_frame, textvariable=self.demosaic_output_folder, wraplength=300)
        lbl_demosaic_out_path.pack(pady=5)
        
        # Dropdown menu (Combobox)
        ttk.Label(left_frame, text="Choose demosaicing method:").pack(pady=(15, 5), anchor='w')
        method_names = list(DEMOSAIC_METHODS.keys())
        self.demosaic_dropdown = ttk.Combobox(left_frame, textvariable=self.demosaic_method, values=method_names, state="readonly")
        self.demosaic_dropdown.pack(pady=5, fill='x')
        self.demosaic_dropdown.current(0) # Select the first item by default

        # Run method button
        btn_run_demosaic = ttk.Button(left_frame, text="Run method", command=self._run_demosaic_method)
        btn_run_demosaic.pack(pady=10, fill='x')
        
        # Result message label
        self.lbl_demosaic_result = ttk.Label(left_frame, text="", foreground="green")
        self.lbl_demosaic_result.pack(pady=5)
        
        # --- Right Section: Image View ---
        ttk.Label(right_frame, text="Input Folder Image View", font=("Helvetica", 12, "bold")).pack(pady=5)
        
        # Listbox for files
        listbox_frame = ttk.Frame(right_frame)
        listbox_frame.pack(fill='x', pady=5)
        self.demosaic_listbox = tk.Listbox(listbox_frame, height=10)
        self.demosaic_listbox.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.demosaic_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.demosaic_listbox.config(yscrollcommand=scrollbar.set)
        
        # Bind click event to display image
        self.demosaic_listbox.bind('<<ListboxSelect>>', lambda e: self._display_selected_image(e, self.demosaic_input_folder, self.demosaic_listbox, self.demosaic_image_label, 'demosaic'))

        # Image display label
        self.demosaic_image_label = ttk.Label(right_frame, text="Click an image name to view it here.")
        self.demosaic_image_label.pack(pady=10)
        
    def _run_demosaic_method(self):
        """Action for the Run method button on the Demosaicing tab."""
        input_path = self.demosaic_input_folder.get()
        output_path = self.demosaic_output_folder.get()
        method_name = self.demosaic_method.get()
        
        if not (os.path.isdir(input_path) and os.path.isdir(output_path) and method_name):
            self.lbl_demosaic_result.config(text="Please select valid folders and a method.", foreground="red")
            return
            
        # Get the corresponding function and run it
        func = DEMOSAIC_METHODS.get(method_name)
        if func:
            func(input_path, output_path)
            self.lbl_demosaic_result.config(text=f"'{method_name}' completed.", foreground="green")
        else:
            self.lbl_demosaic_result.config(text="Error: Unknown method selected.", foreground="red")


    # --- Tab 4: Analysis ---
    def _setup_analysis_tab(self, tab):
        """Sets up the content for the Analysis tab."""
        # Split tab into two main frames (Left/Right)
        main_frame = ttk.Frame(tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        left_frame = ttk.Frame(main_frame, padding=10)
        left_frame.pack(side='left', fill='y', padx=(0, 10))

        right_frame = ttk.Frame(main_frame, padding=10)
        right_frame.pack(side='right', expand=True, fill='both')
        
        # --- Left Section: Controls ---
        
        # Choose folder with original images
        btn_choose_original = ttk.Button(left_frame, text="Choose folder with original images", command=lambda: self._choose_folder(self.analysis_original_folder))
        btn_choose_original.pack(pady=5, fill='x')
        lbl_original_path = ttk.Label(left_frame, textvariable=self.analysis_original_folder, wraplength=300)
        lbl_original_path.pack(pady=5)

        # Choose folder with reconstructed images
        btn_choose_reconstructed = ttk.Button(left_frame, text="Choose folder with reconstructed images", command=lambda: self._choose_folder(self.analysis_reconstructed_folder))
        btn_choose_reconstructed.pack(pady=5, fill='x')
        lbl_reconstructed_path = ttk.Label(left_frame, textvariable=self.analysis_reconstructed_folder, wraplength=300)
        lbl_reconstructed_path.pack(pady=5)

        # Choose analysis.txt output folder
        btn_choose_analysis_out = ttk.Button(left_frame, text="Choose analysis.txt output folder", command=lambda: self._choose_folder(self.analysis_output_folder))
        btn_choose_analysis_out.pack(pady=5, fill='x')
        lbl_analysis_out_path = ttk.Label(left_frame, textvariable=self.analysis_output_folder, wraplength=300)
        lbl_analysis_out_path.pack(pady=5)

        # Choose method dropdown (Now only contains specific methods)
        ttk.Label(left_frame, text="Choose method:").pack(pady=(15, 5), anchor='w')
        # FIX: Removed "All Methods"
        method_names = list(DEMOSAIC_METHODS.keys()) 
        self.analysis_dropdown = ttk.Combobox(left_frame, textvariable=self.analysis_method, values=method_names, state="readonly")
        self.analysis_dropdown.pack(pady=5, fill='x')
        # Select the first item by default if the list isn't empty
        if method_names:
            self.analysis_dropdown.current(0)

        # Metrics display
        ttk.Label(left_frame, text="Metrics: MSE, PSNR, SSIM, MS-SSIM and Lab CIEDE2000", font=("Helvetica", 10)).pack(pady=10, anchor='w')

        # Analyze button
        btn_analyze = ttk.Button(left_frame, text="Analyze", command=self._run_analysis)
        btn_analyze.pack(pady=10, fill='x')
        
        # Result message label
        self.lbl_analysis_result = ttk.Label(left_frame, text="", foreground="green")
        self.lbl_analysis_result.pack(pady=5)
        
        # --- Right Section: Analysis Results (Updated to use Text widget) ---
        ttk.Label(right_frame, text="Analysis Results", font=("Helvetica", 12, "bold")).pack(pady=5)
        
        # Frame for Text widget and Scrollbar
        text_frame = ttk.Frame(right_frame)
        text_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text Widget
        # Using a monospaced font helps with table alignment
        self.txt_analysis_results = tk.Text(text_frame, wrap=tk.NONE, yscrollcommand=scrollbar.set, height=20, width=50, font=('Consolas', 10))
        self.txt_analysis_results.insert(tk.END, "Run analysis to view results (MSE, PSNR, etc.) here.")
        self.txt_analysis_results.config(state=tk.DISABLED) # Make it read-only initially
        self.txt_analysis_results.pack(side=tk.LEFT, expand=True, fill='both')
        
        # Connect scrollbar to Text widget
        scrollbar.config(command=self.txt_analysis_results.yview)

    def _run_analysis(self):
        """Action for the Analyze button."""
        original_folder = self.analysis_original_folder.get()
        reconstructed_folder = self.analysis_reconstructed_folder.get()
        analysis_output_folder = self.analysis_output_folder.get()
        method = self.analysis_method.get()
        
        if not (os.path.isdir(original_folder) and os.path.isdir(reconstructed_folder) and os.path.isdir(analysis_output_folder) and method):
            self.lbl_analysis_result.config(text="Please select all valid folders and a method.", foreground="red")
            return

        try:
            # 1. Run the analysis function
            run_analysis(original_folder, reconstructed_folder, analysis_output_folder, method)
            
            # 2. Read the analysis.txt file
            analysis_file_path = os.path.join(analysis_output_folder, "analysis.txt")
            if not os.path.exists(analysis_file_path):
                 self.lbl_analysis_result.config(text=f"Analysis for {method} complete, but analysis.txt not found.", foreground="red")
                 return
                 
            with open(analysis_file_path, 'r') as file:
                txt_output = file.read()

            # 3. Update the Text widget
            self.txt_analysis_results.config(state=tk.NORMAL) # Enable editing
            self.txt_analysis_results.delete('1.0', tk.END) # Clear existing text
            self.txt_analysis_results.insert(tk.END, txt_output) # Insert new text
            self.txt_analysis_results.config(state=tk.DISABLED) # Make it read-only again
            
            # 4. Update the status label
            self.lbl_analysis_result.config(text=f"Analysis for {method} complete. Results displayed.", foreground="green")

        except Exception as e:
            self.lbl_analysis_result.config(text=f"Error during analysis: {e}", foreground="red")
            self.txt_analysis_results.config(state=tk.NORMAL)
            self.txt_analysis_results.delete('1.0', tk.END)
            self.txt_analysis_results.insert(tk.END, f"Error: {e}")
            self.txt_analysis_results.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()