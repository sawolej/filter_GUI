import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
import analyse
import filters
import interactive
import toSTL
import numpy as np

file_paths =[] 
makeSTL = False

def toggle_makeSTL():
    global makeSTL
    makeSTL = makeSTL_var.get()



def run_analysis():
    for file_path in file_paths:
        parameter = parameter_var.get()

        try:
            analyse.initialize_folders(file_path)
            real_data, imag_data, magnitude, phase, phaseMreal, imagMreal, orig_phase, orig_magnitude, orig_real, orig_imag, combined = analyse.go_get_them(file_path)
        
            data_dict = {
                "real": real_data,
                "imag": imag_data,
                "magnitude": magnitude,
                "phase": phase,
                "phaseMreal": phaseMreal,
                "imagMreal": imagMreal,
                "combined": combined
            }
           
            orig = data_dict[parameter]
            if(makeSTL):
                probe, background = interactive.select_area(orig)
                filtered_H_freq = filters.pre_cut_filter_horizontal_freq(orig, background)
                #filtered_low_freq = filters.pre_cut_filter_frequency(filtered_H_freq, 'low') #TBD
                forSTL = np.where(~np.isnan(probe), filtered_H_freq, np.nan)
                toSTL.makeSTL(forSTL, file_path)     
            else:    
                filtered_H_freq = filters.filter_horizontal_freq(orig)
                filtered_low_freq = filters.filter_frequency(filtered_H_freq, 'low')

        except Exception as e:
            error_info = traceback.format_exc()
            messagebox.showerror("Błąd", f"Wystąpił wyjątek: {str(e)}\nSzczegóły: {error_info}")
           
    messagebox.showinfo("Sukces", f"Done and done")

def add_file():
    global file_paths
    file_path = filedialog.askopenfilename(filetypes=[("MDMA Files", "*.mdma")])
    if file_path:
        file_paths.append(file_path)  
        file_path_entry.insert(tk.END, file_path + "; ")

def clear_files():
    global file_paths
    file_paths.clear()  
    file_path_entry.delete(0, tk.END)  
    
root = tk.Tk()
root.title("Analiza MDMA")

makeSTL_var = tk.BooleanVar(value=makeSTL)

makeSTL_checkbox = tk.Checkbutton(root, text="'ja sam, ja sam', aka Make STL", var=makeSTL_var, command=toggle_makeSTL)
makeSTL_checkbox.pack()

file_path_entry = tk.Entry(root, width=50)
file_path_entry.pack()

add_file_button = tk.Button(root, text="Dodaj plik", command=add_file)
add_file_button.pack()

clear_files_button = tk.Button(root, text="Wyczyść listę plików", command=clear_files)
clear_files_button.pack()

parameters = ["phase", "imag", "magnitude", "real", "phaseMreal", "imagMreal", "combined"]
parameter_var = tk.StringVar(root)
parameter_var.set(parameters[0]) 
parameter_menu = tk.OptionMenu(root, parameter_var, *parameters)
parameter_menu.pack()

run_button = tk.Button(root, text="Run", command=run_analysis)
run_button.pack()

root.mainloop()

