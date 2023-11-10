import tkinter as tk
from tkinter import filedialog, messagebox
import analyse
import filters

file_paths =[] 

def run_analysis():
    for file_path in file_paths:
        parameter = parameter_var.get()

        try:
            analyse.initialize_folders(file_path)
            real_data, imag_data, magnitude, phase, phaseMreal, imagMreal, orig_phase, orig_magnitude, orig_real, orig_imag = analyse.go_get_them(file_path)
        
            data_dict = {
                "real": real_data,
                "imag": imag_data,
                "magnitude": magnitude,
                "phase": phase,
                "phaseMreal": phaseMreal,
                "imagMreal": imagMreal
            }

            orig = data_dict[parameter]
            filtered_H_freq = filters.filter_horizontal_freq(orig)
            filtered_low_freq = filters.filter_frequency(filtered_H_freq, 'low')

        except Exception as e:
                 messagebox.showerror("Błąd", str(e))
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
    
# Utworzenie okna głównego
root = tk.Tk()
root.title("Analiza MDMA")

file_path_entry = tk.Entry(root, width=50)
file_path_entry.pack()

add_file_button = tk.Button(root, text="Dodaj plik", command=add_file)
add_file_button.pack()

clear_files_button = tk.Button(root, text="Wyczyść listę plików", command=clear_files)
clear_files_button.pack()

parameters = ["phase", "imag", "magnitude", "real", "phaseMreal", "imagMreal"]
parameter_var = tk.StringVar(root)
parameter_var.set(parameters[0]) 
parameter_menu = tk.OptionMenu(root, parameter_var, *parameters)
parameter_menu.pack()

run_button = tk.Button(root, text="Run", command=run_analysis)
run_button.pack()

root.mainloop()

