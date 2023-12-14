import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
import analyse
import filters
import interactive
import hole_anal
import toSTL
import numpy as np
import matplotlib.pyplot as plt
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
            filtered_low_freq = [0]
            filtered_H_freq = [0]
            filtered_mag = [0]
            filtered_ph = [0]
           
            orig = data_dict[parameter]
            selected_filter = filter_var.get()
            if(selected_filter==2):
                probe, background = interactive.select_area(orig)
                filtered_H_freq = filters.pre_cut_filter_horizontal_freq(orig, background)
                filtered_low_freq = filters.pre_cut_filter_frequency(filtered_H_freq, 'low') #TBD
                forSTL = np.where(~np.isnan(probe), filtered_H_freq, np.nan)
                toSTL.makeSTL(forSTL, file_path)
            elif(selected_filter==1):
                filtered_H_freq = filters.filter_horizontal_freq(orig)
                filtered_low_freq = filters.filter_frequency(filtered_H_freq, 'low')

            elif(selected_filter==3):
                probe, background = analyse.cut_me2(orig)
                filtered_H_freq = filters.pre_cut_filter_horizontal_freq(orig, background)
                filtered_low_freq = filters.pre_cut_filter_frequency(filtered_H_freq, 'low') #TBD
                forSTL = np.where(~np.isnan(probe), filtered_H_freq, np.nan)
                toSTL.makeSTL(forSTL, file_path)

            if analyse_holes_var.get():
                colorshehe = analyse.visualize_data(phase, magnitude)
                hole_anal.display_image_with_hover_info(colorshehe,file_path,orig_real, orig_imag, orig_phase, orig_magnitude)
                plt.show()

        except Exception as e:
            error_info = traceback.format_exc()
            print(error_info)
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

filter_var = tk.IntVar(value=1)
analyse_holes_var = tk.BooleanVar(value=False)

option1_rb = tk.Radiobutton(root, text="Casual", variable=filter_var, value=1)
option1_rb.pack()

option2_rb = tk.Radiobutton(root, text="ja sam", variable=filter_var, value=2)
option2_rb.pack()

option3_rb = tk.Radiobutton(root, text="auto cut", variable=filter_var, value=3)
option3_rb.pack()

analyse_holes_button = tk.Checkbutton(root, text="Analyse Holes", variable=analyse_holes_var)
analyse_holes_button.pack()

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

