########################################################################################################################
##################################################### IMPORTS ##########################################################
########################################################################################################################
import tkinter as tk

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################
def ask_wanted_morph():

    choice = []

    def choose_all():
        choice.append("all")
        root.destroy()

    def choose_bowl():
        choice.append("Bowl-shaped")
        root.destroy()

    root = tk.Tk()
    root.title("Study type")
    root.geometry("400x150")
    root.resizable(False, False)

    label = tk.Label(root, text="What do you want to study :", font=("Arial", 12))
    label.pack(pady=10)

    bouton_frame = tk.Frame(root)
    bouton_frame.pack(pady=10)

    btn_all = tk.Button(bouton_frame, text="All craters", width=22, command=choose_all)
    btn_bowl = tk.Button(bouton_frame, text="Only bowl-shaped craters", width=22, command=choose_bowl)

    btn_all.grid(row=0, column=0, padx=10)
    btn_bowl.grid(row=0, column=1, padx=10)

    root.mainloop()

    return choice[0]
