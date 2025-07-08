########################################################################################################################
##################################################### IMPORTS ##########################################################
########################################################################################################################
import tkinter as tk
from tkinter import filedialog, simpledialog

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################

def name_user():
    '''
    This function open a window to ask the initials of the user.

    Parameters:
    -----------
    None

    Return:
    -------
    user_initials : str
        Are the user's intials (e.g. Ada Lovelace → AL)
    '''

    def on_submit():
        nonlocal user_initials
        user_initials = entry.get()
        dialog.destroy()

    user_initials = None
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    dialog = tk.Toplevel(root)
    dialog.title("Input")
    dialog.geometry("300x120")
    dialog.resizable(False, False)

    label = tk.Label(dialog, text="Put user initials \n e.g. Ada Lovelace → AL", font=("Arial", 12))
    label.pack(pady=10)

    entry = tk.Entry(dialog, font=("Arial", 12))
    entry.pack(pady=5)

    submit_button = tk.Button(dialog, text="Continue", command=on_submit, font=("Arial", 12))
    submit_button.pack(pady=5)

    dialog.grab_set()
    root.wait_window(dialog)

    root.destroy()

    return user_initials


def ask_wanted_morph():
    '''
    This function guides the user through a selection process:
    1. Choose study type (All / Bowl-shaped)
    2. If All → ask if zone already processed (Yes / No)
    3. If Yes → open file dialog to choose 6 shapefiles

    Parameters:
    -----------
    None

    Returns:
    --------
    main_choice: str
        Is the choice of the user between the study of all kind of craters or only the bowl-shaped ones

    selected_files: list
        Contains the paths of the previous shapefiles results
    '''

    choice = []
    bowl_processed = []
    selected_files = []

    # Step 1 — Study type
    def choose_all():
        choice.append("All")
        root.destroy()

    def choose_bowl():
        choice.append("Bowl-shaped")
        root.destroy()

    root = tk.Tk()
    root.title("Study type")
    root.geometry("400x150")
    root.resizable(False, False)

    label = tk.Label(root, text="What do you want to study:", font=("Arial", 12))
    label.pack(pady=10)

    bouton_frame = tk.Frame(root)
    bouton_frame.pack(pady=10)

    btn_all = tk.Button(bouton_frame, text="All craters", width=22, command=choose_all)
    btn_bowl = tk.Button(bouton_frame, text="Only bowl-shaped craters", width=22, command=choose_bowl)

    btn_all.grid(row=0, column=0, padx=10)
    btn_bowl.grid(row=0, column=1, padx=10)

    root.mainloop()

    # Step 2 — Follow-up question
    if choice[0] == "All":
        def choose_yes():
            bowl_processed.append("Yes")
            second_root.destroy()

        def choose_no():
            bowl_processed.append("No")
            second_root.destroy()

        second_root = tk.Tk()
        second_root.title("Follow-up")
        second_root.geometry("450x150")
        second_root.resizable(False, False)

        label2 = tk.Label(second_root, text="Have you already processed this zone",
                          font=("Arial", 12), wraplength=400)
        label2.pack(pady=10)

        bouton_frame2 = tk.Frame(second_root)
        bouton_frame2.pack(pady=10)

        btn_yes = tk.Button(bouton_frame2, text="Yes", width=15, command=choose_yes)
        btn_no = tk.Button(bouton_frame2, text="No", width=15, command=choose_no)

        btn_yes.grid(row=0, column=0, padx=15)
        btn_no.grid(row=0, column=1, padx=15)

        second_root.mainloop()

        # Step 3 — File selection window
        if bowl_processed[0] == "Yes":
            def browse_files():
                files = filedialog.askopenfilenames(
                    title="Select the 6 shapefiles of the latest study",
                    filetypes=[("Shapefiles", "*.shp"), ("All files", "*.*")]
                )
                if files:
                    selected_files.extend(files)
                third_root.destroy()

            third_root = tk.Tk()
            third_root.title("Choose your files")
            third_root.geometry("500x180")
            third_root.resizable(False, False)

            msg = "Choose your files.\n \n " \
                  "⚠️Be careful, you need to choose the 6 shapefiles (centers, global-results, highest_points, " \
                  "lowest_points, rim, slopes)"
            label3 = tk.Label(third_root, text=msg, font=("Arial", 12), wraplength=480, justify="center")
            label3.pack(pady=20)

            btn_browse = tk.Button(third_root, text="📁 Choose files", command=browse_files, width=20)
            btn_browse.pack(pady=10)

            third_root.mainloop()

    return choice[0], selected_files if selected_files else None


def data_recovery(global_results_shp, rim_shp, centers_shp, lowest_shp, highest_shp, slopes_shp, crater_id, nac_id,
                  highest_points, results_slopes, result_geom_select_crat, centers, lowest_points, rim_approx):

    '''
    This function fills the shapefile with existing data.

    Parameters:
    -----------
    global_results_shp: GeoDataFrame
        Contains the previous global-results of the algorithm previous run

    rim_shp: GeoDataFrame
        Contains the previous rim results of the algorithm previous run

    centers_shp: GeoDataFrame
        Contains the previous centers results of the algorithm previous run

    lowest_shp: GeoDataFrame
        Contains the previous  lowest points results of the algorithm previous run

    highest_shp: GeoDataFrame
        Contains the previous highest points results of the algorithm previous run

    slopes_shp: GeoDataFrame
        Contains the previous slopes results of the algorithm previous run

    crater_id: int
        is the id of the studied crater

    nac_id: str
        is the id of the used DTM

    highest_points:list
        Is the list that will be used to create the highest points shapefile

    results_slopes: list
        Is the list that will be used to create the slopes shapefile

    result_geom_select_crat: list
        Is the list that will be used to create the global-results shapefile

    centers: list
        Is the list that will be used to create the centers shapefile

    lowest_points: list
        Is the list that will be used to create the lowest points shapefile

    rim_approx: list
        Is the list that will be used to create the rim shapefile

    Return:
    -------
    None
    '''

    # Find the row in shapefiles corresponding of teh studied crater
    matched_row_global = global_results_shp[global_results_shp['run_id'] == crater_id]
    matched_row_rim = rim_shp[rim_shp['run_id'] == crater_id]
    matched_row_centers = centers_shp[centers_shp['run_id'] == crater_id]
    matched_row_lowest = lowest_shp[lowest_shp['run_id'] == crater_id]
    matched_row_highest = highest_shp[highest_shp['run_id'] == crater_id]
    matched_row_slopes = slopes_shp[slopes_shp['run_id'] == crater_id]

    row_global = matched_row_global.iloc[0]
    row_rim = matched_row_rim.iloc[0]
    row_centers = matched_row_centers.iloc[0]
    row_lowest = matched_row_lowest.iloc[0]

    common_attrs = {
        'run_id': crater_id,
        'NAC_DTM_ID': nac_id
    }

    for _, row in matched_row_highest.iterrows():
        highest_points.append({
            'geometry': row.geometry,
            **common_attrs,
            'long': row['long'],
            'lat': row['lat'],
            'max_alt': row['max_alt'],
            'position': row['position']
        })

    for _, row in matched_row_slopes.iterrows():
        results_slopes.append({
            'geometry': row.geometry,
            **common_attrs,
            'position': row['position'],
            'slopeStopa': row['slopeStopa'],
            'δStopar': row['δStopar'],
            'meanStopar': row['meanStopar']
        })

    result_geom_select_crat.append({
        'geometry': row_global.geometry,
        **common_attrs,
        "morphology": row_global['morphology'],
        "deterior": row_global['deterior'],
        'center_lon': row_global['center_lon'],
        'center_lat': row_global['center_lat'],
        'ray_maxdia': row_global['ray_maxdia'],
        'mean_diam': row_global['mean_diam'],
        'δ_D': row_global['δ_D'],
        'mean_depth': row_global['mean_depth'],
        'δ_d_1': row_global['δ_d_1'],
        'ratio_dD': row_global['ratio_dD'],
        'δ_dD': row_global['δ_dD'],
        'circu': row_global['circu'],
        'mean_slope': row_global['mean_slope'],
        'mean TRI': row_global['mean TRI'],
        'swirl': row_global['swirl'],
        'hiesinger': row_global['hiesinger']
    })

    # Crater's center
    centers.append({
        'geometry': row_centers.geometry,
        **common_attrs,
        'center_lon': row_centers['center_lon'],
        'center_lat': row_centers['center_lat']
    })

    # Lowest point
    lowest_points.append({
        'geometry': row_lowest.geometry,
        **common_attrs,
        'alt': row_lowest['alt'],
        'position': row_lowest['position']
    })

    # Approximative rim
    rim_approx.append({
        'geometry': row_rim.geometry,
        **common_attrs
    })
