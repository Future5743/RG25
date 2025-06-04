########################################################################################################################
##################################################### IMPORTATIONS #####################################################
########################################################################################################################
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.colors import black
from PIL import Image
import os

########################################################################################################################
######################################################### CODE #########################################################
########################################################################################################################


def wrap_text(c, text, max_width, font_name="Helvetica", font_size=12):
    """
    Coupe un texte pour qu’il rentre dans une largeur max, retourne une liste de lignes.
    """
    c.setFont(font_name, font_size)
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if c.stringWidth(test_line, font_name, font_size) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def creer_rapport_cratere(id, zone, swirl, morph, center_long, center_lat, coord_low,
                          diam, incer_D_hoov, incer_D_stop, d, incer_d_hoov, incer_d_stop,
                          dtoD, incer_dD_hoov, incer_dD_stop, mill, mean_slope, slope, delta_slope):

    output_dir = f'results/RG{zone}/rapports'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"results/RG{zone}/rapports/rapport_{id}.pdf"
    page_width, page_height = A4
    c = canvas.Canvas(output_path, pagesize=A4)

    marge_gauche = 2 * cm
    marge_droite = 2 * cm

    # === Bandeau noir en haut ===
    bandeau_height = 2 * cm
    # === Bandeau noir ===
    c.setFillColor(black)
    c.rect(0, page_height - bandeau_height, page_width, bandeau_height, fill=1)

    # === Logos gauche et droite ===
    try:
        # Logo de droite
        logo_path = "Sign_GEODES_txtBlanc_fondTransp_avecHalo.png"
        with Image.open(logo_path) as logo:
            logo_width_px, logo_height_px = logo.size
            logo_aspect = logo_height_px / logo_width_px
            logo_target_height = 1 * cm
            logo_target_width = logo_target_height / logo_aspect
            x_logo_right = page_width - logo_target_width - 1 * cm
            y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
            c.drawImage(logo_path, x_logo_right, y_logo, width=logo_target_width, height=logo_target_height,
                        mask='auto')

        # Logo de gauche
        logo_left_path = "UdeS_logo_h_blancHR.png"
        with Image.open(logo_left_path) as logo_left:
            logo_width_px, logo_height_px = logo_left.size
            logo_aspect = logo_height_px / logo_width_px
            logo_target_height = 1.5 * cm
            logo_target_width = logo_target_height / logo_aspect
            x_logo_left = 1 * cm  # marge gauche
            y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
            c.drawImage(logo_left_path, x_logo_left, y_logo, width=logo_target_width, height=logo_target_height,
                        mask='auto')

    except Exception as e:
        print(f"⚠️ Erreur chargement logo : {e}")

    ESPACEMENT_SOUS_TITRE = 1 * cm  # ou 20 si tu travailles en points

    # === Titre centré ===
    c.setFont("Helvetica-Bold", 25)
    c.drawCentredString(page_width / 2, page_height - bandeau_height - 1 * cm, f"Rapport sur le cratère {id}")
    c.drawCentredString(page_width / 2, page_height - bandeau_height - 1 * cm - 1 * cm, f"de la zone RG{zone}")

    # === Sous-titre "Informations générales" ===
    c.setFont("Helvetica-Bold", 16)
    sous_titre_y = page_height - bandeau_height - 1 * cm - 1 * cm - 2 * cm  # 2 lignes de titre + 2 cm de marge
    c.drawString(marge_gauche, sous_titre_y, "Informations générales")

    image_path = f"results/RG{zone}/crater_img/crater_{id}.png"
    # === Image ===
    with Image.open(image_path) as img:
        img_width_px, img_height_px = img.size
        aspect_ratio = img_height_px / img_width_px
        img_target_width = 7 * cm
        img_target_height = img_target_width * aspect_ratio
        x_img = marge_gauche
        y_img = sous_titre_y - img_target_height - 1 * cm
        c.drawImage(image_path, x_img, y_img, width=img_target_width, height=img_target_height,
                    preserveAspectRatio=True)

    # === Texte à droite de l'image ===
    x_text = x_img + img_target_width + 1 * cm
    y_text = y_img + img_target_height
    text_max_width = page_width - x_text - marge_droite
    font_size = 11
    line_spacing = 16  # 14 + 2 points d'espace vertical entre blocs

    infos = [
        ("ID", id),
        ("Zone d'étude", f"RG{zone}"),
        ("Swirl", swirl),
        ("Morphologie", morph),
        ("Diamètre moyen", f"{diam}m ± {max(incer_D_hoov, incer_D_stop)}m"),
        ("Profondeur moyenne", f"{d}m ± {max(incer_d_hoov, incer_d_stop)}m"),
        ("Ration d/D", f"{dtoD} ± {max(incer_dD_hoov, incer_dD_stop)}"),
        ("Indice de circularité", f"{mill}"),
        ("Pente moyenne", f"{mean_slope}"),
        ("Coordonnées du centre géométrique", f"({center_long}, {center_lat})"),
        ("Coordonnées du point le plus bas du cratère", f"{coord_low}")
    ]

    for label, value in infos:
        # Préparer lignes avec wrapping
        full_text = f"{label} : {value}"
        lines = wrap_text(c, full_text, text_max_width, font_name="Helvetica", font_size=font_size)

        for i, line in enumerate(lines):
            if ':' in line:
                split_index = line.find(":")
                label_part = line[:split_index + 1]
                value_part = line[split_index + 1:].lstrip()

                # Écrire label en gras
                c.setFont("Helvetica-Bold", font_size)
                c.drawString(x_text, y_text, label_part)

                # Écrire valeur normale juste après le label
                x_value = x_text + c.stringWidth(label_part, "Helvetica-Bold", font_size) + 2
                c.setFont("Helvetica", font_size)
                c.drawString(x_value, y_text, value_part)
            else:
                # Si ligne sans ":" (cas rare), tout en normal
                c.setFont("Helvetica", font_size)
                c.drawString(x_text, y_text, line)
            y_text -= line_spacing

        # Ajouter espace entre les blocs d’infos
        y_text -= 4

    # === 6. Sous-titre "Informations sur les pentes" ===
    y_sous_titre_2 = y_text - ESPACEMENT_SOUS_TITRE
    show_subtitle = True  # Flag pour ne l'afficher qu'une fois

    if y_sous_titre_2 < 3 * cm:
        c.showPage()
        # Redessiner le bandeau noir et logo
        # === Bandeau noir ===
        c.setFillColor(black)
        c.rect(0, page_height - bandeau_height, page_width, bandeau_height, fill=1)

        # === Logos gauche et droite ===
        try:
            # Logo de droite
            with Image.open(logo_path) as logo:
                logo_width_px, logo_height_px = logo.size
                logo_aspect = logo_height_px / logo_width_px
                logo_target_height = 1 * cm
                logo_target_width = logo_target_height / logo_aspect
                x_logo_right = page_width - logo_target_width - 1 * cm
                y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                c.drawImage(logo_path, x_logo_right, y_logo, width=logo_target_width, height=logo_target_height,
                            mask='auto')

            # Logo de gauche
            logo_left_path = "UdeS_logo_h_blancHR.png"
            with Image.open(logo_left_path) as logo_left:
                logo_width_px, logo_height_px = logo_left.size
                logo_aspect = logo_height_px / logo_width_px
                logo_target_height = 1.5 * cm
                logo_target_width = logo_target_height / logo_aspect
                x_logo_left = 1 * cm  # marge gauche
                y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                c.drawImage(logo_left_path, x_logo_left, y_logo, width=logo_target_width, height=logo_target_height,
                            mask='auto')

        except Exception as e:
            print(f"⚠️ Erreur chargement logo : {e}")

        y_sous_titre_2 = page_height - bandeau_height - 1 * cm

    if show_subtitle:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(marge_gauche, y_sous_titre_2, "Informations sur les pentes")
        y_table = y_sous_titre_2 - 1 * cm  # 1 cm sous le sous-titre
        show_subtitle = False
    else:
        y_table = y_sous_titre_2  # déjà décrémenté

    # === 7. Génération du tableau des pentes avec saut de page si nécessaire ===

    table_data = [["Orientation par rapport au Nord", "Pente (°)", "Incertitude (°)"]]
    angles = list(range(0, 36))  # 0 à 350°, tous les 10°
    for angle in angles:
        if angle == 0:
            table_data.append([f"{angle * 10}/360°", slope[angle], delta_slope[angle]])
        else:
            table_data.append([f"{angle * 10}°", slope[angle], delta_slope[angle]])

    table_font_size = 10
    # Largeur totale disponible pour le tableau
    table_width = page_width - marge_gauche - marge_droite
    col_widths = [table_width * 0.4, table_width * 0.3, table_width * 0.3]  # 40% / 30% / 30%
    row_height = 0.7 * cm

    y_table = y_sous_titre_2 - ESPACEMENT_SOUS_TITRE

    for row_index, row in enumerate(table_data):
        # Vérifie si on a encore de la place sur la page
        if y_table < 2.5 * cm:
            c.showPage()
            # Redessiner bandeau et logo
            # === Bandeau noir ===
            c.setFillColor(black)
            c.rect(0, page_height - bandeau_height, page_width, bandeau_height, fill=1)

            # === Logos gauche et droite ===
            try:
                # Logo de droite
                with Image.open(logo_path) as logo:
                    logo_width_px, logo_height_px = logo.size
                    logo_aspect = logo_height_px / logo_width_px
                    logo_target_height = 1 * cm
                    logo_target_width = logo_target_height / logo_aspect
                    x_logo_right = page_width - logo_target_width - 1 * cm
                    y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                    c.drawImage(logo_path, x_logo_right, y_logo, width=logo_target_width, height=logo_target_height,
                                mask='auto')

                # Logo de gauche
                logo_left_path = "UdeS_logo_h_blancHR.png"
                with Image.open(logo_left_path) as logo_left:
                    logo_width_px, logo_height_px = logo_left.size
                    logo_aspect = logo_height_px / logo_width_px
                    logo_target_height = 1.5 * cm
                    logo_target_width = logo_target_height / logo_aspect
                    x_logo_left = 1 * cm  # marge gauche
                    y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                    c.drawImage(logo_left_path, x_logo_left, y_logo, width=logo_target_width, height=logo_target_height,
                                mask='auto')

            except Exception as e:
                print(f"⚠️ Erreur chargement logo : {e}")

            y_table = page_height - bandeau_height - 1 * cm  # recommencer 1cm sous le bandeau

        x = marge_gauche
        for i, cell in enumerate(row):
            c.setFont("Helvetica-Bold" if row_index == 0 else "Helvetica", table_font_size)
            c.rect(x, y_table, col_widths[i], row_height)  # Bordure
            c.drawString(x + 2, y_table + 0.2 * cm, str(cell))  # Texte dans cellule
            x += col_widths[i]
        y_table -= row_height

    def draw_justified_text(c, text, x, y, max_width, font_name="Helvetica", font_size=12, line_height=14):
        c.setFont(font_name, font_size)
        words = text.split()
        lines = []
        current_line = []

        def get_line_width(words):
            return sum(c.stringWidth(w, font_name, font_size) for w in words) + c.stringWidth(" ", font_name,
                                                                                              font_size) * (
                               len(words) - 1)

        for word in words:
            if get_line_width(current_line + [word]) <= max_width:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        if current_line:
            lines.append(current_line)

        for i, line_words in enumerate(lines):
            if i == len(lines) - 1:
                # dernière ligne : alignement à gauche simple
                text_line = " ".join(line_words)
                c.drawString(x, y, text_line)
            else:
                # Justifier la ligne
                line_width = get_line_width(line_words)
                space_count = len(line_words) - 1
                if space_count > 0:
                    space_width = (max_width - line_width) / space_count + c.stringWidth(" ", font_name, font_size)
                else:
                    space_width = 0
                pos_x = x
                for j, w in enumerate(line_words):
                    c.drawString(pos_x, y, w)
                    pos_x += c.stringWidth(w, font_name, font_size)
                    if j < space_count:
                        pos_x += space_width
            y -= line_height
        return y  # retourne la nouvelle coordonnée y après écriture du texte

    # Dans ta fonction principale, à partir de y_table :

    # 1. Position sous-titre Indice TRI
    y_sous_titre_tri = y_table - ESPACEMENT_SOUS_TITRE

    # Texte long justifié à afficher sous le sous-titre
    long_text = "L'indice TRI est un indice permettant de blablabla bla. Lorem ipsum lorem ipsum. hiefneianslbf phsefib" \
                "bslvbskib ohsgshrpvnfb shvlihvgsihv hp;sovbns"

    # hauteur minimale restante sous le sous-titre
    remaining_height = y_sous_titre_tri

    # Charger l'image pour calcul hauteur
    indice_tri_image_path = f"results/RG{zone}/TRI/TRI_{id}.png"
    with Image.open(indice_tri_image_path) as tri_img:
        img_width_px, img_height_px = tri_img.size
        aspect_ratio = img_height_px / img_width_px
        img_target_width = page_width - marge_gauche - marge_droite
        img_target_height = img_target_width * aspect_ratio

    # Estimation hauteur texte (approximative, 14 pts par ligne)
    font_size_text = 12
    line_height = font_size_text + 2
    max_text_width = page_width - marge_gauche - marge_droite
    # On calcule nombre de lignes en divisant la largeur totale des mots par max width
    text_lines_count = (len(long_text) * font_size_text) // max_text_width + 3  # approximation grossière

    text_height = text_lines_count * line_height

    # 1. Toujours écrire le sous-titre + texte sur la page actuelle
    if y_sous_titre_tri < 5 * cm:  # marge de sécurité avant d’écrire du texte
        c.showPage()
        # redessiner bandeau et logo
        # === Bandeau noir ===
        c.setFillColor(black)
        c.rect(0, page_height - bandeau_height, page_width, bandeau_height, fill=1)

        # === Logos gauche et droite ===
        try:
            # Logo de droite
            with Image.open(logo_path) as logo:
                logo_width_px, logo_height_px = logo.size
                logo_aspect = logo_height_px / logo_width_px
                logo_target_height = 1 * cm
                logo_target_width = logo_target_height / logo_aspect
                x_logo_right = page_width - logo_target_width - 1 * cm
                y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                c.drawImage(logo_path, x_logo_right, y_logo, width=logo_target_width, height=logo_target_height,
                            mask='auto')

            # Logo de gauche
            logo_left_path = "UdeS_logo_h_blancHR.png"
            with Image.open(logo_left_path) as logo_left:
                logo_width_px, logo_height_px = logo_left.size
                logo_aspect = logo_height_px / logo_width_px
                logo_target_height = 1.5 * cm
                logo_target_width = logo_target_height / logo_aspect
                x_logo_left = 1 * cm  # marge gauche
                y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                c.drawImage(logo_left_path, x_logo_left, y_logo, width=logo_target_width, height=logo_target_height,
                            mask='auto')

        except Exception as e:
            print(f"⚠️ Erreur chargement logo : {e}")

        y_sous_titre_tri = page_height - bandeau_height - ESPACEMENT_SOUS_TITRE

    # Sous-titre
    c.setFont("Helvetica-Bold", 16)
    c.drawString(marge_gauche, y_sous_titre_tri, "Indice TRI")

    # Texte justifié sous le sous-titre
    y_text_start = y_sous_titre_tri - ESPACEMENT_SOUS_TITRE
    y_after_text = draw_justified_text(
        c, long_text, marge_gauche, y_text_start,
        max_text_width, font_name="Helvetica",
        font_size=font_size_text, line_height=line_height
    )

    # 2. Calculer si l’image rentre sur cette page (sous le texte)
    if y_after_text - img_target_height - 1 * cm < 2.5 * cm:
        # Trop bas, donc image sur nouvelle page
        c.showPage()
        # redessiner bandeau et logo
        # === Bandeau noir ===
        c.setFillColor(black)
        c.rect(0, page_height - bandeau_height, page_width, bandeau_height, fill=1)

        # === Logos gauche et droite ===
        try:
            # Logo de droite
            with Image.open(logo_path) as logo:
                logo_width_px, logo_height_px = logo.size
                logo_aspect = logo_height_px / logo_width_px
                logo_target_height = 1 * cm
                logo_target_width = logo_target_height / logo_aspect
                x_logo_right = page_width - logo_target_width - 1 * cm
                y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                c.drawImage(logo_path, x_logo_right, y_logo, width=logo_target_width, height=logo_target_height,
                            mask='auto')

            # Logo de gauche
            logo_left_path = "UdeS_logo_h_blancHR.png"  # 🔁 Ton logo gauche ici
            with Image.open(logo_left_path) as logo_left:
                logo_width_px, logo_height_px = logo_left.size
                logo_aspect = logo_height_px / logo_width_px
                logo_target_height = 1.5 * cm
                logo_target_width = logo_target_height / logo_aspect
                x_logo_left = 1 * cm  # marge gauche
                y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                c.drawImage(logo_left_path, x_logo_left, y_logo, width=logo_target_width, height=logo_target_height,
                            mask='auto')

        except Exception as e:
            print(f"⚠️ Erreur chargement logo : {e}")

        y_img_tri = page_height - bandeau_height - 1 * cm - img_target_height
    else:
        # Elle rentre sous le texte
        y_img_tri = y_after_text - 0.5 * cm

    # Image centrée
    x_img_tri = marge_gauche + ((page_width - marge_gauche - marge_droite) - img_target_width) / 2
    c.drawImage(indice_tri_image_path, x_img_tri, y_img_tri, width=img_target_width, height=img_target_height,
                preserveAspectRatio=True, mask='auto')


    # === Sous-titre "Profils topographiques" ===
    ESPACEMENT_SOUS_TITRE = 1 * cm  # Assure-toi que cette constante est définie une seule fois en haut du fichier

    # Nouvelle page si pas assez de place pour sous-titre + au moins une image
    estimated_image_height = 5 * cm  # estimation moyenne avant chargement d'images
    if y_img_tri - estimated_image_height - ESPACEMENT_SOUS_TITRE < 3 * cm:
        c.showPage()
        # === Bandeau noir ===
        c.setFillColor(black)
        c.rect(0, page_height - bandeau_height, page_width, bandeau_height, fill=1)

        # === Logos gauche et droite ===
        try:
            # Logo de droite
            with Image.open(logo_path) as logo:
                logo_width_px, logo_height_px = logo.size
                logo_aspect = logo_height_px / logo_width_px
                logo_target_height = 1 * cm
                logo_target_width = logo_target_height / logo_aspect
                x_logo_right = page_width - logo_target_width - 1 * cm
                y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                c.drawImage(logo_path, x_logo_right, y_logo, width=logo_target_width, height=logo_target_height,
                            mask='auto')

            # Logo de gauche
            logo_left_path = "UdeS_logo_h_blancHR.png"
            with Image.open(logo_left_path) as logo_left:
                logo_width_px, logo_height_px = logo_left.size
                logo_aspect = logo_height_px / logo_width_px
                logo_target_height = 1.5 * cm
                logo_target_width = logo_target_height / logo_aspect
                x_logo_left = 1 * cm  # marge gauche
                y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                c.drawImage(logo_left_path, x_logo_left, y_logo, width=logo_target_width, height=logo_target_height,
                            mask='auto')

        except Exception as e:
            print(f"⚠️ Erreur chargement logo : {e}")

        y_profils = page_height - bandeau_height - ESPACEMENT_SOUS_TITRE
    else:
        y_profils = y_img_tri - ESPACEMENT_SOUS_TITRE

    # Afficher le sous-titre
    c.setFont("Helvetica-Bold", 16)
    c.drawString(marge_gauche, y_profils, "Profils topographiques")
    y_profils -= ESPACEMENT_SOUS_TITRE

    # === Affichage des 18 images ===
    profils_dir = f"results/RG{zone}/profils/{swirl}/{id}"
    for i in range(0, 18):
        image_path = os.path.join(profils_dir, f"Profil_{i*10}_{(i+18)*10}.png")
        if not os.path.exists(image_path):
            print(f"⚠️ Image manquante : {image_path}")
            continue
        try:
            with Image.open(image_path) as img:
                img_width_px, img_height_px = img.size
                img_target_width = page_width - marge_gauche - marge_droite
                aspect_ratio = img_height_px / img_width_px
                img_target_height = img_target_width * aspect_ratio

                # Vérifie si image tient sur la page actuelle
                if y_profils - img_target_height < 2.5 * cm:
                    c.showPage()
                    # === Bandeau noir ===
                    c.setFillColor(black)
                    c.rect(0, page_height - bandeau_height, page_width, bandeau_height, fill=1)

                    # === Logos gauche et droite ===
                    try:
                        # Logo de droite
                        with Image.open(logo_path) as logo:
                            logo_width_px, logo_height_px = logo.size
                            logo_aspect = logo_height_px / logo_width_px
                            logo_target_height = 1 * cm
                            logo_target_width = logo_target_height / logo_aspect
                            x_logo_right = page_width - logo_target_width - 1 * cm
                            y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                            c.drawImage(logo_path, x_logo_right, y_logo, width=logo_target_width,
                                        height=logo_target_height, mask='auto')

                        # Logo de gauche
                        logo_left_path = "UdeS_logo_h_blancHR.png"
                        with Image.open(logo_left_path) as logo_left:
                            logo_width_px, logo_height_px = logo_left.size
                            logo_aspect = logo_height_px / logo_width_px
                            logo_target_height = 1.5 * cm
                            logo_target_width = logo_target_height / logo_aspect
                            x_logo_left = 1 * cm  # marge gauche
                            y_logo = page_height - (bandeau_height / 2 + logo_target_height / 2)
                            c.drawImage(logo_left_path, x_logo_left, y_logo, width=logo_target_width,
                                        height=logo_target_height, mask='auto')

                    except Exception as e:
                        print(f"⚠️ Erreur chargement logo : {e}")

                    y_profils = page_height - bandeau_height - ESPACEMENT_SOUS_TITRE

                x_img = marge_gauche + (page_width - marge_gauche - marge_droite - img_target_width) / 2
                c.drawImage(image_path, x_img, y_profils - img_target_height,
                            width=img_target_width, height=img_target_height, preserveAspectRatio=True, mask='auto')

                y_profils -= img_target_height + ESPACEMENT_SOUS_TITRE

        except Exception as e:
            print(f"❌ Erreur chargement {image_path} : {e}")


    # === Finalisation ===
    c.showPage()
    c.save()  # <== AJOUTE CETTE LIGNE
    print(f"✅ Rapport PDF généré : {output_path}")
