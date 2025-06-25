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
    Splits a long text string into multiple lines so that each line fits within a specified width on a ReportLab canvas.

    Parameters:
    -----------
    c : reportlab.pdfgen.canvas.Canvas
        The ReportLab canvas object used to calculate text width and render text.

    text : str
        The full text string to be wrapped.

    max_width : float
        The maximum width (in points) that each line of text can occupy.

    font_name : str, optional
        The name of the font to be used when measuring text width (default is "Helvetica").

    font_size : int or float, optional
        The size of the font in points (default is 12).

    Returns:
    --------
    list of str
        A list of strings, each representing a line of text that fits within the specified width.
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


def draw_justified_text(c, text, x, y, max_width, font_name="Helvetica", font_size=12, line_height=14):
    """
    Draws justified text on a PDF canvas (ReportLab).

    Parameters:
    -----------
    c: (canvas.Canvas)
        ReportLab canvas object.

    text: str
        The paragraph to draw.

    x: float
        X coordinate (left margin).

    y: float
        Starting Y coordinate.

    max_width: float
        Maximum allowed width for the text block.

    font_name: str
        Font name.

    font_size: int
        Font size.

    line_height: int
        Line spacing in points.

    Returns:
    --------
    float: The Y coordinate after drawing the text (for continuation).
    """
    c.setFont(font_name, font_size)
    words = text.split()
    lines = []
    current_line = []

    def get_line_width(words):
        return sum(c.stringWidth(w, font_name, font_size) for w in words) + c.stringWidth(" ", font_name, font_size) * (
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
            c.drawString(x, y, " ".join(line_words))  # Last line left-aligned
        else:
            line_width = get_line_width(line_words)
            space_count = len(line_words) - 1
            if space_count > 0:
                space_width = (max_width - line_width) / space_count + c.stringWidth(" ", font_name, font_size)
            else:
                space_width = 0
            pos_x = x
            for j, word in enumerate(line_words):
                c.drawString(pos_x, y, word)
                pos_x += c.stringWidth(word, font_name, font_size)
                if j < space_count:
                    pos_x += space_width
        y -= line_height
    return y


def banner(c, page_width, page_height, bandeau_height):
    """
    Draws a black top banner with institutional logos.

    Parameters:
    -----------
    c: canvas.Canvas
        ReportLab canvas object.

    page_width: float
        Width of the page in points.

    page_height: float
        Height of the page in points.

    bandeau_height: float
        Height of the top banner.

    Returns:
    --------
    None
    """
    c.setFillColor(black)
    c.rect(0, page_height - bandeau_height, page_width, bandeau_height, fill=1)

    try:
        # Right logo
        logo_path = "logo/Sign_GEODES_txtBlanc_fondTransp_avecHalo.png"
        with Image.open(logo_path) as logo:
            logo_width_px, logo_height_px = logo.size
            aspect = logo_height_px / logo_width_px
            target_height = 1 * cm
            target_width = target_height / aspect
            x_logo = page_width - target_width - 1 * cm
            y_logo = page_height - (bandeau_height / 2 + target_height / 2)
            c.drawImage(logo_path, x_logo, y_logo, width=target_width, height=target_height, mask='auto')

        # Left logo
        logo_left_path = "logo/UdeS_logo_h_blancHR.png"
        with Image.open(logo_left_path) as logo_left:
            logo_width_px, logo_height_px = logo_left.size
            aspect = logo_height_px / logo_width_px
            target_height = 1.5 * cm
            target_width = target_height / aspect
            x_logo = 1 * cm
            y_logo = page_height - (bandeau_height / 2 + target_height / 2)
            c.drawImage(logo_left_path, x_logo, y_logo, width=target_width, height=target_height, mask='auto')

    except Exception as e:
        print(f"⚠️ Error loading logo: {e}")


def create_crater_report(id, zone, swirl, morph, state, center_long, center_lat, coord_low,
                          diam, incer_D_hoov, d, incer_d_hoov,
                          dtoD, incer_dD_hoov, mill, mean_slope, slope, delta_slope):
    """
    Generates a multi-page PDF report for a crater, including:
    - Header and logos
    - General info (ID, morphometry, coordinates)
    - Topography stats and slope table
    - TRI visualization
    - Topographic profiles

    Parameters:
    -----------
    id: str
        Crater ID

    zone: int
        Study zone number
    swirl: str
        Swirl name (used for folder structure)

    morph: str
        Morphological classification

    center_long: float
        Longitude of crater center

    center_lat: float
        Latitude of crater center

    coord_low: str
        Coordinates of the lowest point

    diam: float
        Mean diameter

    incer_D_hoov: float
        Diameter uncertainty

    d: float
        Mean depth

    incer_d_hoov: float
        Depth uncertainty

    dtoD: float
        Depth-to-diameter ratio

    incer_dD_hoov: float
        Uncertainty of d/D

    mill: float
        Circularity index

    mean_slope: float
        Average slope

    slope: list
        Slope per azimuth angle

    delta_slope: list
        Slope uncertainty per angle

    Returns:
    --------
    None
    """

    output_dir = f'results/RG{zone}/reports'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"results/RG{zone}/reports/report_{id}.pdf"
    page_width, page_height = A4
    c = canvas.Canvas(output_path, pagesize=A4)

    marge_gauche = 2 * cm
    marge_droite = 2 * cm

    # === Bandeau noir en haut ===
    bandeau_height = 2 * cm

    # === Logos gauche et droite ===
    banner(c, page_width, page_height, bandeau_height)

    ESPACEMENT_SOUS_TITRE = 1 * cm

    # === Title ===
    c.setFont("Helvetica-Bold", 25)
    c.drawCentredString(page_width / 2, page_height - bandeau_height - 1 * cm, f"Crater report {id}")
    c.drawCentredString(page_width / 2, page_height - bandeau_height - 1 * cm - 1 * cm, f"of RG{zone}")

    # === New part: General information ===
    c.setFont("Helvetica-Bold", 16)
    sous_titre_y = page_height - bandeau_height - 1 * cm - 1 * cm - 2 * cm  # 2 lignes de titre + 2 cm de marge
    c.drawString(marge_gauche, sous_titre_y, "General information")

    image_path = f"results/RG{zone}/crater_img/crater_{id}.png"

    with Image.open(image_path) as img:
        img_width_px, img_height_px = img.size
        aspect_ratio = img_height_px / img_width_px
        img_target_width = 7 * cm
        img_target_height = img_target_width * aspect_ratio
        x_img = marge_gauche
        y_img = sous_titre_y - img_target_height - 1 * cm
        c.drawImage(image_path, x_img, y_img, width=img_target_width, height=img_target_height,
                    preserveAspectRatio=True)

    x_text = x_img + img_target_width + 1 * cm
    y_text = y_img + img_target_height
    text_max_width = page_width - x_text - marge_droite
    font_size = 11
    line_spacing = 16

    infos = [
        ("ID", id),
        ("Study area", f"RG{zone}"),
        ("Swirl", swirl),
        ("Morphology", morph),
        ("State of degradation", state),
        ("Mean Diameter", f"{int(diam)}m ± {incer_D_hoov}m"),
        ("Mean depht", f"{'%.1f' % round(d, 1)}m ± {incer_d_hoov}m"),
        ("d/D ratio", f"{dtoD} ± {incer_dD_hoov}"),
        ("Circularity index", f"{mill}"),
        ("Mean slope", f"{mean_slope}°"),
        ("Geometric center coordinates", f"({center_long}, {center_lat})"),
        ("Coordinates of the crater's lowest point", f"{coord_low}")
    ]

    for label, value in infos:
        full_text = f"{label} : {value}"
        lines = wrap_text(c, full_text, text_max_width, font_name="Helvetica", font_size=font_size)

        for i, line in enumerate(lines):
            if ':' in line:
                split_index = line.find(":")
                label_part = line[:split_index + 1]
                value_part = line[split_index + 1:].lstrip()

                c.setFont("Helvetica-Bold", font_size)
                c.drawString(x_text, y_text, label_part)

                x_value = x_text + c.stringWidth(label_part, "Helvetica-Bold", font_size) + 2
                c.setFont("Helvetica", font_size)
                c.drawString(x_value, y_text, value_part)
            else:
                c.setFont("Helvetica", font_size)
                c.drawString(x_text, y_text, line)
            y_text -= line_spacing

        y_text -= 4

    # Nez part: general information on slopes ===
    y_sous_titre_2 = y_text - ESPACEMENT_SOUS_TITRE
    show_subtitle = True

    if y_sous_titre_2 < 3 * cm:
        c.showPage()
        banner(c, page_width, page_height, bandeau_height)

        y_sous_titre_2 = page_height - bandeau_height - 1 * cm

    if show_subtitle:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(marge_gauche, y_sous_titre_2, "Slopes data")
        y_table = y_sous_titre_2 - 1 * cm  # 1 cm sous le sous-titre
        show_subtitle = False
    else:
        y_table = y_sous_titre_2

    table_data = [["North orientation","Slope (°)", "Uncertainty (°)"]]
    angles = list(range(0, 36))
    for angle in angles:
        if angle == 0:
            table_data.append([f"{angle * 10}/360°", slope[angle], delta_slope[angle]])
        else:
            table_data.append([f"{angle * 10}°", slope[angle], delta_slope[angle]])

    table_font_size = 10
    table_width = page_width - marge_gauche - marge_droite
    col_widths = [table_width * 0.4, table_width * 0.3, table_width * 0.3]
    row_height = 0.7 * cm

    y_table = y_sous_titre_2 - ESPACEMENT_SOUS_TITRE

    for row_index, row in enumerate(table_data):
        if y_table < 2.5 * cm:
            c.showPage()
            banner(c, page_width, page_height, bandeau_height)

            y_table = page_height - bandeau_height - 1 * cm

        x = marge_gauche
        for i, cell in enumerate(row):
            c.setFont("Helvetica-Bold" if row_index == 0 else "Helvetica", table_font_size)
            c.rect(x, y_table, col_widths[i], row_height)
            c.drawString(x + 2, y_table + 0.2 * cm, str(cell))
            x += col_widths[i]
        y_table -= row_height

    y_sous_titre_tri = y_table - ESPACEMENT_SOUS_TITRE

    long_text = "The Topographic Roughness Index (TRI) is a measure used to quantify the ruggedeness or the " \
                "unevenness of terrain. It reflects how much elevation change over a given area."

    remaining_height = y_sous_titre_tri

    # Load image for height calculation
    indice_tri_image_path = f"results/RG{zone}/TRI/TRI_{id}.png"
    with Image.open(indice_tri_image_path) as tri_img:
        img_width_px, img_height_px = tri_img.size
        aspect_ratio = img_height_px / img_width_px
        img_target_width = page_width - marge_gauche - marge_droite
        img_target_height = img_target_width * aspect_ratio

    font_size_text = 11
    line_height = font_size_text + 2
    max_text_width = page_width - marge_gauche - marge_droite

    text_lines_count = (len(long_text) * font_size_text) // max_text_width + 3

    text_height = text_lines_count * line_height

    if y_sous_titre_tri < 5 * cm:
        c.showPage()
        banner(c, page_width, page_height, bandeau_height)

        y_sous_titre_tri = page_height - bandeau_height - ESPACEMENT_SOUS_TITRE

    # === New part: Topographic profiles ===
    c.setFont("Helvetica-Bold", 16)
    c.drawString(marge_gauche, y_sous_titre_tri, "Topographic roughness index (TRI)")

    # Justified text under subtitle
    y_text_start = y_sous_titre_tri - ESPACEMENT_SOUS_TITRE
    y_after_text = draw_justified_text(
        c, long_text, marge_gauche, y_text_start,
        max_text_width, font_name="Helvetica",
        font_size=font_size_text, line_height=line_height
    )

    # Computation to see if the picture is between two pages
    if y_after_text - img_target_height - 1 * cm < 2.5 * cm:
        c.showPage()
        banner(c, page_width, page_height, bandeau_height)

        y_img_tri = page_height - bandeau_height - 1 * cm - img_target_height
    else:
        y_img_tri = y_after_text - 0.5 * cm

    # Centered picture
    x_img_tri = marge_gauche + ((page_width - marge_gauche - marge_droite) - img_target_width) / 2
    c.drawImage(indice_tri_image_path, x_img_tri, y_img_tri, width=img_target_width, height=img_target_height,
                preserveAspectRatio=True, mask='auto')

    # === New part: Topographic profiles ===
    ESPACEMENT_SOUS_TITRE = 1 * cm

    estimated_image_height = 5 * cm
    if y_img_tri - estimated_image_height - ESPACEMENT_SOUS_TITRE < 3 * cm:
        c.showPage()
        banner(c, page_width, page_height, bandeau_height)
        y_profils = page_height - bandeau_height - ESPACEMENT_SOUS_TITRE
    else:
        y_profils = y_img_tri - ESPACEMENT_SOUS_TITRE

    # Display of the subtitle
    c.setFont("Helvetica-Bold", 16)
    c.drawString(marge_gauche, y_profils, "Topographic profiles")
    y_profils -= ESPACEMENT_SOUS_TITRE

    # === Display of 18 images ===
    profils_dir = f"results/RG{zone}/profiles/{swirl}/{id}"
    for i in range(0, 18):
        image_path = os.path.join(profils_dir, f"Profile_{i*10}_{(i+18)*10}.png")
        if not os.path.exists(image_path):
            print(f"⚠️ Missing picture : {image_path}")
            continue
        try:
            with Image.open(image_path) as img:
                img_width_px, img_height_px = img.size
                img_target_width = page_width - marge_gauche - marge_droite
                aspect_ratio = img_height_px / img_width_px
                img_target_height = img_target_width * aspect_ratio

                # Checks if image fits on current page
                if y_profils - img_target_height < 2.5 * cm:
                    c.showPage()
                    banner(c, page_width, page_height, bandeau_height)

                    y_profils = page_height - bandeau_height - ESPACEMENT_SOUS_TITRE

                x_img = marge_gauche + (page_width - marge_gauche - marge_droite - img_target_width) / 2
                c.drawImage(image_path, x_img, y_profils - img_target_height,
                            width=img_target_width, height=img_target_height, preserveAspectRatio=True, mask='auto')

                y_profils -= img_target_height + ESPACEMENT_SOUS_TITRE

        except Exception as e:
            print(f"❌ Loading error {image_path} : {e}")


    # === Finalization ===
    c.showPage()
    c.save()
    print(f"✅ PDF report generated : {output_path}")
