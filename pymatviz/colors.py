"""Colors used in pymatviz."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Final

    from pymatviz.typing import Rgb256ColorType, RgbColorType


# Element type based colors
ELEM_TYPE_COLORS: Final = {
    "Diatomic Nonmetal": "green",
    "Noble Gas": "purple",
    "Alkali Metal": "red",
    "Alkaline Earth Metal": "orange",
    "Metalloid": "darkgreen",
    "Polyatomic Nonmetal": "teal",
    "Transition Metal": "blue",
    "Post Transition Metal": "cyan",
    "Lanthanide": "brown",
    "Actinide": "gray",
    "Nonmetal": "green",
    "Halogen": "teal",
    "Metal": "lightblue",
    "Alkaline Metal": "magenta",
    "Transactinide": "olive",
}

# The following element-based colors are copied from matterviz:
# https://github.com/janosh/matterviz/blob/85a044cd/src/lib/colors.ts#L20-L242
ELEM_COLORS_JMOL_256: dict[str, Rgb256ColorType] = {
    "H": (255, 255, 255),
    "He": (217, 255, 255),
    "Li": (204, 128, 255),
    "Be": (194, 255, 0),
    "B": (255, 181, 181),
    "C": (144, 144, 144),
    "N": (48, 80, 248),
    "O": (255, 13, 13),
    "F": (144, 224, 80),
    "Ne": (179, 227, 245),
    "Na": (171, 92, 242),
    "Mg": (138, 255, 0),
    "Al": (191, 166, 166),
    "Si": (240, 200, 160),
    "P": (255, 128, 0),
    "S": (255, 255, 48),
    "Cl": (31, 240, 31),
    "Ar": (128, 209, 227),
    "K": (143, 64, 212),
    "Ca": (61, 255, 0),
    "Sc": (230, 230, 230),
    "Ti": (191, 194, 199),
    "V": (166, 166, 171),
    "Cr": (138, 153, 199),
    "Mn": (156, 122, 199),
    "Fe": (224, 102, 51),
    "Co": (240, 144, 160),
    "Ni": (80, 208, 80),
    "Cu": (200, 128, 51),
    "Zn": (125, 128, 176),
    "Ga": (194, 143, 143),
    "Ge": (102, 143, 143),
    "As": (189, 128, 227),
    "Se": (255, 161, 0),
    "Br": (166, 41, 41),
    "Kr": (92, 184, 209),
    "Rb": (112, 46, 176),
    "Sr": (0, 255, 0),
    "Y": (148, 255, 255),
    "Zr": (148, 224, 224),
    "Nb": (115, 194, 201),
    "Mo": (84, 181, 181),
    "Tc": (59, 158, 158),
    "Ru": (36, 143, 143),
    "Rh": (10, 125, 140),
    "Pd": (0, 105, 133),
    "Ag": (192, 192, 192),
    "Cd": (255, 217, 143),
    "In": (166, 117, 115),
    "Sn": (102, 128, 128),
    "Sb": (158, 99, 181),
    "Te": (212, 122, 0),
    "I": (148, 0, 148),
    "Xe": (66, 158, 176),
    "Cs": (87, 23, 143),
    "Ba": (0, 201, 0),
    "La": (112, 212, 255),
    "Ce": (255, 255, 199),
    "Pr": (217, 255, 199),
    "Nd": (199, 255, 199),
    "Pm": (163, 255, 199),
    "Sm": (143, 255, 199),
    "Eu": (97, 255, 199),
    "Gd": (69, 255, 199),
    "Tb": (48, 255, 199),
    "Dy": (31, 255, 199),
    "Ho": (0, 255, 156),
    "Er": (0, 230, 117),
    "Tm": (0, 212, 82),
    "Yb": (0, 191, 56),
    "Lu": (0, 171, 36),
    "Hf": (77, 194, 255),
    "Ta": (77, 166, 255),
    "W": (33, 148, 214),
    "Re": (38, 125, 171),
    "Os": (38, 102, 150),
    "Ir": (23, 84, 135),
    "Pt": (208, 208, 224),
    "Au": (255, 209, 35),
    "Hg": (184, 184, 208),
    "Tl": (166, 84, 77),
    "Pb": (87, 89, 97),
    "Bi": (158, 79, 181),
    "Po": (171, 92, 0),
    "At": (117, 79, 69),
    "Rn": (66, 130, 150),
    "Fr": (66, 0, 102),
    "Ra": (0, 125, 0),
    "Ac": (112, 171, 250),
    "Th": (0, 186, 255),
    "Pa": (0, 161, 255),
    "U": (0, 143, 255),
    "Np": (0, 128, 255),
    "Pu": (0, 107, 255),
    "Am": (84, 92, 242),
    "Cm": (120, 92, 227),
    "Bk": (138, 79, 227),
    "Cf": (161, 54, 212),
    "Es": (179, 31, 212),
    "Fm": (179, 31, 186),
    "Md": (179, 13, 166),
    "No": (189, 13, 135),
    "Lr": (199, 0, 102),
    "Rf": (204, 0, 89),
    "Db": (209, 0, 79),
    "Sg": (217, 0, 69),
    "Bh": (224, 0, 56),
    "Hs": (230, 0, 46),
    "Mt": (235, 0, 38),
}

# Scale color value to [0, 1] for matplotlib
ELEM_COLORS_JMOL: dict[str, RgbColorType] = {
    elem: (r / 255, g / 255, b / 255)
    for elem, (r, g, b) in ELEM_COLORS_JMOL_256.items()
}

ELEM_COLORS_VESTA_256: dict[str, Rgb256ColorType] = {
    "Ac": (112, 171, 250),
    "Ag": (192, 192, 192),
    "Al": (129, 178, 214),
    "Am": (84, 92, 242),
    "Ar": (207, 254, 196),
    "As": (116, 208, 87),
    "At": (117, 79, 69),
    "Au": (255, 209, 35),
    "B": (31, 162, 15),
    "Ba": (0, 201, 0),
    "Be": (94, 215, 123),
    "Bh": (224, 0, 56),
    "Bi": (158, 79, 181),
    "Bk": (138, 79, 227),
    "Br": (126, 49, 2),
    "C": (76, 76, 76),
    "Ca": (90, 150, 189),
    "Cd": (255, 217, 143),
    "Ce": (255, 255, 199),
    "Cf": (161, 54, 212),
    "Cl": (49, 252, 2),
    "Cm": (120, 92, 227),
    "Co": (0, 0, 175),
    "Cr": (0, 0, 158),
    "Cs": (87, 23, 143),
    "Cu": (34, 71, 220),
    "Db": (209, 0, 79),
    "Dy": (31, 255, 199),
    "Er": (0, 230, 117),
    "Es": (179, 31, 212),
    "Eu": (97, 255, 199),
    "F": (176, 185, 230),
    "Fe": (181, 113, 0),
    "Fm": (179, 31, 186),
    "Fr": (66, 0, 102),
    "Ga": (158, 227, 115),
    "Gd": (69, 255, 199),
    "Ge": (126, 110, 166),
    "H": (255, 204, 204),
    "He": (252, 232, 206),
    "Hf": (77, 194, 255),
    "Hg": (184, 184, 208),
    "Ho": (0, 255, 156),
    "Hs": (230, 0, 46),
    "I": (148, 0, 148),
    "In": (166, 117, 115),
    "Ir": (23, 84, 135),
    "K": (161, 33, 246),
    "Kr": (250, 193, 243),
    "La": (90, 196, 73),
    "Li": (134, 223, 115),
    "Lr": (199, 0, 102),
    "Lu": (0, 171, 36),
    "Md": (179, 13, 166),
    "Mg": (251, 123, 21),
    "Mn": (167, 8, 157),
    "Mo": (84, 181, 181),
    "Mt": (235, 0, 38),
    "N": (176, 185, 230),
    "Na": (249, 220, 60),
    "Nb": (115, 194, 201),
    "Nd": (199, 255, 199),
    "Ne": (254, 55, 181),
    "Ni": (183, 187, 189),
    "No": (189, 13, 135),
    "Np": (0, 128, 255),
    "O": (254, 3, 0),
    "Os": (38, 102, 150),
    "P": (192, 156, 194),
    "Pa": (0, 161, 255),
    "Pb": (87, 89, 97),
    "Pd": (0, 105, 133),
    "Pm": (163, 255, 199),
    "Po": (171, 92, 0),
    "Pr": (217, 255, 199),
    "Pt": (208, 208, 224),
    "Pu": (0, 107, 255),
    "Ra": (0, 125, 0),
    "Rb": (112, 46, 176),
    "Re": (38, 125, 171),
    "Rf": (204, 0, 89),
    "Rh": (10, 125, 140),
    "Rn": (66, 130, 150),
    "Ru": (36, 143, 143),
    "S": (255, 250, 0),
    "Sb": (158, 99, 181),
    "Sc": (181, 99, 171),
    "Se": (154, 239, 15),
    "Sg": (217, 0, 69),
    "Si": (27, 59, 250),
    "Sm": (143, 255, 199),
    "Sn": (154, 142, 185),
    "Sr": (0, 255, 0),
    "Ta": (77, 166, 255),
    "Tb": (48, 255, 199),
    "Tc": (59, 158, 158),
    "Te": (212, 122, 0),
    "Th": (0, 186, 255),
    "Ti": (120, 202, 255),
    "Tl": (166, 84, 77),
    "Tm": (0, 212, 82),
    "U": (0, 143, 255),
    "V": (229, 25, 0),
    "W": (33, 148, 214),
    "Xe": (66, 158, 176),
    "Y": (148, 255, 255),
    "Yb": (0, 191, 56),
    "Zn": (143, 143, 129),
    "Zr": (0, 255, 0),
}

ELEM_COLORS_VESTA: dict[str, RgbColorType] = {
    elem: (r / 255, g / 255, b / 255)
    for elem, (r, g, b) in ELEM_COLORS_VESTA_256.items()
}


# High-contrast color scheme optimized for metal alloys while preserving some familiar
# colors. Merge with ELEM_COLORS_VESTA to get a complete color scheme while only
# overriding metal colors.
ELEM_COLORS_ALLOY_256: dict[str, Rgb256ColorType] = ELEM_COLORS_VESTA_256 | {
    # Alkali metals - bright purples
    "Li": (0, 53, 0),  # Dark green
    "Na": (0, 41, 255),  # Deep blue
    "K": (0, 255, 0),  # Bright green
    "Rb": (0, 255, 255),  # Cyan
    "Cs": (255, 0, 0),  # Bright red
    # Alkaline earth metals - yellows/oranges
    "Be": (255, 0, 255),  # Magenta
    "Mg": (255, 255, 0),  # Yellow
    "Ca": (255, 255, 255),  # White
    "Sr": (38, 154, 0),  # Green
    "Ba": (0, 150, 255),  # Blue
    # Transition metals - maximizing contrast
    "Sc": (207, 26, 128),  # Pink
    "Ti": (216, 219, 127),  # Light yellow-green
    "V": (255, 150, 0),  # Orange
    "Cr": (197, 163, 255),  # Light purple
    "Mn": (0, 46, 133),  # Dark blue
    "Fe": (0, 151, 134),  # Teal
    "Co": (0, 255, 121),  # Bright green
    "Ni": (99, 0, 62),  # Dark red/burgundy
    "Cu": (129, 0, 255),  # Purple
    "Zn": (168, 74, 0),  # Brown
    "Zr": (108, 96, 208),  # Medium blue-purple
    "Nb": (134, 228, 15),  # Lime green
    # Post-transition metals - earth tones
    "Al": (102, 211, 188),  # Light teal
    "Ga": (255, 121, 143),  # Pink
    "In": (131, 143, 93),  # Olive green
    "Sn": (197, 163, 255),  # Light purple
    "Tl": (0, 46, 133),  # Dark blue
    "Pb": (0, 151, 134),  # Teal
    "Bi": (0, 255, 121),  # Bright green
    # Noble metals - preserving traditional colors
    "Ru": (99, 0, 62),  # Dark red/burgundy
    "Rh": (129, 0, 255),  # Purple
    "Pd": (168, 74, 0),  # Brown
    "Ag": (108, 96, 208),  # Medium blue-purple
    "Os": (134, 228, 15),  # Lime green
    "Ir": (102, 211, 188),  # Light teal
    "Pt": (255, 121, 143),  # Pink
    "Au": (131, 143, 93),  # Olive green
}

ELEM_COLORS_ALLOY: dict[str, RgbColorType] = {
    elem: (r / 255, g / 255, b / 255)
    for elem, (r, g, b) in ELEM_COLORS_ALLOY_256.items()
}

# Pastel color scheme aiming for high contrast between elements.
# Note: This is a foundational set. For a comprehensive set covering all elements
# with optimal contrast, further refinement and extension may be needed.
ELEM_COLORS_PASTEL_256: dict[str, Rgb256ColorType] = {
    "H": (255, 204, 204),  # Light Pinkish Red
    "He": (224, 255, 255),  # Light Cyan
    "Li": (255, 182, 193),  # Light Pink
    "Be": (210, 245, 210),  # Pale Green
    "B": (255, 223, 186),  # Peach
    "C": (200, 200, 200),  # Light Grey
    "N": (173, 216, 230),  # Light Blue
    "O": (255, 179, 179),  # Light Red
    "F": (180, 238, 180),  # Pale Mint Green
    "Ne": (255, 210, 230),  # Pale Pink/Lavender
    "Na": (255, 235, 150),  # Pale Yellow
    "Mg": (190, 250, 190),  # Light Lime Green
    "Al": (211, 211, 211),  # Light Silver/Grey
    "Si": (240, 230, 140),  # Khaki/Pale Gold
    "P": (255, 190, 120),  # Light Orange/Apricot
    "S": (255, 255, 180),  # Pale Yellow
    "Cl": (150, 245, 150),  # Light Sea Green
    "Ar": (230, 230, 250),  # Lavender
    "K": (255, 180, 220),  # Light Orchid
    "Ca": (160, 240, 160),  # Light Pistachio
    "Sc": (220, 220, 180),  # Pale Olive
    "Ti": (190, 210, 220),  # Pale Blue-Grey
    "V": (255, 200, 200),  # Pale Pink-Red
    "Cr": (180, 220, 180),  # Pale Sage Green
    "Mn": (230, 200, 230),  # Pale Lilac
    "Fe": (220, 180, 140),  # Light Tan
    "Co": (200, 200, 255),  # Pale Cornflower Blue
    "Ni": (210, 225, 210),  # Very Pale Green-Grey
    "Cu": (255, 200, 170),  # Light Copper/Salmon
    "Zn": (200, 220, 230),  # Pale Bluish Grey
    "Ga": (255, 210, 210),  # Pale Rosy Pink
    "Ge": (210, 210, 190),  # Pale Beige
    "As": (220, 190, 220),  # Pale Mauve
    "Se": (180, 240, 210),  # Pale Aqua Green
    "Br": (255, 190, 190),  # Light Coral
    "Kr": (240, 220, 255),  # Pale Lavender Blue
    "Rb": (255, 170, 210),  # Light Magenta Pink
    "Sr": (170, 255, 170),  # Light Mint
    "Y": (210, 255, 225),  # Very Pale Aqua
    "Zr": (190, 230, 230),  # Pale Cyan-Grey
    "Nb": (225, 225, 190),  # Pale Yellow-Beige
    "Mo": (200, 230, 200),  # Pale Grey-Green
    "Tc": (230, 210, 230),  # Pale Purple-Grey
    "Ru": (215, 215, 240),  # Pale Periwinkle
    "Rh": (240, 215, 215),  # Pale Pinkish Grey
    "Pd": (220, 220, 220),  # Very Light Grey
    "Ag": (230, 230, 230),  # Very Light Grey (Silver)
    "Cd": (255, 240, 200),  # Pale Cream
    "In": (200, 210, 220),  # Pale Blue Steel
    "Sn": (210, 210, 210),  # Light Grey
    "Sb": (220, 200, 210),  # Pale Pinkish Mauve
    "Te": (190, 240, 190),  # Pale Yellow-Green
    "I": (240, 190, 240),  # Pale Magenta
    "Xe": (210, 240, 255),  # Very Pale Sky Blue
    "Cs": (255, 160, 200),  # Light Deep Pink
    "Ba": (150, 255, 150),  # Light Bright Green
    "La": (190, 255, 210),  # Pale Celadon
    "Au": (255, 240, 180),  # Pale Gold
    "Pt": (220, 220, 230),  # Light Platinum Grey
    # Add more elements as needed, ensuring colors are pastel and contrast well.
}

ELEM_COLORS_PASTEL: dict[str, RgbColorType] = {
    elem: (r / 255, g / 255, b / 255)
    for elem, (r, g, b) in ELEM_COLORS_PASTEL_256.items()
}
