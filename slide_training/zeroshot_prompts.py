"""
Class names and prompt templates for zero-shot classification.

CLASSES[dataset] maps each label, exactly as stored in the UNI2-h feature files, to one or more
class descriptions (synonyms). Every description is combined with every template; the prompts of a
class are encoded, projected, normalised and averaged into one class embedding (prompt ensembling).
Edit freely; the prompts used are saved with each run.
"""

TEMPLATES = [
    "{}.",
    "a photomicrograph showing {}.",
    "a photomicrograph of {}.",
    "an image of {}.",
    "an image showing {}.",
    "an example of {}.",
    "{} is shown.",
    "this is {}.",
    "there is {}.",
    "a histopathological image showing {}.",
    "a histopathological image of {}.",
    "a histopathological photograph of {}.",
    "a histopathological photograph showing {}.",
    "shows {}.",
    "presence of {}.",
    "{} is present.",
    "an H&E stained image of {}.",
    "an H&E stained image showing {}.",
    "an H&E image showing {}.",
    "an H&E image of {}.",
    "{}, H&E stain.",
    "{}, H&E.",
]

CLASSES = {
    # BACH (folder names)
    "bach": {
        "Normal":   ["normal breast tissue", "non-neoplastic breast tissue", "breast tissue without lesion"],
        "Benign":   ["benign breast tissue", "benign breast lesion", "breast fibroadenoma",
                     "fibrocystic change of the breast"],
        "InSitu":   ["breast carcinoma in situ", "ductal carcinoma in situ", "DCIS",
                     "in situ carcinoma of the breast"],
        "Invasive": ["invasive breast carcinoma", "invasive ductal carcinoma", "IDC",
                     "infiltrating breast carcinoma"],
    },
    # CRC-VAL-HE-7K (NCT-CRC-HE tissue classes)
    "crc": {
        "ADI":  ["adipose tissue", "adipocytes", "fat tissue", "fat cells"],
        "BACK": ["background", "slide background", "empty background"],
        "DEB":  ["debris", "necrosis", "necrotic debris"],
        "LYM":  ["lymphocytes", "lymphoid tissue", "lymph node tissue", "immune cells"],
        "MUC":  ["mucus", "mucin", "mucus pool"],
        "MUS":  ["smooth muscle", "smooth muscle tissue", "muscularis propria"],
        "NORM": ["normal colon mucosa", "normal colonic mucosa", "normal colorectal mucosa"],
        "STR":  ["cancer-associated stroma", "tumor stroma", "desmoplastic stroma"],
        "TUM":  ["colorectal adenocarcinoma epithelium", "colorectal adenocarcinoma",
                 "colon adenocarcinoma", "tumor epithelium"],
    },
    # MHIST (majority-vote labels)
    "mhist": {
        "HP":  ["hyperplastic polyp", "colorectal hyperplastic polyp"],
        "SSA": ["sessile serrated adenoma", "sessile serrated polyp", "sessile serrated lesion"],
    },
    # BreakHis, binary
    "breakhis": {
        "benign":    ["benign breast tumor", "benign breast tissue", "benign breast lesion"],
        "malignant": ["malignant breast tumor", "breast carcinoma", "breast cancer",
                      "malignant breast tissue"],
    },
    # BreakHis, 8 subtypes (labels = 'subtype' column of breakhis/metadata.csv)
    "breakhis_subtype": {
        "adenosis":            ["breast adenosis", "adenosis of the breast", "sclerosing adenosis"],
        "fibroadenoma":        ["breast fibroadenoma", "fibroadenoma of the breast"],
        "phyllodes_tumor":     ["phyllodes tumor of the breast", "phyllodes tumor",
                                "breast phyllodes tumor"],
        "tubular_adenoma":     ["tubular adenoma of the breast", "breast tubular adenoma"],
        "ductal_carcinoma":    ["invasive ductal carcinoma of the breast", "breast ductal carcinoma",
                                "ductal carcinoma"],
        "lobular_carcinoma":   ["invasive lobular carcinoma of the breast", "breast lobular carcinoma",
                                "lobular carcinoma"],
        "mucinous_carcinoma":  ["mucinous carcinoma of the breast", "breast mucinous carcinoma",
                                "colloid carcinoma of the breast"],
        "papillary_carcinoma": ["papillary carcinoma of the breast", "breast papillary carcinoma"],
    },
}


def prompts_for(dataset):
    """{label: [prompt, ...]} for one dataset."""
    return {label: [t.format(d) for d in descs for t in TEMPLATES]
            for label, descs in CLASSES[dataset].items()}
