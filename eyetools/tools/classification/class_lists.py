from __future__ import annotations

TASK_CLASS_MAP = {
    "modality": [
        "Color fundus photography",
        "OCT",
        "Fluorescein angiography",
        "Fundus autofluorescence",
        "MRI",
        "CT",
    ],
    "cfp_quality": [
        "Gradable",
        "Poor Image Quality",
        "Out of focus",
    ],
    "laterality": ["OD", "OS"],
    "multidis": [
        "age-related macular degeneration",
        "diabetic retinopathy",
        "retinal detachment",
        "glaucoma",
        "macular hole",
        "normal",
        "pathologic myopia",
        "retinitis pigmentosa",
        "epiretinal membrane",
        "central serous chorioretinopathy",
    ],
    "cfp_age": ["age"],
}

MULTIDIS_SIGNS = {
    "sign": [
        "drusen",
        "exudate",
        "hemorrhage",
        "macular atrophy",
        "macular edema",
        "subretinal hemorrhage",
    ]
}

DEFAULT_SHOW_N = {"multidis": 5, "cfp_quality": 3, "modality": 3, "laterality": 1, "cfp_age": 1}

__all__ = ["TASK_CLASS_MAP", "MULTIDIS_SIGNS", "DEFAULT_SHOW_N"]