"""Standalone region labelling from OLD/utils.py, without legacy analysis imports."""
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import load_img

def get_region_label(pos_m) : 
    """Legacy AAL-to-project mapping; input DataFrame coordinates are in metres."""
    # Load atlas
    atlas = datasets.fetch_atlas_aal()
    atlas_img = load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    inv_affine = np.linalg.inv(affine)
    ijk = nib.affines.apply_affine(inv_affine, pos_m.values* 1000)
    ijk = np.round(ijk).astype(int)
    shape = atlas_data.shape
    inside = ((ijk[:,0] >= 0) & (ijk[:,0] < shape[0]) & (ijk[:,1] >= 0) & (ijk[:,1] < shape[1]) & (ijk[:,2] >= 0) & (ijk[:,2] < shape[2]))

    label_values = np.zeros(len(ijk), dtype=int)
    label_values[inside] = atlas_data[
        ijk[inside,0],
        ijk[inside,1],
        ijk[inside,2]
    ].astype(int)
    mapping = dict(zip(map(int, atlas.indices), atlas.labels))
    regions = [mapping.get(v, "Background") for v in label_values]

    aal_to_region = {
        # Background
        "Background": 'N',

        # Frontal
        "Frontal_Sup_L": "DLPFC",
        "Frontal_Sup_R": "DLPFC",
        "Frontal_Sup_Orb_L": "OFC",
        "Frontal_Sup_Orb_R": "OFC",
        "Frontal_Mid_L": "DLPFC",
        "Frontal_Mid_R": "DLPFC",
        "Frontal_Mid_Orb_L": "OFC",
        "Frontal_Mid_Orb_R": "OFC",
        "Frontal_Inf_Oper_L": "VLPFC",
        "Frontal_Inf_Oper_R": "VLPFC",
        "Frontal_Inf_Tri_L": "VLPFC",
        "Frontal_Inf_Tri_R": "VLPFC",
        "Frontal_Inf_Orb_L": "OFC",
        "Frontal_Inf_Orb_R": "OFC",
        "Rolandic_Oper_L": "VLPFC",   # closest to frontal operculum
        "Rolandic_Oper_R": "VLPFC",
        "Supp_Motor_Area_L": "premotor",
        "Supp_Motor_Area_R": "premotor",
        "Olfactory_L": "OFC",
        "Olfactory_R": "OFC",
        "Rectus_L": "OFC",
        "Rectus_R": "OFC",
        "Insula_L": "INS",
        "Insula_R": "INS",

        # Cingulate
        "Cingulum_Ant_L": "ACC",
        "Cingulum_Ant_R": "ACC",
        "Cingulum_Mid_L": "ACC",
        "Cingulum_Mid_R": "ACC",
        "Cingulum_Post_L": "PCC",
        "Cingulum_Post_R": "PCC",

        # Motor / sensory
        "Precentral_L": "M1",
        "Precentral_R": "M1",
        "Postcentral_L": "S1",
        "Postcentral_R": "S1",
        "Paracentral_Lobule_L": "M1",
        "Paracentral_Lobule_R": "M1",

        # Parietal
        "Parietal_Sup_L": "parietal",
        "Parietal_Sup_R": "parietal",
        "Parietal_Inf_L": "parietal",
        "Parietal_Inf_R": "parietal",
        "SupraMarginal_L": "parietal",
        "SupraMarginal_R": "parietal",
        "Angular_L": "parietal",
        "Angular_R": "parietal",
        "Precuneus_L": "PCC",
        "Precuneus_R": "PCC",

        # Temporal
        "Heschl_L": "A1",
        "Heschl_R": "A1",
        "Temporal_Sup_L": "STG",
        "Temporal_Sup_R": "STG",
        "Temporal_Mid_L": "MTG",
        "Temporal_Mid_R": "MTG",
        "Temporal_Inf_L": "VS",      # ventral temporal
        "Temporal_Inf_R": "VS",
        "Temporal_Pole_Sup_L": "TP",
        "Temporal_Pole_Sup_R": "TP",
        "Temporal_Pole_Mid_L": "TP",
        "Temporal_Pole_Mid_R": "TP",
        "Fusiform_L": "VS",
        "Fusiform_R": "VS",

        # Medial temporal
        "Hippocampus_L": "HPC",
        "Hippocampus_R": "HPC",
        "ParaHippocampal_L": "PHC",
        "ParaHippocampal_R": "PHC",
        "Amygdala_L": "AMY",
        "Amygdala_R": "AMY",

        # Occipital (no equivalent in REGION)
        "Calcarine_L": 'Occ',
        "Calcarine_R": 'Occ',
        "Cuneus_L": 'Occ',
        "Cuneus_R": 'Occ',
        "Lingual_L": 'Occ',
        "Lingual_R": 'Occ',
        "Occipital_Sup_L": 'Occ',
        "Occipital_Sup_R": 'Occ',
        "Occipital_Mid_L": 'Occ',
        "Occipital_Mid_R": 'Occ',
        "Occipital_Inf_L": 'Occ',
        "Occipital_Inf_R": 'Occ',

        # Deep nuclei
        "Caudate_L": 'Caud',
        "Caudate_R": 'Caud',
        "Putamen_L": 'Put',
        "Putamen_R": 'Put',
        "Pallidum_L": 'Pal',
        "Pallidum_R": 'Pal',
        "Thalamus_L": "THAL",
        "Thalamus_R": "THAL",

        # Cerebellum / vermis (not represented)
        "Cerebelum_Crus1_L": 'CERB',
        "Cerebelum_Crus1_R": 'CERB',
        "Cerebelum_Crus2_L": 'CERB',
        "Cerebelum_Crus2_R": 'CERB',
        "Cerebelum_3_L": 'CERB',
        "Cerebelum_3_R": 'CERB',
        "Cerebelum_4_5_L": 'CERB',
        "Cerebelum_4_5_R": 'CERB',
        "Cerebelum_6_L": 'CERB',
        "Cerebelum_6_R": 'CERB',
        "Cerebelum_7b_L": 'CERB',
        "Cerebelum_7b_R": 'CERB',
        "Cerebelum_8_L": 'CERB',
        "Cerebelum_8_R": 'CERB',
        "Cerebelum_9_L": 'CERB',
        "Cerebelum_9_R": 'CERB',
        "Cerebelum_10_L": 'CERB',
        "Cerebelum_10_R": 'CERB',
        "Vermis_1_2": 'CERB',
        "Vermis_3": 'CERB',
        "Vermis_4_5": 'CERB',
        "Vermis_6": 'CERB',
        "Vermis_7": 'CERB',
        "Vermis_8": 'CERB',
        "Vermis_9": 'CERB',

        "Frontal_Sup_Medial_L": "DLPFC",
        "Frontal_Sup_Medial_R": "DLPFC",

        "Frontal_Med_Orb_L": "OFC",
        "Frontal_Med_Orb_R": "OFC",
    }

    regions_meg = [aal_to_region.get(v) for v in regions]

    for i, r in enumerate(regions_meg) :
        if r == None : 
            print(regions[i])
    
    return regions_meg
