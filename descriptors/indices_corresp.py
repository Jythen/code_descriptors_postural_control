import pandas
import numpy as np

dic_groups = {}

dic_groups["Positional"] = ["mean_distance", "maximal_distance", "rms", "amplitude",
                      "quotient_both_direction", "planar_deviation", "coefficient_sway_direction",
                      "confidence_ellipse_area", "principal_sway_direction"]


dic_groups["Dynamic"] = ["sway_length", "mean_velocity", "LFS", "sway_area_per_second", "phase_plane_parameter",
                        "length_over_area", "fractal_dimension", "zero_crossing", "peak_velocity_pos", 
                        "peak_velocity_neg", "peak_velocity_all", "mean_peak", "mean_distance_peak", "mean_frequency",
                        ]

dic_groups["Frequentist"] =  ["total_power", "power_frequency_50", "power_frequency_95", "frequency_mode",
                      "centroid_frequency", "frequency_dispersion", "energy_content_below_05", "energy_content_05_2", 
                      "energy_content_above_2", "frequency_quotient"]

dic_groups["Stochastic"] = ["short_time_diffusion", "long_time_diffusion",
            "critical_time", "critical_displacement", "critical_displacement", "short_time_scaling",
            "long_time_scaling"]   



def get_corresp(df):
    
    dic_group = {}

    for group in ["Positional","Dynamic","Frequentist","Stochastic"]:
    
        features_group = [f for f in df.columns if len([u for u in dic_groups[group] \
                    if f.replace("_opened_eyes","").replace("_closed_eyes","").replace("_Closed","") \
                    .replace("_Open","").replace("_Foam","").replace("_Firm","").replace("_Radius","") \
                    .replace("_Power_Spectrum_Density","").replace("_Diffusion","") \
                    .replace("_Sway_Density","").replace("_SPD","").replace("_ML_AND_AP","") \
                    .replace("_ML","").replace("_AP","").replace("FEATURE_","") == u])>0]
    
        for feature in features_group:
            dic_group[feature] = group
            
    for feature in df.columns:
        if "GENERATIVE_MODEL" in feature:
            dic_group[feature] = "Generative"

    dic_axis = {}
    
    for feature in df.columns:
        
        if "_ML" in feature and not "_AP" in feature:
            dic_axis[feature] = "ML"
        elif "_AP" in feature and not "_ML" in feature:
            dic_axis[feature] = "AP"
        else:
            dic_axis[feature] = "ML_AND_AP"

   
    for f in df.columns:
        
        if f not in dic_group:
#            print(f, "metadata")
            dic_group[f] = "Morphological characteristics"
            
        if f not in dic_axis:
            dic_axis[f] = "Morphological characteristics"
            
    dic_names = {}
    for f in df.columns:
        dic_names[f] = f.replace("FEATURE_","").replace("GENERATIVE_MODEL_","").replace("opened_eyes","OE") \
                      .replace("closed_eyes","CE").replace("_Power_Spectrum_Density","") \
                      .replace("_Diffusion","").replace("_"," ")
        
            
    return {"dic_names":dic_names, 
            "dic_groups":dic_group, 
            "dic_axis":dic_axis
            }





