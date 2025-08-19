import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

import numpy as np
from scipy.signal import find_peaks, peak_widths

def filter_largest_peak(intensities, prominence=10, drop_frac=0.05):
    """
    Keep only the largest peak from the raw intensity array,
    preserving the natural taper of the peak.
    
    Parameters
    ----------
    intensities : array-like
        The raw data (1D).
    prominence : float
        Minimum prominence to detect peaks.
    drop_frac : float
        Fraction of peak height to determine taper cutoff.
    """
    intensities = np.array(intensities, dtype=float)
    intensities = subtract_background(intensities, fraction=0.08)

    # Find all peaks
    peaks, _ = find_peaks(intensities, prominence=prominence)
    if len(peaks) == 0:
        return np.zeros_like(intensities)

    # Get the largest peak index
    largest_peak_idx = peaks[np.argmax(intensities[peaks])]
    peak_height = intensities[largest_peak_idx]

    # Get initial half-prominence bounds
    results_half = peak_widths(intensities, [largest_peak_idx], rel_height=0.5)
    left_idx = int(np.floor(results_half[2][0]))
    right_idx = int(np.ceil(results_half[3][0]))

    # Expand left until drop below fraction of peak height
    while left_idx > 0 and intensities[left_idx] > peak_height * drop_frac:
        left_idx -= 1

    # Expand right until drop below fraction of peak height
    while right_idx < len(intensities) - 1 and intensities[right_idx] > peak_height * drop_frac:
        right_idx += 1

    # Keep the raw values in the expanded region
    filtered = np.zeros_like(intensities)
    filtered[left_idx:right_idx+1] = intensities[left_idx:right_idx+1]

    return filtered

def subtract_background(intensities, fraction=0.08):
    """
    Subtract average background noise from the intensity array.
    Background estimated from the first `fraction` of the bins.
    """
    intensities = np.array(intensities, dtype=float)
    n_bins = len(intensities)
    n_bg = max(1, int(n_bins * fraction))  # at least 1 bin
    
    background_level = np.mean(intensities[:n_bg])
    corrected = intensities - background_level
    corrected[corrected < 0] = 0
    return corrected

def peak_normalize(curve):
    """Normalize curve by its peak value."""
    EPS = 1e-12
    return curve / max(curve.max(), EPS)

def area_normalize(model_curve, meas_curve, xleft, xright):
    """
    Normalize model curve so its area matches the measurement curve in [xleft:xright].
    """
    EPS = 1e-12
    model_area = model_curve[xleft:xright].sum()
    meas_area = meas_curve[xleft:xright].sum()
    if model_area < EPS:
        return model_curve  # avoid divide by zero
    return (model_curve / model_area) * meas_area

if __name__ == "__main__":
    # Data
    data_strings = [
        "78;78;73;73;76;92;70;81;80;72;79;66;75;88;111;80;89;6042;14053;3146;725;320;194;123;95;90;69;71;85;75;79;70;75;89;71;78;62;70;76;86;58;83;81;85;63;76;71;83;99;175;158;103;73;81;72;70;85;77;60;80;50;73;79;81;80;74;75;85;63;67;75;75;69;60;58;68;61;71;75;67;74;62;64;53;56;79;67;75;66;64;66;73;60;68;78;59;66;80;64;69;61;74;85;63;67;72;65;65;61;59;65;63;72;71;63;59;65;55;75;57;53;76;73;72;79;75;81;58",
        "61;57;57;48;58;51;61;53;43;49;53;46;53;58;59;64;50;4959;18288;5289;1078;395;202;133;69;53;56;44;59;60;55;58;44;47;40;48;52;43;51;43;41;41;44;44;37;45;47;47;77;82;62;54;60;37;38;34;57;51;49;48;55;33;40;50;39;48;46;37;52;38;42;53;42;38;51;40;61;35;36;43;46;59;29;59;38;57;41;47;49;59;60;42;49;52;55;45;48;32;51;43;37;42;39;53;56;47;40;56;48;38;46;58;36;37;51;40;43;49;42;36;42;41;36;38;49;43;51;42",
        "80;78;76;84;61;80;71;80;78;71;85;80;97;99;120;95;103;810;1817;472;179;150;111;97;88;94;84;83;82;83;89;91;98;83;98;79;96;68;62;84;79;88;71;89;80;93;86;117;292;320;180;101;95;90;90;95;73;87;68;78;75;93;93;85;71;88;68;91;64;80;76;74;96;76;74;89;92;81;66;89;100;95;81;93;79;70;92;79;73;71;82;82;92;64;67;78;67;76;65;58;72;76;77;77;82;89;81;80;74;92;84;90;82;76;78;82;77;69;74;74;89;87;74;78;62;68;87;63",
        "65;60;66;56;53;57;54;64;60;52;67;57;60;58;78;52;63;6379;30609;9637;1783;594;279;177;93;61;71;67;56;55;66;49;53;57;53;55;40;47;56;51;56;59;55;40;39;63;36;64;47;56;99;85;65;55;53;38;50;55;54;58;49;48;53;49;53;50;56;53;67;40;59;70;58;47;53;57;54;61;49;53;62;49;64;55;55;49;40;44;49;52;48;47;39;53;50;41;47;52;44;52;46;39;48;55;54;45;50;45;44;68;53;46;41;46;49;54;52;47;47;50;47;60;42;57;60;50;43;49",
        "76;61;75;66;66;56;61;68;76;78;61;56;67;70;57;75;68;6135;36364;12469;2109;798;385;176;97;70;58;56;53;54;44;49;58;57;48;40;42;55;47;50;50;43;46;46;50;44;54;49;51;73;77;64;59;52;61;50;42;56;49;50;53;44;53;47;50;67;56;56;52;49;42;56;49;43;46;48;48;54;57;42;51;48;42;39;44;45;56;51;39;54;38;45;48;46;67;49;42;49;42;47;45;55;56;48;49;44;58;51;38;57;37;37;60;51;51;40;52;43;54;51;48;50;45;53;59;53;41;35",
        "85;82;59;68;81;98;95;66;72;81;77;89;65;81;98;110;96;728;7499;4476;778;289;195;157;114;98;97;79;84;78;69;70;69;77;78;77;89;88;74;83;69;75;69;85;90;86;85;74;103;251;292;157;92;82;94;74;81;64;83;78;97;78;67;75;81;69;76;86;81;101;75;72;57;92;64;82;101;76;74;99;67;82;65;71;71;91;83;72;92;81;70;63;82;79;77;96;73;71;60;88;96;69;70;74;81;76;70;73;89;81;79;83;75;73;85;73;68;90;75;51;57;70;50;78;74;63;72;70",
        "69;63;60;55;66;60;69;70;66;61;70;75;62;58;80;83;94;3660;15889;5079;956;387;193;144;95;83;57;68;76;57;57;60;77;65;60;71;67;55;52;60;63;75;45;56;61;57;66;58;67;69;85;100;60;57;62;58;77;81;70;64;65;72;61;72;67;46;64;64;67;52;50;55;61;54;63;52;52;70;66;54;62;48;60;57;59;61;54;57;34;48;53;72;52;65;56;58;44;56;52;80;50;70;56;50;53;56;73;54;72;67;75;55;55;47;61;61;46;58;54;71;62;53;57;67;52;68;62;62",
        "175;176;172;187;198;180;189;172;157;167;125;158;170;164;195;167;195;2559;12499;4188;928;419;308;227;171;183;190;161;152;161;161;152;147;173;182;144;169;158;150;155;156;158;145;137;172;182;160;143;170;194;204;198;190;166;141;145;152;150;156;155;146;159;158;166;145;148;160;131;145;161;145;135;163;137;138;153;170;162;145;141;140;149;156;136;170;123;144;155;146;142;136;136;156;161;151;142;158;140;134;150;183;139;144;141;156;172;153;127;152;173;148;160;144;144;158;149;131;139;135;148;133;176;157;141;138;137;147;148",
        "154;146;169;157;169;155;133;176;165;174;162;158;153;147;185;181;172;424;2021;932;344;181;203;170;167;157;151;139;179;153;141;141;137;167;156;149;179;161;159;171;171;151;159;127;163;139;161;142;172;259;367;309;215;159;163;166;133;152;143;148;156;154;160;152;176;165;151;149;173;174;131;153;157;152;174;158;175;138;166;147;136;151;142;154;160;167;162;165;166;159;145;146;168;164;154;159;152;158;125;165;154;146;157;176;127;124;175;173;149;160;146;176;148;157;135;136;148;154;145;128;160;152;137;139;136;135;157;158"
    ]

    data_lists = [[int(x) for x in s.split(';')] for s in data_strings]

    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        raw_data = np.array(data_lists[i])
        filtered_data = filter_largest_peak(raw_data)
        
        ax.plot(raw_data, label="Raw Data", alpha=0.5)
        ax.plot(filtered_data, label="Filtered Largest Peak", linewidth=2)
        ax.set_title(f"HLONG{i+1}")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig("filtered_peaks.png")
