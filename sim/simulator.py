import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_CHANNELS = 8            
                          
N_PATIENTS = 12          
N_TRIALS = 300            
CALIBRATION_TRIALS = 60   
KL_WINDOW = 50            
COV_SHRINKAGE = 0.15      
                          
                          
TARGET_DIRECTIONS_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315])  

RNG_SEED = 42


@dataclass
class PatientConfig:
    patient_id: str
    preferred_directions_deg: np.ndarray   
    baseline_rate: np.ndarray              
    amplitude: np.ndarray                 
    noise_std: float                       
    drift_severity_deg: float              


def make_patient(patient_id: str, drift_severity_deg: float, rng: np.random.Generator) -> PatientConfig:
   
    preferred = (TARGET_DIRECTIONS_DEG.astype(float) + rng.uniform(-10, 10, size=N_CHANNELS)) % 360
    baseline = rng.uniform(8.0, 12.0, size=N_CHANNELS)  
    amplitude = rng.uniform(3.0, 6.0, size=N_CHANNELS)
    return PatientConfig(
        patient_id=patient_id,
        preferred_directions_deg=preferred,
        baseline_rate=baseline,
        amplitude=amplitude,
        noise_std=0.6,
        drift_severity_deg=drift_severity_deg,
    )


def cosine_tuned_features(true_angle_deg, patient: PatientConfig, per_channel_drift_deg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    
    current_preferred = patient.preferred_directions_deg + per_channel_drift_deg
    angle_diff_rad = np.deg2rad(true_angle_deg - current_preferred)
    signal = patient.baseline_rate + patient.amplitude * np.cos(angle_diff_rad)
    noise = rng.normal(0, patient.noise_std, size=N_CHANNELS)
    return signal + noise


def calibrate_decoder(patient: PatientConfig):
    """A decoder calibrated ONCE, at trial 0, to each channel's un-drifted preferred direction."""
    return patient.preferred_directions_deg.copy()


def population_vector_decode(features: np.ndarray, baseline_rate: np.ndarray, calibrated_preferred_directions_deg: np.ndarray) -> float:
    
    weights = features - baseline_rate
    rad = np.deg2rad(calibrated_preferred_directions_deg)
    x = np.sum(weights * np.cos(rad))
    y = np.sum(weights * np.sin(rad))
    return np.rad2deg(np.arctan2(y, x)) % 360


def angle_error_deg(true_deg, decoded_deg) -> float:
    """Shortest-path angular difference, always returned as a positive degree value."""
    diff = (decoded_deg - true_deg + 180) % 360 - 180
    return abs(diff)


def shrink_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Blends a sample covariance matrix toward a diagonal one, for numerical stability."""
    diag_avg = np.mean(np.diag(cov))
    k = cov.shape[0]
    return (1 - shrinkage) * cov + shrinkage * diag_avg * np.eye(k)


def multivariate_gaussian_kl(mean_p, cov_p, mean_q, cov_q, shrinkage=COV_SHRINKAGE) -> float:
    
    k = mean_p.shape[0]
    cov_p = shrink_covariance(cov_p, shrinkage)
    cov_q = shrink_covariance(cov_q, shrinkage)

    cov_q_inv = np.linalg.inv(cov_q)
    diff = mean_q - mean_p
    term_trace = np.trace(cov_q_inv @ cov_p)
    term_quad = diff @ cov_q_inv @ diff
    _, logdet_p = np.linalg.slogdet(cov_p)
    _, logdet_q = np.linalg.slogdet(cov_q)

    return float(0.5 * (term_trace + term_quad - k + (logdet_q - logdet_p)))


def simulate_patient_session(patient: PatientConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Runs one full synthetic session for one patient: N_TRIALS trials, one row of telemetry per trial."""
    calibrated_pd = calibrate_decoder(patient)

    channel_drift_steps = rng.normal(0, patient.drift_severity_deg, size=(N_TRIALS, N_CHANNELS))
    cumulative_drift = np.cumsum(channel_drift_steps, axis=0)

    all_features = []
    rows = []
    baseline_mean = None
    baseline_cov = None

    for trial in range(N_TRIALS):
        true_angle = float(rng.choice(TARGET_DIRECTIONS_DEG))
        drift_vec = cumulative_drift[trial]
        features = cosine_tuned_features(true_angle, patient, drift_vec, rng)
        all_features.append(features)

        decoded_angle = population_vector_decode(features, patient.baseline_rate, calibrated_pd)
        err = angle_error_deg(true_angle, decoded_angle)

        if trial == CALIBRATION_TRIALS - 1:
            calib_window = np.array(all_features[:CALIBRATION_TRIALS])
            baseline_mean = calib_window.mean(axis=0)
            baseline_cov = np.cov(calib_window, rowvar=False)

        kl = np.nan
        if trial >= CALIBRATION_TRIALS + KL_WINDOW - 1 and baseline_mean is not None:
            current_window = np.array(all_features[trial - KL_WINDOW + 1: trial + 1])
            current_mean = current_window.mean(axis=0)
            current_cov = np.cov(current_window, rowvar=False)
            kl = multivariate_gaussian_kl(current_mean, current_cov, baseline_mean, baseline_cov)

        rows.append({
            "patient_id": patient.patient_id,
            "trial": trial,
            "true_angle_deg": true_angle,
            "decoded_angle_deg": decoded_angle,
            "angle_error_deg": err,
            "kl_divergence": kl,
            "mean_abs_channel_drift_deg": float(np.mean(np.abs(drift_vec))),
        })

    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(RNG_SEED)

    
    drift_severities = (
        [0.0, 0.0, 0.05, 0.05] +                          
        [0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.3]          
    )
    assert len(drift_severities) == N_PATIENTS

    all_sessions = []
    for i, severity in enumerate(drift_severities):
        patient_id = f"patient_{i+1:02d}"
        patient = make_patient(patient_id, severity, rng)
        session_df = simulate_patient_session(patient, rng)
        all_sessions.append(session_df)

    full_df = pd.concat(all_sessions, ignore_index=True)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "telemetry.csv"
    full_df.to_csv(out_path, index=False)

    print(f"Simulated {N_PATIENTS} patients x {N_TRIALS} trials. Saved to {out_path}\n")
    print(f"{'patient_id':<12} {'severity':<9} {'start_err':<10} {'end_err':<9} {'end_kl':<8}")
    for i, severity in enumerate(drift_severities):
        pid = f"patient_{i+1:02d}"
        pdf = full_df[full_df.patient_id == pid]
        start_err = pdf.angle_error_deg.iloc[:CALIBRATION_TRIALS].mean()
        end_err = pdf.angle_error_deg.iloc[-10:].mean()
        end_kl = pdf.kl_divergence.iloc[-1]
        print(f"{pid:<12} {severity:<9.2f} {start_err:<10.1f} {end_err:<9.1f} {end_kl:<8.2f}")


if __name__ == "__main__":
    main()