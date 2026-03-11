## Core definitions (locked)
Inside PM2.5: I_p(t)  = park monitor time series (hourly, Bangkok time)
Outside PM2.5: O_p(t) = primary district reference station for park’s district

Anomaly: x'_p(t) = x_p(t) - baseline_p(t)  (baseline via STL or regression trend+seasonal)

Wavelet: Morlet CWT on x'_p(t)
Power: P_x,p(s,τ) = |W_x,p(s,τ)|^2

Bands B_k (pre-declare): [fill later: e.g., 12h, 24h, 7d windows]
Regimes g(t) (pre-declare): wet/dry, low/high wind, day/night, season (station primary; reanalysis robustness)

Metric 1 (attenuation): A_p(k,g) = log( Pow_I,p(B_k,g) / Pow_O,p(B_k,g) )
Reporting form: %Red_p(k,g) = 100*(1 - exp(A_p(k,g)))

Metric 2 (coherence): C_p(k,g) = mean of R^2_p(s,τ) over s∈B_k, τ∈g (exclude COI)

- sites_parks_20_bkk_pm25.csv: 20 Bangkok park sensor locations (district + lat/lon), extracted from the installation-location PDF.
