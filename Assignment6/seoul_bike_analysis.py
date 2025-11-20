"""Exploratory data analysis and modeling for the Seoul Bike Sharing dataset.

This script avoids external dependencies because the execution environment blocks
package downloads. If the original dataset is unavailable, it synthesizes a
realistic sample with similar feature ranges to demonstrate the workflow.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DATA_PATH = Path("Assignment6/data/SeoulBikeData.csv")
FIGURES_DIR = Path("Assignment6/figures")


def synthetic_row(day: int, hour: int) -> Dict[str, float | str]:
    """Generate a synthetic observation inspired by the Seoul Bike dataset."""
    # Determine season from day of year approximation
    if day < 80 or day >= 355:
        season = "Winter"
    elif day < 172:
        season = "Spring"
    elif day < 264:
        season = "Summer"
    else:
        season = "Autumn"

    # Temperature follows a sinusoidal seasonal cycle with diurnal variation
    base_temp = {
        "Winter": 0,
        "Spring": 10,
        "Summer": 25,
        "Autumn": 12,
    }[season]
    temp = base_temp + 7 * math.sin(2 * math.pi * hour / 24) + random.gauss(0, 2)

    humidity = max(10, min(95, 40 + (50 - temp) * 0.6 + random.gauss(0, 8)))
    wind = max(0, random.gauss(2.5, 1))
    visibility = max(50, random.gauss(1600 - humidity * 10, 150))
    solar = max(0.0, (12 - abs(hour - 14)) * 0.35 + random.gauss(0, 0.3))

    rainfall = 0.0
    snowfall = 0.0
    if season in {"Summer", "Spring"} and random.random() < 0.12:
        rainfall = max(0.0, random.expovariate(0.9))
    if season == "Winter" and random.random() < 0.07:
        snowfall = max(0.0, random.expovariate(1.3))

    holiday = "Holiday" if random.random() < 0.03 else "No Holiday"
    functioning_day = "Yes" if random.random() > 0.02 else "No"

    demand = max(
        0,
        50
        + hour * 8
        + temp * 20
        - humidity * 3
        - rainfall * 40
        - snowfall * 30
        + (200 if functioning_day == "Yes" else -300)
        + (150 if holiday == "No Holiday" else -100)
        + random.gauss(0, 120),
    )

    return {
        "Hour": hour,
        "Temperature_C": round(temp, 2),
        "Humidity_percent": round(humidity, 2),
        "Wind_speed_m_s": round(wind, 2),
        "Visibility_10m": round(visibility / 10, 1),
        "Solar_Radiation_MJ_m2": round(solar, 2),
        "Rainfall_mm": round(rainfall, 2),
        "Snowfall_cm": round(snowfall, 2),
        "Season": season,
        "Holiday": holiday,
        "Functioning_Day": functioning_day,
        "Rented_Bike_Count": round(demand, 0),
    }


def ensure_dataset(rows: int = 800) -> List[Dict[str, float | str]]:
    """Load dataset from disk or synthesize one if not available."""
    if DATA_PATH.exists():
        with DATA_PATH.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return [
                {k: (float(v) if k != "Season" and k != "Holiday" and k != "Functioning_Day" else v) for k, v in row.items()}
                for row in reader
            ]

    random.seed(42)
    data: List[Dict[str, float | str]] = []
    for day in range(rows // 24):
        for hour in range(24):
            data.append(synthetic_row(day, hour))
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Hour",
                "Temperature_C",
                "Humidity_percent",
                "Wind_speed_m_s",
                "Visibility_10m",
                "Solar_Radiation_MJ_m2",
                "Rainfall_mm",
                "Snowfall_cm",
                "Season",
                "Holiday",
                "Functioning_Day",
                "Rented_Bike_Count",
            ],
        )
        writer.writeheader()
        writer.writerows(data)
    return data


def column(values: List[Dict[str, float | str]], key: str) -> List[float]:
    return [float(row[key]) for row in values]


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return num / den if den else 0.0


def save_svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{body}</svg>'
    path.write_text(svg)


def histogram_svg(values: Sequence[float], title: str, filename: str, bins: int = 20) -> None:
    vmin, vmax = min(values), max(values)
    bin_size = (vmax - vmin) / bins or 1
    counts = [0 for _ in range(bins)]
    for v in values:
        idx = min(bins - 1, int((v - vmin) / bin_size))
        counts[idx] += 1

    max_count = max(counts)
    width, height = 600, 350
    bar_width = width / bins
    bars = []
    for i, c in enumerate(counts):
        bar_height = (c / max_count) * (height - 80)
        x = i * bar_width
        y = height - bar_height - 40
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" fill="#4682B4" />')
    text = f'<text x="10" y="20" font-size="16">{title}</text>'
    save_svg(FIGURES_DIR / filename, width, height, text + "".join(bars))


def boxplot_svg(values: Sequence[float], title: str, filename: str) -> None:
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    lower = max(min(values), q1 - 1.5 * iqr)
    upper = min(max(values), q3 + 1.5 * iqr)

    width, height = 260, 160
    scale = (height - 60) / (max(values) - min(values) or 1)

    def y(v: float) -> float:
        return height - 30 - (v - min(values)) * scale

    elements = [f'<text x="10" y="20" font-size="14">{title}</text>']
    elements.append(f'<line x1="120" y1="{y(lower):.1f}" x2="120" y2="{y(upper):.1f}" stroke="#333" />')
    elements.append(f'<rect x="80" y="{y(q3):.1f}" width="80" height="{(q3 - q1) * scale:.1f}" fill="#f4a460" stroke="#333" />')
    elements.append(f'<line x1="80" x2="160" y1="{y(median):.1f}" y2="{y(median):.1f}" stroke="#333" stroke-width="2" />')
    save_svg(FIGURES_DIR / filename, width, height, "".join(elements))


def scatter_svg(x: Sequence[float], y_vals: Sequence[float], title: str, xlabel: str, ylabel: str, filename: str) -> None:
    width, height = 520, 360
    margin = 50
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y_vals), max(y_vals)
    x_scale = (width - 2 * margin) / (x_max - x_min or 1)
    y_scale = (height - 2 * margin) / (y_max - y_min or 1)

    def px(v: float) -> float:
        return margin + (v - x_min) * x_scale

    def py(v: float) -> float:
        return height - margin - (v - y_min) * y_scale

    points = [f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2" fill="rgba(70,130,180,0.6)" />' for a, b in zip(x, y_vals)]

    # Simple linear regression for trend line
    slope = pearson(x, y_vals) * (statistics.pstdev(y_vals) / (statistics.pstdev(x) or 1))
    intercept = statistics.fmean(y_vals) - slope * statistics.fmean(x)
    line_pts = [
        (x_min, slope * x_min + intercept),
        (x_max, slope * x_max + intercept),
    ]
    line = f'<line x1="{px(line_pts[0][0]):.1f}" y1="{py(line_pts[0][1]):.1f}" x2="{px(line_pts[1][0]):.1f}" y2="{py(line_pts[1][1]):.1f}" stroke="#b22222" stroke-width="2" />'

    labels = (
        f'<text x="10" y="20" font-size="16">{title}</text>'
        f'<text x="{width/2:.1f}" y="{height - 10}" font-size="12" text-anchor="middle">{xlabel}</text>'
        f'<text x="10" y="{height/2:.1f}" font-size="12" transform="rotate(-90 10,{height/2:.1f})">{ylabel}</text>'
    )

    save_svg(FIGURES_DIR / filename, width, height, labels + "".join(points) + line)


def heatmap_svg(labels: Sequence[str], matrix: List[List[float]], title: str, filename: str) -> None:
    size = 24
    width = height = size * len(labels) + 120
    cells = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            intensity = (value + 1) / 2  # map [-1,1] to [0,1]
            red = int(255 * intensity)
            blue = int(255 * (1 - intensity))
            color = f"rgb({red},{0},{blue})"
            x = 80 + j * size
            y = 60 + i * size
            cells.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{color}" />')
    label_elements = []
    for idx, label in enumerate(labels):
        label_elements.append(f'<text x="{80 + idx * size + size/2}" y="50" font-size="10" text-anchor="middle" transform="rotate(-60 {80 + idx * size + size/2},50)">{label}</text>')
        label_elements.append(f'<text x="40" y="{60 + idx * size + size/2}" font-size="10" text-anchor="end">{label}</text>')
    title_text = f'<text x="10" y="20" font-size="16">{title}</text>'
    save_svg(FIGURES_DIR / filename, width, height, title_text + "".join(cells + label_elements))


def encode_features(data: List[Dict[str, float | str]]) -> Tuple[List[List[float]], List[float], List[str]]:
    seasons = sorted({row["Season"] for row in data})
    holidays = sorted({row["Holiday"] for row in data})
    functioning = sorted({row["Functioning_Day"] for row in data})

    feature_names = [
        "Hour",
        "Temperature_C",
        "Humidity_percent",
        "Wind_speed_m_s",
        "Visibility_10m",
        "Solar_Radiation_MJ_m2",
        "Rainfall_mm",
        "Snowfall_cm",
    ] + [f"Season_{s}" for s in seasons] + [f"Holiday_{h}" for h in holidays] + [
        f"Functioning_{f}" for f in functioning
    ]

    X: List[List[float]] = []
    y: List[float] = []
    for row in data:
        features: List[float] = [
            float(row["Hour"]),
            float(row["Temperature_C"]),
            float(row["Humidity_percent"]),
            float(row["Wind_speed_m_s"]),
            float(row["Visibility_10m"]),
            float(row["Solar_Radiation_MJ_m2"]),
            float(row["Rainfall_mm"]),
            float(row["Snowfall_cm"]),
        ]
        features.extend([1.0 if row["Season"] == s else 0.0 for s in seasons])
        features.extend([1.0 if row["Holiday"] == h else 0.0 for h in holidays])
        features.extend([1.0 if row["Functioning_Day"] == f else 0.0 for f in functioning])
        X.append(features)
        y.append(float(row["Rented_Bike_Count"]))
    return X, y, feature_names


def train_test_split(X: List[List[float]], y: List[float], test_ratio: float = 0.2):
    combined = list(zip(X, y))
    random.shuffle(combined)
    split = int(len(combined) * (1 - test_ratio))
    train, test = combined[:split], combined[split:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(X_test), list(y_train), list(y_test)


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B)):
            for j in range(len(B[0])):
                result[i][j] += A[i][k] * B[k][j]
    return result


def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*A)]


def gaussian_solve(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    # Augment matrix
    M = [row + [b_i] for row, b_i in zip([row[:] for row in A], b)]
    for col in range(n):
        # Pivot
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        pivot_val = M[col][col] or 1e-12
        # Normalize
        for j in range(col, n + 1):
            M[col][j] /= pivot_val
        # Eliminate
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def linear_regression(X: List[List[float]], y: List[float]) -> List[float]:
    Xt = matrix_transpose(X)
    XtX = matrix_multiply(Xt, X)
    Xty = matrix_multiply(Xt, [[v] for v in y])
    coeffs = gaussian_solve(XtX, [row[0] for row in Xty])
    return coeffs


def predict_linear(X: List[List[float]], coeffs: List[float]) -> List[float]:
    return [sum(a * b for a, b in zip(row, coeffs)) for row in X]


def knn_regression(train_X, train_y, test_X, k: int = 5) -> List[float]:
    predictions = []
    for row in test_X:
        dists = []
        for t_row, t_y in zip(train_X, train_y):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, t_row)))
            dists.append((dist, t_y))
        dists.sort(key=lambda x: x[0])
        top_k = [val for _, val in dists[:k]]
        predictions.append(statistics.fmean(top_k))
    return predictions


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return statistics.fmean((a - b) ** 2 for a, b in zip(y_true, y_pred))


def main() -> None:
    data = ensure_dataset()

    # EDA plots
    target = column(data, "Rented_Bike_Count")
    histogram_svg(target, "Rented Bike Count", "seoul_bike_rentals_hist.svg")
    boxplot_svg(target, "Rented Bike Count", "seoul_bike_rentals_box.svg")

    temp = column(data, "Temperature_C")
    humidity = column(data, "Humidity_percent")
    solar = column(data, "Solar_Radiation_MJ_m2")
    rainfall = column(data, "Rainfall_mm")

    histogram_svg(temp, "Temperature (C)", "seoul_bike_temp_hist.svg")
    histogram_svg(humidity, "Humidity (%)", "seoul_bike_humidity_hist.svg")

    scatter_svg(temp, target, "Temperature vs Demand", "Temperature (C)", "Rented Bikes", "seoul_bike_temp_scatter.svg")
    scatter_svg(humidity, target, "Humidity vs Demand", "Humidity (%)", "Rented Bikes", "seoul_bike_humidity_scatter.svg")
    scatter_svg(solar, target, "Solar Radiation vs Demand", "Solar (MJ/m2)", "Rented Bikes", "seoul_bike_solar_scatter.svg")

    # Correlation heatmap
    numeric_keys = [
        "Rented_Bike_Count",
        "Hour",
        "Temperature_C",
        "Humidity_percent",
        "Wind_speed_m_s",
        "Visibility_10m",
        "Solar_Radiation_MJ_m2",
        "Rainfall_mm",
        "Snowfall_cm",
    ]
    numeric_cols = {k: column(data, k) for k in numeric_keys}
    corr_matrix: List[List[float]] = []
    for row_key in numeric_keys:
        corr_matrix.append([pearson(numeric_cols[row_key], numeric_cols[col]) for col in numeric_keys])
    heatmap_svg(numeric_keys, corr_matrix, "Correlation Heatmap", "seoul_bike_corr_heatmap.svg")

    # Modeling
    X, y, feature_names = encode_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lin_coeffs = linear_regression(X_train, y_train)
    lin_preds = predict_linear(X_test, lin_coeffs)
    lin_mse = mse(y_test, lin_preds)

    knn_preds = knn_regression(X_train, y_train, X_test, k=5)
    knn_mse = mse(y_test, knn_preds)

    print(f"Linear Regression MSE: {lin_mse:.2f}")
    print(f"kNN Regression (k=5) MSE: {knn_mse:.2f}")


if __name__ == "__main__":
    main()
