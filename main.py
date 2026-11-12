"""A simple linear regression project that predicts exam scores from study hours.
공부 시간으로 시험 점수를 예측하는 단순 선형 회귀 프로젝트.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots even when no display is available.
# 화면이 없는 환경에서도 그래프를 파일로 저장합니다.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=5.0, help="Average daily study hours / 예측할 하루 평균 공부 시간")
    args = parser.parse_args()
    if not np.isfinite(args.hours) or not 0 <= args.hours <= 24:
        parser.error("Enter a finite number from 0 to 24.\n공부 시간은 0~24 사이의 유한한 숫자로 입력하세요.")

    # 1. Load data: X is a two-dimensional table and y is a one-dimensional target.
    # 1. 데이터 읽기: X는 2차원 표, y는 1차원 값입니다.
    data = pd.read_csv(BASE_DIR / "study_scores.csv")
    X = data[["study_hours"]]
    y = data["exam_score"]

    # 2. Split into 80% training data and 20% test data.
    # 2. 학습용 80%, 테스트용 20%로 분리합니다.
    # Do not use test data to train the model.
    # 테스트 데이터는 모델 학습에 사용하지 않습니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Predicted score = slope × study hours + intercept.
    # 3. 예측 점수 = 기울기 × 공부 시간 + 절편
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Evaluate performance on unseen test data.
    # 4. 처음 보는 테스트 데이터에서 성능을 평가합니다.
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    # The baseline always predicts the mean training score, ignoring study hours.
    # 공부 시간을 무시하고 학습 데이터 평균만 예측하는 기준 모델입니다.
    baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))

    # 5. Predict a score for the study time entered by the user.
    # 5. 사용자가 입력한 공부 시간에 대한 점수를 예측합니다.
    new_data = pd.DataFrame({"study_hours": [args.hours]})
    predicted_score = model.predict(new_data)[0]
    report = [
        "Study Hours and Exam Score Prediction (Synthetic Data)\n공부 시간에 따른 시험 점수 예측 (가상 데이터)",
        f"Data: {len(data)} total / {len(X_train)} training / {len(X_test)} test\n데이터: 총 {len(data)}명 / 학습 {len(X_train)}명 / 테스트 {len(X_test)}명",
        f"Equation: predicted score = {model.coef_[0]:.2f} × hours + {model.intercept_:.2f}\n회귀식: 예측 점수 = {model.coef_[0]:.2f} × 공부 시간 + {model.intercept_:.2f}",
        f"MAE (mean absolute error): {mae:.2f} points\nMAE (평균 절대 오차): {mae:.2f}점",
        f"RMSE (root mean squared error): {rmse:.2f} points\nRMSE (큰 오차에 더 민감): {rmse:.2f}점",
        f"R² (coefficient of determination): {r2:.3f}\nR² (결정계수): {r2:.3f}",
        f"Training-mean baseline MAE: {baseline_mae:.2f} points\n평균값만 예측하는 기준 모델 MAE: {baseline_mae:.2f}점",
        f"Predicted score for {args.hours:g} hours per day: {predicted_score:.1f} points\n하루 평균 {args.hours:g}시간 공부할 때 예측 점수: {predicted_score:.1f}점",
        "Note: synthetic results do not establish real-world performance or causation.\n주의: 가상 데이터의 결과이며 실제 학생의 성적이나 인과관계를 설명하지 않습니다.",
    ]
    if not X_train["study_hours"].min() <= args.hours <= X_train["study_hours"].max():
        report.append("Warning: outside the training range; extrapolation may be unreliable.\n입력 시간이 학습 범위를 벗어났습니다. 외삽 예측은 신뢰하기 어렵습니다.")
    if not 0 <= predicted_score <= 100:
        report.append("Warning: prediction outside 0–100; the value has not been clipped.\n선형 회귀 예측이 점수 범위(0~100)를 벗어났습니다. 점수를 강제로 보정하지 않았습니다.")

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    result = X_test.copy()
    result["actual_score"] = y_test
    result["predicted_score"] = predictions
    result["absolute_error"] = abs(y_test - predictions)
    result.sort_index().to_csv(output_dir / "test_predictions.csv", index=False, float_format="%.2f")
    (output_dir / "results.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    # 6. Use English plot labels to avoid requiring an additional Korean font.
    # 6. 한글 폰트 설치 없이도 볼 수 있도록 그래프에는 영문을 사용합니다.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X_train["study_hours"], y_train, label="Training data", alpha=0.65, color="#2563eb")
    ax.scatter(X_test["study_hours"], y_test, label="Test data", marker="s", color="#f59e0b")
    line_X = pd.DataFrame({"study_hours": np.linspace(X["study_hours"].min(), X["study_hours"].max(), 100)})
    ax.plot(line_X["study_hours"], model.predict(line_X), color="#dc2626", label="Linear regression")
    ax.set(title="Study Hours vs. Exam Score (Synthetic Data)", xlabel="Average study hours per day", ylabel="Exam score (out of 100)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "regression_plot.png", dpi=180)
    plt.close(fig)
    print("\n".join(report))
    print(f"Output folder: {output_dir}\n결과 저장 폴더: {output_dir}")


if __name__ == "__main__":
    main()
