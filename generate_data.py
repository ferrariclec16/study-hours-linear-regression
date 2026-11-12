"""Generate 100 synthetic student records for practice, not real survey data.
학습용 가상 학생 100명의 데이터를 생성합니다. 실제 조사 자료가 아닙니다.
"""

import csv
import random
from pathlib import Path


def main():
    rng = random.Random(42)
    path = Path(__file__).resolve().parent / "study_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["study_hours", "exam_score"])
        for _ in range(100):
            # Average daily study time during the week before the exam (hours/day).
            # 시험 전 1주일의 하루 평균 공부 시간(시간/일)
            hours = round(rng.uniform(0.5, 9.0), 1)
            # Add random noise to the linear relationship to simulate individual differences.
            # 선형 관계에 개인차를 흉내 낸 무작위 오차를 추가합니다.
            score = round(min(100, max(0, 30 + 7 * hours + rng.gauss(0, 7))), 1)
            writer.writerow([hours, score])
    print(f"Synthetic data saved: {path}\n가상 데이터 저장: {path}")


if __name__ == "__main__":
    main()
