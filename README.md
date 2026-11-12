# Study Hours and Exam Score Prediction
공부 시간에 따른 시험 점수 예측

A beginner-friendly simple linear regression project with English explanations followed by Korean translations.  
영어 설명 바로 아래에 한국어 번역을 배치한 초보자용 단순 선형 회귀 프로젝트입니다.

**[View the executed notebook](AI_Linear_Regression_Study_Scores.ipynb)**  
**[실행 결과가 포함된 노트북 보기](AI_Linear_Regression_Study_Scores.ipynb)**

[Run in Google Colab](https://colab.research.google.com/github/ferrariclec16/study-hours-linear-regression/blob/main/AI_Linear_Regression_Study_Scores.ipynb)  
[Google Colab에서 실행](https://colab.research.google.com/github/ferrariclec16/study-hours-linear-regression/blob/main/AI_Linear_Regression_Study_Scores.ipynb)

The notebook includes the data, code, output tables, and plots; no separate CSV download is needed.  
노트북에 데이터, 코드, 결과 표와 그래프가 포함되어 있어 CSV를 별도로 다운로드할 필요가 없습니다.

### Dataset
데이터

The dataset contains 100 synthetic student records, not real survey data.  
실제 조사 자료가 아닌 가상 학생 데이터 100개를 사용합니다.

`study_hours`: average daily study hours during the week before the exam.  
`study_hours`: 시험 전 1주일의 하루 평균 공부 시간(시간/일).

`exam_score`: exam score from 0 to 100.  
`exam_score`: 0~100점 범위의 시험 점수.

`generate_data.py` uses random seed 42 and the formula `30 + 7 × hours + noise`, then clips scores to 0–100.  
`generate_data.py`는 난수 시드 42와 `30 + 7 × 공부 시간 + 오차`를 사용하고 점수를 0~100으로 제한합니다.

Noise follows a normal distribution with mean 0 and standard deviation 7.  
오차는 평균 0, 표준편차 7인 정규분포를 따릅니다.

### Run locally
로컬에서 실행

Use Python 3.10 or later and run these commands in the project folder.  
Python 3.10 이상을 사용하고 프로젝트 폴더에서 아래 명령을 실행하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

To predict a score for a different study time:  
다른 공부 시간으로 점수를 예측하려면:

```bash
python main.py --hours 6.5
```

To regenerate the same data, run `python generate_data.py`; this overwrites the CSV.  
같은 데이터를 다시 생성하려면 `python generate_data.py`를 실행하세요. 기존 CSV를 덮어씁니다.

### Method and results
방법과 결과

Split the data into 80 training records and 20 test records using random seed 42.  
난수 시드 42로 학습용 80개와 테스트용 20개로 데이터를 분할합니다.

Fit the linear regression model on training data only, then evaluate it on the test set.  
학습 데이터만으로 선형 회귀 모델을 학습한 뒤 테스트 데이터에서 평가합니다.

Fitted equation: predicted score = 7.44 × study hours + 27.94.  
학습된 회귀식: 예측 점수 = 7.44 × 공부 시간 + 27.94.

Test MAE: 4.42 points; RMSE: 5.35 points; R²: 0.905.  
테스트 MAE: 4.42점, RMSE: 5.35점, R²: 0.905.

Training-mean baseline MAE: 15.05 points.  
학습 데이터 평균값 기준 모델의 MAE: 15.05점.

Predicted score for five hours per day: 65.2 points.  
하루 5시간 공부할 때 예측 점수: 65.2점.

MAE and RMSE measure prediction error in score points; lower values are better.  
MAE와 RMSE는 점 단위의 예측 오차를 나타내며 낮을수록 좋습니다.

R² is the coefficient of determination, not an accuracy percentage.  
R²는 결정계수이며 정확도 백분율이 아닙니다.

### Generated files
생성되는 파일

Each run updates the following files in the `outputs` folder.  
실행할 때마다 `outputs` 폴더의 아래 파일이 갱신됩니다.

`results.txt`: model equation, evaluation metrics, and the new prediction.  
`results.txt`: 회귀식, 평가 지표, 새로운 예측 결과.

`test_predictions.csv`: actual scores, predicted scores, and absolute errors.  
`test_predictions.csv`: 실제 점수, 예측 점수, 절대 오차.

`regression_plot.png`: training/test scatter plot and fitted regression line.  
`regression_plot.png`: 학습·테스트 산점도와 회귀선.

### Limitations
한계

A linear relationship was built into the synthetic data, so these results do not demonstrate real-world predictive performance.  
가상 데이터에 선형 관계를 넣어 생성했으므로 이 결과는 실제 예측 성능을 입증하지 않습니다.

Study time alone does not account for sleep, prior knowledge, or exam difficulty.  
공부 시간만으로는 수면, 사전 지식, 시험 난이도를 고려할 수 없습니다.

The slope is not evidence of causation, and predictions outside the training range may be unreliable.  
기울기는 인과관계의 근거가 아니며 학습 범위를 벗어난 예측은 신뢰하기 어려울 수 있습니다.

To use real data, keep the same column names and units and check missing or invalid values first.  
실제 데이터로 바꾸려면 같은 열 이름과 단위를 유지하고 결측치나 잘못된 값을 먼저 확인하세요.
