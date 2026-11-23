Overview
Competition ML@DAA - Edition (2026/2027)

This competition is related to the Group Practical Assignment of the course unit Data and Machine Learning (Dados e Aprendizagem Automática), aimed at students of the Master's in Informatics Engineering, Master's in Mathematics and Computing, Master's in Systems Engineering, Master's in Data Science and Engineering, Master's in Artificial Intelligence, Master's in Cybersecurity, Master's in Advanced Computing, and Master's in Computer Networks.
Competition Description

The provided dataset originates from a nationwide survey conducted in Portugal during the 2019 calendar year. This study aimed to collect comprehensive information about the resident population, covering dimensions such as health status, healthcare utilization, determinants and lifestyles, as well as general socioeconomic characteristics.

In this competition, students are expected to develop and optimize Machine Learning models capable of predicting an individual's age group based on their profile, with the ultimate goal of characterizing these age groups.

Objective

The objective of this project is to apply Machine Learning models to predict each individual's age group based on their socioeconomic data, health status, healthcare utilization, determinants, and lifestyles.

Beyond the accurate prediction of the age bracket, the main focus centers on the detection of profile inconsistencies and anomalies, culminating in the detailed characterization of each age group at both socioeconomic and health levels.

Whenever the predictive model identifies a biological and lifestyle profile characteristic of a senior age bracket (e.g., 60-64 years old) in an individual with a declared age of 18 years — or vice versa —, we are facing an epidemiological anomaly, a biological profile deviation, or a potential recording error.

Thus, students must:

    Train and optimize robust models to accurately predict the 16 age groups in the dataset;
    Identify profile anomalies: analyze cases where there is a significant deviation between the declared real age and the age predicted by the model;
    Characterize the age groups: extract socioeconomic, pathological, and behavioral patterns that define each age group, as well as the factors that justify the identified anomalies.

The dataset consists of 237 explanatory variables covering various domains:

- Demographics and work: Region (NUTS), education level (HATLEVEL), labor status (MAINSTAT);

- Health status and chronic diseases: self-perceived health status (HS1), presence of hypertension, diabetes, depression, chronic pain (family of variables CD1A to CD1P);

- Limitations and autonomy: vision/hearing/mobility difficulties (PL1-PL9) and autonomy in daily tasks (PC1A-PC1E, HA1A-HA1G);

- Lifestyle and habits: physical exercise (PE1-PE8), tobacco consumption (IN43-SK6), alcohol consumption (AL1-AL6), and dietary habits (DH1-IN71).

Two datasets are provided. One of them, the training dataset, should be used to train and tune the Machine Learning model. The second should be used as the test dataset. Thus, for each entry in the test dataset, you must predict the age range, using the 0 to 15 scale that encodes the following age groups in years:

    15-17
    18-19
    20-24
    25-29
    30-34
    35-39
    40-44
    45-49
    50-54
    55-59
    60-64
    65-69
    70-74
    75-79
    80-84
    85+

Metric

The evaluation metric used will be the Macro F1-Score across the 16 age group classes:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Macro F1</mtext>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>N</mi>
  </mfrac>
  <munderover>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>0</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>N</mi>
      <mo>&#x2212;<!-- − --></mo>
      <mn>1</mn>
    </mrow>
  </munderover>
  <msub>
    <mtext>F1-Score</mtext>
    <mi>i</mi>
  </msub>
</math>


where

N = 16

represents the 16 age groups.

Selecting the Macro F1-Score ensures that all age groups carry the same weight in the model's evaluation, regardless of the number of individuals in each bracket within the dataset.
Submission File Format

You can submit up to 3 valid files per day (invalid submissions do not count towards the daily limit). In the end, only one file (chosen by you) will count towards the competition ranking. You must submit the file in CSV format, with exactly 3655 entries (the same as in the test dataset) plus the header AGE_RANGE. The submission will cause an error if the file is improperly formatted or has extra rows and/or columns. You can find an example of a submission file in the Data tab of this competition.

Please do not build the file by hand!




Rules


Two main datasets are provided:

- dataset_train.csv

- dataset_test.csv

The training dataset (dataset_train.csv) should be used to develop, train, and optimize the Machine Learning model. In this set, information regarding the individual's true age range (AGE_RANGE) is provided. The model to be developed must be based on the socioeconomic profile, self-perceived health status, chronic pathologies, autonomy in daily tasks, and lifestyle habits (alcohol, tobacco, diet, and physical exercise) that characterize each individual.

In turn, the test dataset (dataset_test.csv) will be used to validate the generalization ability and accuracy (Macro F1-Score) of the model on unseen data. In this dataset, the target class corresponding to the age range is not provided. You must predict this class for each entry.

Additionally, students must use the model's predictions to identify epidemiological anomalies: cases where the model's estimated age deviates significantly from the declared true age, signaling prematurely aged or atypically young biological profiles, and detailed characterize each age group at both socioeconomic and health levels.
Features:

    PID - individual identification number;
    HHID - housing unit/household identification number;
    REGION, NUTS1, and NUTS2_AG - geographical region of residence of the individual;
    DEG_URB - degree of urbanization of the area of residence (densely, intermediate, or thinly populated area);
    SEX - sex of the individual;
    HATLEVEL - highest level of education completed;
    MAINSTAT - labor status;
    JOBISCO_COD and LOCNACE_COD - main occupation and economic activity sector;
    HHINCOME - quintile of net monthly income per equivalized adult of the household;
    HS1 - general self-perceived health status;
    HS2 and HS3 - presence of long-standing health problems and degree of limitation in daily activities;
    CD1A – CD1P - binary indicators of presence/absence of chronic diseases and conditions in the past 12 months;
    PL1 – PL9 - degree of difficulty and use of aids for sensory and motor functions;
    PC1A – PC1E - degree of difficulty and need for help in personal care activities;
    HA1A – HA1G - degree of difficulty in performing household and administrative tasks;
    BM1 and BM2 - anthropometric data;
    PE1 – PE8 - physical effort level at work, walking/cycling commuting habits, and weekly physical exercise frequency;
    DH1 – IN71 - food consumption frequency;
    IN43 – SK6 - tobacco consumption status and history;
    AL1 – AL6 - frequency and volume of alcoholic beverage consumption and heavy episodic drinking occasions;
    SS1 – SS3 - level of social support, contact network, and perception of help in case of need;
    AGE_RANGE - age group of the individual.
