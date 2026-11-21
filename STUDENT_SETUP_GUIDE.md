# SignCadence Student Setup Guide

## 1. About SignCadence

SignCadence is an AI-assisted communication project initially focused on Namibian Sign Language (NSL).

The project uses:
- Python 3.12 – programming language
- MediaPipe – hand landmark detection
- OpenCV – webcam access
- NumPy – numerical processing
- Scikit-learn – machine learning
- Git and GitHub – project collaboration

**Important:** SignCadence is currently a research prototype, not a validated NSL translator. The AI model still requires sufficient consented and community-verified NSL data.

## 2. Software requirements

Before starting, install:

1. Python 3.12
2. Git
3. Visual Studio Code
4. A laptop with a working webcam

## 3. Download SignCadence

Open PowerShell or the VS Code terminal.

Clone the repository:

```powershell
git clone https://github.com/administratortjiuana-hub/SignCadence.git
```

Enter the project folder:

```powershell
cd SignCadence
```

## 4. Create a virtual environment

Run:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 5. Install dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project packages:

```powershell
python -m pip install -r requirements.txt
```

Check the installation:

```powershell
python -m pip check
```

Expected result:

```text
No broken requirements found.
```

## 6. Open the project in VS Code

Run:

```powershell
code .
```

## 7. Project structure

```text
SignCadence/
│
├── dataset/
│   ├── landmarks/
│   └── processed/
│
├── models/
│   └── hand_landmarker.task
│
├── src/
│   ├── main.py
│   ├── vocabulary_manager.py
│   ├── preprocessing/
│   └── recognition/
│       ├── train_model.py
│       └── recognize_sign.py
│
├── requirements.txt
├── README.md
├── STUDENT_SETUP_GUIDE.md
└── .gitignore
```

Some research datasets and trained model files may not be included in the repository. The MediaPipe hand landmark model file is included.

## 8. Run SignCadence

From the project root, run:

```powershell
python src/main.py
```

The application displays the 10 provisional vocabulary concepts and their verification statuses.

Concepts marked as pending cannot be collected as official NSL training examples. This is an intentional research safeguard.

## 9. Dataset preparation

After appropriate consent and verification have been obtained, authorised research data can be processed using:

```powershell
python src/preprocessing/prepare_dataset.py
```

The script prepares hand landmark features for machine learning.

## 10. Train the AI model

Once sufficient verified data has been collected from multiple signers, run:

```powershell
python src/recognition/train_model.py
```

The training program checks whether the dataset contains enough examples, labels, and different signers.

Training will stop if the dataset does not meet its requirements.

## 11. Live sign recognition

After a valid trained model has been created, run:

```powershell
python src/recognition/recognize_sign.py
```

This starts the webcam-based recognition prototype.

## 12. Basic Git commands

Check project changes:

```powershell
git status
```

Download the latest project changes:

```powershell
git pull origin main
```

Stage changes:

```powershell
git add .
```

Commit changes:

```powershell
git commit -m "Describe your changes"
```

Push changes to GitHub:

```powershell
git push origin main
```

## 13. Research ethics and data protection

All students must follow these rules:

- Obtain informed consent before recording participants.
- Obtain explicit permission for AI training and intended data sharing.
- Work with fluent NSL signers to verify vocabulary.
- Keep signer identities anonymous.
- Never upload private participant recordings or research datasets to GitHub.
- Do not present unverified concepts as official NSL signs.
- Respect the rights and decisions of the Deaf community.

## 14. Project repository

GitHub:

https://github.com/administratortjiuana-hub/SignCadence

**Project goal:** Develop a responsible, community-validated AI-assisted communication tool for Namibian Sign Language.