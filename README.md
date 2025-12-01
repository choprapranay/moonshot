
To try our project online, use this link: `https://moonshot-deploy.vercel.app/`, however, keep in mind our backend runs on free tier of render, which limits our RAM usage thus we were only able to deploy a miniature version of our ML model that is only trained on two weeks worth of data. Which means the heatmap will have much less data and less accurate. Please run localhost to try out the full version of the ML model.

This is for the Bait, a baseball analytics company

Moonshot is a powerful tool to help batters improve their pitch discipline. As in baseball, that is one of the most important factors to determining the victorious team. This can work well with something like Trajekt, helping target batters' weak spots to improve overall runs per plate appearance.

Moonshot allows you to analyse the optimal swing/take locations for a batter through a colored heatmap where red regions represent where batters should not swing and blue regions represent where batters should swing. This is overlaid with a game of your choosing and relevant pitches said batter has faced. This way, you can analyse where batters' weak spots (and for that matter where pitchers can target) and create a personalized training regimen. 

Authors:
Yash Jain
Pranay Chopra
Sumedh Gadepalli 
Jeff Lu
Sambhav Athreya

Setup instructions:
# Getting Started

To get this project up and running on your local machine, follow these steps:

## Prerequisites

Make sure you have the following installed:

- [Node.js](https://nodejs.org/) (version 14 or higher)  
- [Python](https://www.python.org/) (version 3.8 or higher)  
- [Pip](https://pip.pypa.io/en/stable/) (Python package manager)

---

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/choprapranay/moonshot.git
cd moonshot
```
### 2. Navigate to the backend directory and install the required Python packages:
```bash
cd backend
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```
### 3. Navigate to the frontend directory and install the required Node.js packages:
```bash
cd frontend
npm install
```
## Configuration
Create a .env file inside the backend directory and add your environment variables:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

Create a .env.local file inside the frontend directory and add your environment variables:
```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Running the Application
Start the backend server:
```bash
cd backend
uvicorn app.main:app --reload
```
Start the frontend development server:
```bash
cd frontend
npm run dev
```
Once both servers are running, open your browser and navigate to `http://localhost:3000` to view the application.

### Testing
# Test Suite for FetchGameDataUseCase

This test suite provides 100% code coverage for the `FetchGameDataUseCase` use case interactor.

## Running the Tests

### Install Dependencies
```bash
cd backend
pip install pytest pytest-cov pandas fastapi
```

### Run Tests with Coverage
```bash
# From backend directory
pytest app/tests/test_fetch_game_data.py --cov=app/use_cases/fetch_game_data --cov-report=term-missing --cov-report=html

```

# Test Suite for NerualNetsCA

This test suite tests code coverage for all of NeuralNetsCA use cases. `TestInference` and `TestBuildDatasetUseCase` and `TestTrainModel`

```bash
# From backend 
python -m pytest tests/test_train_NN.py -v
```

