
To try our project online, use this link: `https://moonshot-deploy.vercel.app/`, however, keep in mind our backend runs on free tier of render, which limits our RAM usage thus we were only able to deploy a miniature version of our ML model that is only trained on two weeks worth of data. Which means the heatmap will have much less data and less accurate. Please run localhost to try out the full version of the ML model.

This is for BaiT, a baseball analytics company focused in distilling large-scale performance data into actionable insights.

Moonshot is an AI-driven decision tool to help batters improve their pitch discipline. In baseball, that is one of the most important factors to determining the victorious team. This can work well with something like Trajekt, helping target batters' weak spots to improve overall runs per plate appearance.

Moonshot allows you to analyse the optimal swing/take locations for a batter through a colored heatmap where red regions represent where batters should not swing and blue regions represent where batters should swing. This is overlaid with a game of your choosing and relevant pitches said batter has faced. This way, you can analyse where batters' weak spots (and for that matter where pitchers can target) and create a personalized training regimen. 


Implementation of SOLID design principles: 
- Single Responsibility Principle (SRP): We split our system into a clear separation of layers (domain, use-cases, etc.) and each file only focuses on a singular task. For instance, use-cases only coordinate logic (ex: fetch_game_data.py only fetches game data) and repositories handle external data access. 

- Open/Closed Principle: We keep core logic untouched and extend behaviour by adding new adapter implementations. For instance, our use-cases depend on the interfaces found in app/domain/interfaces.py. The system only grows by adding to the repository classes, not to the business logic. 

- Liskov Substitution Principle: Our use cases only depend on the domain interfaces, allowing any repository implementation (Supabase, PyBaseball, tests) to be swapped in without altering the use case code. As all implementations follow the same interface contract, the system behaves correctly no matter which one is injected (this is confirmed by our tests). 

- Interface Segregation Principle: Our codebase employs many small, focused interfaces (e.g., game data, model storage) so that modules only depend on the exact capabilities they require. This allows the infrastructure and ML components to only import the specific contracts that are relevant. This prevents bloat and keeps our system clean. 

- Dependency Inversion Principle: The use cases depend strictly on domain interfaces (ex: at startup, a concrete repository like SupabaseRepository is injected), thereby allowing high-level logic to stay independent of external systems and allows us to replace infrastructure without altering business logic and test the whole system in a series of mocks. 

Implementation of design patterns: 
- One design pattern we've implemented is the Adapter Design Pattern, which allows us to convert the interface of one module to one required by another. This is exemplified in our neural network implementation, where we have adapters such as a DatasetController to convert raw external output into domain input required by the use case. 
- A pattern we've applied outside of the ones discussed in class is the Singleton Design Pattern, which is implemented in the connection to Supabase. This ensures that only one connection to the database exists, thereby saving computational resources. 

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
pytest app/tests/test_fetch_game_data.py --cov=app/use_cases/fetch_game_data --cov-report=term-missing --cov-report=html'''

```

# Test Suite for NeuralNetsCA

This test suite tests code coverage for all of NeuralNetsCA use cases. `TestInference` and `TestBuildDatasetUseCase` and `TestTrainModel`

```bash
# From backend 
python -m pytest tests/test_train_NN.py -v
```

# Test Suite for LinearRegression

This test suite tests code coverage for the run_value_regression use case interactor. 

```bash
# From backend directory
pytest tests/test_run_value_regression.py \
  --cov=backend.use_cases.run_value_regression \
  --cov-report=html

```


