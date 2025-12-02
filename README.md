#

Disclaimer: To see our project online, use this link: `https://moonshot-deploy.vercel.app/`, however, note that our backend employs the free tier of render, which limits our RAM usage. Thus, we were only able to deploy a miniature version of our ML model that is only trained on two weeks worth of data. This has a direct influence on the accuracy of the heatmaps that are displayed. Please run localhost to try out the full version of the ML model.

This project was developed in partnership with BaiT, a baseball analytics company focused in distilling large-scale performance data into actionable insights. 

### Introduction and Problem Domain

Modern baseball generates extremely large amounts of pitch-level data. Every pitch contains information about location, movement, velocity, pitch type, and outcome. Although teams collect all of this, the insights that matter most for improving hitting performance remain difficult to extract. Coaches still need clear answers to simple but essential questions: **Where should a batter swing? Where should they take? How can pitchers exploit a hitter’s weaknesses?**

Moonshot addresses this challenge by transforming raw Statcast data into personalized swing or take recommendations. Developed in partnership with BaiT, Moonshot produces visual outputs that support targeted training programs and informed game-day strategy.

### Problem Statement

Pitch discipline is one of the strongest predictors of run production, yet most tools only describe past outcomes rather than what a player should do going forward. Traditional heatmaps do not incorporate run expectancy or the likelihood of different pitch outcomes.

Moonshot solves this by computing the expected run value for swinging or taking at any pitch location. The system:

1. Trains a neural network to estimate the probability of each possible pitch outcome for a batter.
2. Runs a regression model to determine the run value associated with each outcome.
3. Combines these probabilities and run values to compute an expected value (EV) for swing versus take decisions.
4. Visualizes the results as a color-coded map where **blue indicates swing** and **red indicates take**, optimized to maximize the batter’s run expectancy.

This allows Moonshot to reveal hitter weaknesses, pitcher attack zones, and training priorities grounded in objective data rather than intuition.


### Implementation of SOLID design principles: 
- Single Responsibility Principle (SRP): We split our system into a clear separation of layers (domain, use-cases, etc.) and each file only focuses on a singular task. For instance, use-cases only coordinate logic (ex: fetch_game_data.py only fetches game data) and repositories handle external data access. 

- Open/Closed Principle: We keep core logic untouched and extend behaviour by adding new adapter implementations. For instance, our use-cases depend on the interfaces found in app/domain/interfaces.py. The system only grows by adding to the repository classes, not to the business logic. 

- Liskov Substitution Principle: Our use cases only depend on the domain interfaces, allowing any repository implementation (Supabase, PyBaseball, tests) to be swapped in without altering the use case code. As all implementations follow the same interface contract, the system behaves correctly no matter which one is injected (this is confirmed by our tests). 

- Interface Segregation Principle: Our codebase employs many small, focused interfaces (e.g., game data, model storage) so that modules only depend on the exact capabilities they require. This allows the infrastructure and ML components to only import the specific contracts that are relevant. This prevents bloat and keeps our system clean. 

- Dependency Inversion Principle: The use cases depend strictly on domain interfaces (ex: at startup, a concrete repository like SupabaseRepository is injected), thereby allowing high-level logic to stay independent of external systems and allows us to replace infrastructure without altering business logic and test the whole system in a series of mocks. 

Implementation of design patterns: 
- One design pattern we've implemented is the Adapter Design Pattern, which allows us to convert the interface of one module to one required by another. This is exemplified in our neural network implementation, where we have adapters such as a DatasetController to convert raw external output into domain input required by the use case. 
- A pattern we've applied outside of the ones discussed in class is the Singleton Design Pattern, which is implemented in the connection to Supabase. This ensures that only one connection to the database exists, thereby saving computational resources. 


### Authors

- [Yash Jain]([https://github.com/YashJain14](https://github.com/InfiniteInbox))
- [Pranay Chopra](https://github.com/choprapranay) 
- [Sumedh Gadepalli](https://github.com/sumedh71)
- [Jeff Lu](https://github.com/Jeff15321) 
- [Sambhav Athreya](https://github.com/SambhavAthreyaGit) 

---

## Tech Stack

### Frontend
- **Next.js** - React framework for server-side rendering and static site generation
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework

### Backend
- **FastAPI** - Modern Python web framework for building APIs
- **Uvicorn** - ASGI server for running FastAPI applications

### Machine Learning Tools
- **PyTorch** - Deep learning framework for neural network training and inference
- **scikit-learn** - Machine learning utilities (LabelEncoder, StandardScaler)
- **NumPy & Pandas** - Data manipulation and numerical computing
- **pybaseball** - MLB Statcast data fetching

### Database & Storage
- **Supabase** - PostgreSQL-based backend-as-a-service for data storage

### Architecture
- **Clean Architecture** - Separation of concerns with domain, use cases, and infrastructure layers [Potential violation: for the linear regression/expected value calculation, we retain the JSON files directly in the folder since they're intermediate, temporary algorithmic outputs, and the domain logic works in tandem with them. They belong to the pipeline of the ML workflow, which is why it's easier to retain them with the business logic. These files are just snapshots of the algorithm’s internal data to allow repeatable runs and debugging. These JSON files are not persistence or infrastructure storage in our case, which is why we believed it would be appropriate to store them with the domain logic.]

(design patterns and use of SOLID principles discussed above)

### Algorithm/ML
Moonshot is built on a suite of two models; the first is a neural network. We utilize a deep neural network to output a distribution of outcome likelihoods, with an embedding layer applied to the batters, pitch type and outcome type prior to feeding into the linear nodes of the model. Fine-tuning was done, and we experimented with a more advanced model using an attention mechanism where the feature and classifier networks were linear nodes followed by BatchNormalization and Dropout. In parallel, using the same training set, we performed a linear regression to compute the average run values that occur for each outcome (ex: a walk yields 0.296 runs on average). Both the probability distributions for each player and the average run values for the outcomes were passed into an algorithm computing the expected values for each batter. The algorithm iterates through the likelihood of swing/take outcomes for each batter and then multiplies each outcome with the average run values it yields (generated by the regression) to then produce expected values for each set of swing/take outcomes. The differences of each are then computed to see which plate decision is more beneficial. 

---

## Getting Started

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

---

## Testing

### FetchGameDataUseCase

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

### NeuralNetsCA

This test suite tests code coverage for all of NeuralNetsCA use cases. `TestInference` and `TestBuildDatasetUseCase` and `TestTrainModel`

```bash
# From backend 
python -m pytest tests/test_train_NN.py -v
```

If you wish to train or run prediction yourself, you should run 

```bash
# From NeuralNetsCA
python main.py --mode train --data-source pybaseball --model-type standard --start-date 2023-03-30 --end-date 2023-04-30 --epochs 5 --lr 0.001 --batch-size 16
```

### LinearRegression

This test suite tests code coverage for the run_value_regression use case interactor. 

```bash
# From backend directory
pytest tests/test_run_value_regression.py \
  --cov=backend.use_cases.run_value_regression \
  --cov-report=html

```


###  USER GUIDE
This guide explains how a coach can use Moonshot to support player development, scouting, and pre-game planning.

Who This Guide Is For

This tool is designed for a batting or pitching coach who:

- Reviews player tendencies
- Prepares hitters for upcoming opponents
- Designs targeted training plans
- Wants quick, visual insights without needing data science skills

### Step 1: Input a Game ID

On the main page of Moonshot:

1. Enter the **Game ID** from the MLB game you want to analyze.
2. Press submit to load the pitch data for that game.
3. Select the **batter** you want to generate the heatmap for.

After selecting the batter, Moonshot filters all pitch events in that game to only those faced by the chosen player.

Moonshot automatically retrieves pitch location, pitch type, release information, and outcomes for that batter within the specified game.

---

### Step 2: View the Heatmap

Once a batter is selected, Moonshot generates a personalized swing or take heatmap using the trained model.

The heatmap shows the optimal decision for each location in the strike zone.

Color scale:

- **Red** indicates the expected value of swinging is lower than taking.
- **Blue** indicates the expected value of swinging is higher than taking.

Coaches can immediately identify:

- Zones where the hitter is most likely to create run value
- Locations where swinging produces negative outcomes
- Weakness areas pitchers can target
- High-value regions hitters should focus on attacking

---

### Step 3: Apply Insights to Coaching and Strategy

#### For Batting Coaches

Use the heatmap to identify:

- Red regions where the hitter should avoid swinging  
- Blue regions where swings create value  
- Areas where the player is chasing too often  
- Zones where the hitter is taking hittable pitches  

These insights guide:

- Pitch recognition drills  
- Machine pitch targeting  
- Strike zone discipline training  
- Approach adjustments before a series  

#### For Pitching Coaches

Use the heatmap to understand:

- Where an opposing hitter struggles  
- How to design attack sequences  
- Which locations consistently produce weak outcomes  
- How to match your pitcher’s strengths with a batter’s weak zones  


