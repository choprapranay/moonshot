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

### View Coverage Report
After running tests, open `htmlcov/index.html` in your browser to see the detailed coverage report.

## Test Coverage

The test suite covers:

### Path 1: Game Exists in Database
- ✅ Game exists with pitches and players
- ✅ Game exists with pitches but no players
- ✅ Game exists with None players
- ✅ Game exists with empty pitches list
- ✅ Game exists with multiple pitches
- ✅ Pitch with None speed
- ✅ Pitch with empty pitch_type
- ✅ Pitch missing speed key

### Path 2: Game Does NOT Exist in Database
- ✅ External service returns None → raises 404
- ✅ External service returns empty DataFrame → raises 404
- ✅ External data with players → saves everything
- ✅ External data with no players
- ✅ External data with empty names Series
- ✅ External data with NaN values
- ✅ External data with whitespace in names (stripped)
- ✅ Multiple pitches with same player
- ✅ Verifies pitches are saved to repository
- ✅ Empty pitch_type becomes empty string
- ✅ All edge cases combined
- ✅ Empty string player names
- ✅ Verifies game saved correctly
- ✅ Date parsing with time component

## Test Structure

All tests use the fake repositories and services:
- `FakeGameRepository`
- `FakePitchRepository`
- `FakePlayerRepository`
- `FakeGameDataService`

This ensures tests are isolated and don't require external dependencies.

