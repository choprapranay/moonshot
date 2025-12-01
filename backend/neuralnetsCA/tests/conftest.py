import sys
from pathlib import Path

# Add the neuralnetsCA directory to Python path so imports work correctly
neuralnets_dir = Path(__file__).parent.parent
sys.path.insert(0, str(neuralnets_dir))
