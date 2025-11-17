from dataclasses import dataclass

from domain.interfaces import PitchEventRepositoryInterface, DataPreprocessorInterface
from domain.entities import DatasetSplit

@dataclass
class BuildDatasetConfig:
    start_date: str
    end_date: str

class BuildDatasetUseCase:
    def __init__(self,
                 pitch_event_repo: PitchEventRepositoryInterface,
                 data_preprocessor: DataPreprocessorInterface,
                 config: BuildDatasetConfig):
        self.pitch_event_repo = pitch_event_repo
        self.data_preprocessor = data_preprocessor
        self.config = config

    def execute(self) -> DatasetSplit:
        df = self.pitch_event_repo.get_events(
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        dataset_split = self.data_preprocessor.preprocess(df)
        return dataset_split