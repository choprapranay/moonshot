from domain.boundaries import (
    BuildDatasetOutputBoundary,
    BuildDatasetOutputData
)
from interface_adapters.presenters.view_model import ViewModel, DatasetViewModel


class DatasetPresenter(BuildDatasetOutputBoundary):
    def __init__(self, view_model: ViewModel):
        self._view_model = view_model
    
    def present_dataset(self, output_data: BuildDatasetOutputData) -> None:
        if output_data.success and output_data.dataset_info:
            dataset_vm = DatasetViewModel(
                success=True,
                num_samples=output_data.dataset_info.num_samples,
                num_batters=output_data.dataset_info.num_batters,
                num_pitch_types=output_data.dataset_info.num_pitch_types,
                num_outcomes=output_data.dataset_info.num_outcomes,
                outcome_distribution=output_data.dataset_info.outcome_distribution,
                dataset=output_data.dataset,
                raw_data=output_data.raw_data
            )
        else:
            dataset_vm = DatasetViewModel(
                success=False,
                error_message=output_data.error_message
            )
        
        self._view_model.update_dataset(dataset_vm)
