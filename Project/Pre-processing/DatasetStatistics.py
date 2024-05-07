# Import Libraries
import pathlib



"""
    0. Get the dataset statistics.
    1. (Halfway done) Pre-process (scaling, sorting into folders, decolorizing)
    2. Create tensor chunks using pickle (memory management don't run out of RAM)
    3. Define the model(s).
    4. Training
    5. Testing
    6. ???
"""
def main():

    # This gets the original dataset stats
    import DatasetAnalysis.DatasetStatistics as ds

    # Prepare the directories paths for the dataset.

    abs_path = pathlib.Path(__file__).parent.absolute()

    path = abs_path.parent / "Datasets"

    dataset_directory_path: pathlib.Path = path

    # Creates a DatasetStatisticsManager object and provides its result to stdout.
    ds.DatasetStatisticsManager(dataset_directory_path=dataset_directory_path, num_of_classes=3)



main()
