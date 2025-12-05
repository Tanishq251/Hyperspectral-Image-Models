import yaml
from pathlib import Path
from tabulate import tabulate


# Convenience functions for simple usage
def get_dataset_list(config_path='config/dataset.yaml'):
    """Get list of all available datasets"""
    viewer = DatasetViewer(config_path)
    return list(viewer.datasets.keys())


def get_dataset_info(dataset_name, config_path='config/dataset.yaml'):
    """Get info dict for a specific dataset"""
    viewer = DatasetViewer(config_path)
    if dataset_name in viewer.datasets:
        return viewer.datasets[dataset_name]
    return None


def print_dataset_list(config_path='config/dataset.yaml'):
    """Print list of all datasets"""
    viewer = DatasetViewer(config_path)
    viewer.list_datasets()


def print_dataset_summary(config_path='config/dataset.yaml'):
    """Print summary table of all datasets"""
    viewer = DatasetViewer(config_path)
    viewer.print_summary_table()


def print_dataset_details(dataset_name, config_path='config/dataset.yaml'):
    """Print detailed info for a specific dataset"""
    viewer = DatasetViewer(config_path)
    viewer.print_dataset_info(dataset_name)


def print_all_datasets(config_path='config/dataset.yaml'):
    """Print detailed info for all datasets"""
    viewer = DatasetViewer(config_path)
    viewer.print_all_datasets()


def print_datasets(*dataset_names, config_path='config/dataset.yaml'):
    """Print detailed info for multiple datasets"""
    viewer = DatasetViewer(config_path)
    for name in dataset_names:
        viewer.print_dataset_info(name)


class DatasetViewer:
    """View and display dataset information from config in CLI"""
    
    def __init__(self, config_path='config/dataset.yaml'):
        self.config_path = config_path
        self.datasets = self._load_config()
    
    def _load_config(self):
        """Load dataset config from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('datasets', {})
        except FileNotFoundError:
            print(f"Error: Config file not found at {self.config_path}")
            return {}
    
    def list_datasets(self):
        """Print list of all available datasets"""
        if not self.datasets:
            print("No datasets found")
            return
        
        dataset_names = list(self.datasets.keys())
        print(f"\n{'='*60}")
        print(f"Available Datasets ({len(dataset_names)})")
        print(f"{'='*60}")
        for i, name in enumerate(dataset_names, 1):
            print(f"  {i:2d}. {name}")
        print(f"{'='*60}\n")
    
    def print_summary_table(self):
        """Print summary table of all datasets"""
        if not self.datasets:
            print("No datasets found")
            return
        
        table_data = []
        for name, info in self.datasets.items():
            table_data.append([
                name,
                info.get('height', 'N/A'),
                info.get('width', 'N/A'),
                info.get('bands', 'N/A'),
                info.get('num_classes', 'N/A'),
                info.get('labeled_samples', 'N/A'),
            ])
        
        headers = ['Dataset', 'Height', 'Width', 'Bands', 'Classes', 'Labeled Samples']
        print(f"\n{'='*100}")
        print("Dataset Summary")
        print(f"{'='*100}")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print(f"{'='*100}\n")
    
    def print_dataset_info(self, dataset_name):
        """Print detailed info for a specific dataset"""
        if dataset_name not in self.datasets:
            print(f"Error: Dataset '{dataset_name}' not found")
            self.list_datasets()
            return
        
        info = self.datasets[dataset_name]
        
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        print(f"Spatial Dimensions: {info.get('height', 'N/A')} x {info.get('width', 'N/A')}")
        print(f"Number of Bands: {info.get('bands', 'N/A')}")
        print(f"Number of Classes: {info.get('num_classes', 'N/A')}")
        print(f"Total Samples: {info.get('total_samples', 'N/A'):,}")
        print(f"Labeled Samples: {info.get('labeled_samples', 'N/A'):,}")
        
        # Print class-wise information
        class_names = info.get('class_names', [])
        class_samples = info.get('class_samples', {})
        
        if class_names:
            print(f"\n{'Class Details':^70}")
            print(f"{'-'*70}")
            
            table_data = []
            for i, class_name in enumerate(class_names, 1):
                samples = class_samples.get(f'Class_{i}', 0)
                table_data.append([
                    i,
                    class_name,
                    f"{samples:,}"
                ])
            
            headers = ['#', 'Class Name', 'Samples']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        print(f"{'='*70}\n")
    
    def print_all_datasets(self):
        """Print detailed info for all datasets"""
        if not self.datasets:
            print("No datasets found")
            return
        
        for dataset_name in self.datasets.keys():
            self.print_dataset_info(dataset_name)


if __name__ == '__main__':
    import sys
    
    viewer = DatasetViewer()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'list':
            viewer.list_datasets()
        elif cmd == 'summary':
            viewer.print_summary_table()
        elif cmd == 'all':
            viewer.print_all_datasets()
        else:
            # Treat as dataset name
            viewer.print_dataset_info(cmd)
    else:
        # Default: show summary
        viewer.print_summary_table()
