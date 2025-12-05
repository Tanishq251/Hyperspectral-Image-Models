import sys
from pathlib import Path

# Add parent directory to path so we can import models
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import list_models, get_model_config
from tabulate import tabulate


class ModelViewer:
    """View and display available models"""
    
    def __init__(self):
        self.models = list_models()
    
    def list_models(self):
        """Print list of all available models"""
        print(f"\n{'='*60}")
        print(f"Available Models ({len(self.models)})")
        print(f"{'='*60}")
        for i, name in enumerate(self.models, 1):
            print(f"  {i:2d}. {name}")
        print(f"{'='*60}\n")
    
    def print_summary_table(self):
        """Print summary table of all models"""
        table_data = []
        for name in self.models:
            cfg = get_model_config(name)
            expects_4d = cfg.get('expects_4d', False)
            table_data.append([
                name,
                'Yes' if expects_4d else 'No'
            ])
        
        headers = ['Model Name', 'Expects 4D Input']
        print(f"\n{'='*70}")
        print("Model Summary")
        print(f"{'='*70}")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print(f"{'='*70}\n")
    
    def print_model_info(self, model_name):
        """Print detailed info for a specific model"""
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found")
            self.list_models()
            return
        
        cfg = get_model_config(model_name)
        
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        print(f"Expects 4D Input: {cfg.get('expects_4d', False)}")
        
        if cfg:
            print(f"\nConfiguration:")
            for key, value in cfg.items():
                if key != 'expects_4d':
                    print(f"  {key}: {value}")
        
        print(f"{'='*70}\n")
    
    def print_all_models(self):
        """Print detailed info for all models"""
        for model_name in self.models:
            self.print_model_info(model_name)


# Convenience functions for simple usage
def get_model_list():
    """Get list of all available models"""
    return list_models()


def print_model_list():
    """Print list of all models"""
    viewer = ModelViewer()
    viewer.list_models()


def print_model_summary():
    """Print summary table of all models"""
    viewer = ModelViewer()
    viewer.print_summary_table()


def print_model_details(model_name):
    """Print detailed info for a specific model"""
    viewer = ModelViewer()
    viewer.print_model_info(model_name)


def print_all_models():
    """Print detailed info for all models"""
    viewer = ModelViewer()
    viewer.print_all_models()


def print_models(*model_names):
    """Print detailed info for multiple models"""
    viewer = ModelViewer()
    for name in model_names:
        viewer.print_model_info(name)


if __name__ == '__main__':
    import sys
    
    viewer = ModelViewer()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'list':
            viewer.list_models()
        elif cmd == 'summary':
            viewer.print_summary_table()
        elif cmd == 'all':
            viewer.print_all_models()
        else:
            # Treat as model name
            viewer.print_model_info(cmd)
    else:
        # Default: show summary
        viewer.print_summary_table()
