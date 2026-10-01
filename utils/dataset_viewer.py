"""Backward-compatible re-export wrapper — all logic lives in viewer.py."""

from utils.viewer import (
    DatasetViewer,
    get_dataset_list,
    get_dataset_info,
    print_dataset_list,
    print_dataset_summary,
    print_dataset_details,
    print_all_datasets,
    print_datasets,
)

__all__ = [
    'DatasetViewer',
    'get_dataset_list',
    'get_dataset_info',
    'print_dataset_list',
    'print_dataset_summary',
    'print_dataset_details',
    'print_all_datasets',
    'print_datasets',
]


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
            viewer.print_dataset_info(cmd)
    else:
        viewer.print_summary_table()
