"""Backward-compatible re-export wrapper — all logic lives in viewer.py."""

from utils.viewer import (
    ModelViewer,
    get_model_list,
    print_model_list,
    print_model_summary,
    print_model_details,
    print_all_models,
    print_models,
)

__all__ = [
    'ModelViewer',
    'get_model_list',
    'print_model_list',
    'print_model_summary',
    'print_model_details',
    'print_all_models',
    'print_models',
]


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
            viewer.print_model_info(cmd)
    else:
        viewer.print_summary_table()
