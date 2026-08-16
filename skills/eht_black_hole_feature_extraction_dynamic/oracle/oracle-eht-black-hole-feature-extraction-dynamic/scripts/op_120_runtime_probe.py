"""Cheap environment probe. Non-mutating. Run with the run-local python."""
import importlib


def main():
    report = {}
    for mod in ('numpy', 'scipy', 'torch', 'torchkbnufft'):
        try:
            m = importlib.import_module(mod)
            report[mod] = getattr(m, '__version__', 'unknown')
        except Exception as e:  # noqa: BLE001
            report[mod] = f'MISSING ({e})'
    try:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
    except Exception as e:  # noqa: BLE001
        report['cuda_available'] = f'unknown ({e})'
    for k, v in report.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
