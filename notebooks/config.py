from pathlib import Path


def get_paths():
   
    import sys
    from pathlib import Path

    if Path('/kaggle/input').exists():
        WORK_DIR = Path('/kaggle/working')
        RAW_ROOT = Path('/kaggle/input/neurovision-raw')
    elif Path('/content').exists():
        WORK_DIR = Path('/content')
        RAW_ROOT = Path('/content/drive/MyDrive/NeuroVision/data/raw')
    else:
        HERE = Path.cwd()
        ROOT = HERE.parent if HERE.name == 'notebooks' else HERE
        WORK_DIR = ROOT
        RAW_ROOT = ROOT / 'data' / 'raw'

    return {
        'WORK_DIR'   : WORK_DIR,
        'RAW_ROOT'   : RAW_ROOT,
        'RAW_TRAIN'  : RAW_ROOT / 'Train',
        'RAW_VAL'    : RAW_ROOT / 'Val',
        'PREP_CLS'   : WORK_DIR / 'data' / 'prepared' / 'classification',
        'PREP_DET'   : WORK_DIR / 'data' / 'prepared' / 'detection',
        'REPORT_DIR' : WORK_DIR / 'outputs' / 'reports',
        'MODEL_CLS'  : WORK_DIR / 'models' / 'classification',
        'MODEL_DET'  : WORK_DIR / 'models' / 'detection',
        'OUT_DIR'    : WORK_DIR / 'outputs',
    }


if __name__ == '__main__':
    paths = get_paths()
    for name, path in paths.items():
        exists = 'OK' if path.exists() else 'MISSING'
        print(f'{name:<15}: {path}  [{exists}]')
