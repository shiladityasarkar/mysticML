import subprocess

def _auto_install_dependencies():
    try:
        import numpy
    except ImportError:
        print("numpy is not installed. Installing...")
        subprocess.call(['pip', 'install', 'numpy'])

    try:
        import pandas
    except ImportError:
        print("pandas is not installed. Installing...")
        subprocess.call(['pip', 'install', 'pandas'])

    try:
        import sklearn
    except ImportError:
        print("scikit-learn is not installed. Installing...")
        subprocess.call(['pip', 'install', 'scikit-learn'])

    try:
        import scipy
    except ImportError:
        print("scipy is not installed. Installing...")
        subprocess.call(['pip', 'install', 'scipy'])

    try:
        import math
    except ImportError:
        print("math is not installed. Installing...")
        subprocess.call(['pip', 'install', 'math'])

_auto_install_dependencies()
