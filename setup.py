from setuptools import setup

setup(
    name='word-detector',
    version='1.0.0',
    description='Word Detector',
    author='Harald Scheidl',
    packages=['word_detector'],
    url="https://github.com/githubharald/WordDetector",
    install_requires=['numpy', 'scikit-learn', 'opencv-python'],
    python_requires='>=3.7'
)
