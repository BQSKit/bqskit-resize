from setuptools import setup

setup(
        name='bqskit-resize',
        version = '1.0',
        url = 'https://github.com/BQSKit/bqskit-resize',
        author = 'Siyuan Niu',
        author_email = 'siyuanniu@lbl.gov',
        description = 'Quantum Circuit Resizing Algorithm',
        install_requires=[
            'numpy',
            'bqskit @ git+https://github.com/peachnuts/bqskit.git@add_reset'
            ],
)

