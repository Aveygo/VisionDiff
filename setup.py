from setuptools import setup

setup(
    name='VisionDiff',
    version='0.1.0',    
    description='Using the [Differential Transformer](https://arxiv.org/abs/2410.05258) in a vision-friendly way, similar to [VisionMamba](https://github.com/kyegomez/VisionMamba).',
    url='https://github.com/Aveygo/VisionDiff',
    author='aveygo',
    author_email='aveygo.au@gmail.com',
    license='GPLv3',
    packages=['VisionDiff'],
    install_requires=[
        'torch',
        'numpy',
        'einops',                     
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Other/Nonlisted Topic'
    ],
)
