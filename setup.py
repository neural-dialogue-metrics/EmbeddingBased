from setuptools import setup

setup(
    name='EmbeddingBased',
    version='0.1',
    description='Embedding-based metrics for machine translation',
    url='https://github.com/neural-dialogue-metrics/EmbeddingBased.git',
    author='cgsdfc',
    author_email='cgsdfc@126.com',
    keywords=[
        'NL', 'CL', 'MT',
        'natural language processing',
        'computational linguistics',
        'machine translation',
    ],
    scripts=['embedding_metrics.py', 'make_testdata.py'],
    packages=[
        'embedding_based',
        'embedding_based.tests',
    ],
    package_data={
        'embedding_based.tests': ['data/*'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL-v3',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic'
    ],
    license='LICENCE.txt',
    long_description=open('README.md').read(),
    install_requires=['gensim', 'numpy'],
)
