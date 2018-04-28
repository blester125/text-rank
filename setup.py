from setuptools import setup, find_packages

version = "0.1.1"

setup(
    name="text_rank",
    version=version,
    description="Text Rank with Cython",
    author="Brian Lester",
    author_email="blester125@gmail.com",
    url="https://github.com/blester125/text_rank",
    download_url=f"https://github.com/blester125/text_rank/archive/{version}.tar.gz",
    license="MIT",
    packages=find_packages(),
    package_data={
        'text_rank': [
            'text_rank/data/Automatic_Summarization-sents.json',
            'text_rank/data/Automatic_Summarization-tokens.json',
        ],
    },
    include_package_data=True,
    install_requires=[
        'numpy',
        'tqdm',
    ],
    keywords=['NLP']
)
