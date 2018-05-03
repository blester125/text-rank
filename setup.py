from setuptools import setup, find_packages

name = "text_rank"
version = "0.1.2"

setup(
    name=name,
    version=version,
    description="Text Rank in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Brian Lester",
    author_email="blester125@gmail.com",
    url=f"https://github.com/blester125/{name}",
    download_url=f"https://github.com/blester125/{name}/archive/{version}.tar.gz",
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
