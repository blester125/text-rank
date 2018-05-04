from setuptools import setup, find_packages

class About(object):
    NAME='text_rank'
    VERSION='0.2.0'
    AUTHOR='blester125'
    EMAIL=f'{AUTHOR}@gmail.com'
    URL=f'https://github.com/{AUTHOR}/{NAME}'
    DL_URL=f'{URL}/archive/{VERSION}.tar.gz'
    LICENSE='MIT',

setup(
    name=About.NAME,
    version=About.VERSION,
    description='Text Rank in Python',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author=About.AUTHOR,
    author_email=About.EMAIL,
    url=About.URL,
    download_url=About.DL_URL,
    license=About.LICENSE,
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
    keywords=['NLP'],
)
