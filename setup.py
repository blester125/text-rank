import ast
from typing import Optional
from setuptools import setup, find_packages


def get_version(file_name: str, version_name: str = "__version__") -> Optional[str]:
    with open(file_name) as f:
        tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if node.targets[0].id == version_name:
                    return node.value.s
    raise ValueError(f"Unable to find an assignment to the variable {version_name}")


class About(object):
    NAME = "text-rank"
    VERSION = get_version(f"text_rank/__init__.py")
    AUTHOR = "blester125"
    EMAIL = f"{AUTHOR}@gmail.com"
    URL = f"https://github.com/{AUTHOR}/{NAME}"
    DL_URL = f"{URL}/archive/{VERSION}.tar.gz"
    LICENSE = "MIT"
    DESCRIPTION = "Text Rank"


ext_modules = []


setup(
    name=About.NAME,
    version=About.VERSION,
    description=About.DESCRIPTION,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author=About.AUTHOR,
    author_email=About.EMAIL,
    url=About.URL,
    download_url=About.DL_URL,
    license=About.LICENSE,
    python_requires=">=3.6",
    packages=find_packages(),
    package_data={
        "text_rank": [
            "text_rank/data/automatic-summarization-sents.json",
            "text_rank/data/automatic-summarization-tokens.json",
            "text_rank/data/paper-example-keywords.json",
            "text_rank/data/paper-example-summarize.json",
        ],
    },
    include_package_data=True,
    install_requires=["numpy",],
    setup_requires=[],
    extras_require={"test": ["pytest"],},
    keywords=["NLP", "Summarization", "Keyword Extraction", "Text Rank", "Page Rank", "Graph"],
    entry_points={"console_scripts": ["text-rank-demo = text_rank.demo:main", "text-rank = text_rank.main:main",],},
    ext_modules=ext_modules,
    classifiers={
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    },
)
