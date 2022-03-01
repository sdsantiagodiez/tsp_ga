from setuptools import find_packages, setup


# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()


# get the dependencies and installs
with open("pytsp/requirements.txt", encoding="utf-8") as f:
    # Make sure we strip all comments and options (e.g "--extra-index-url")
    # that arise from a modified pip.conf file that configure global options
    # when running kedro build-reqs
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

setup(
    name="pytsp",
    version="0.1.0",
    author="Santiago Diez",
    author_email="sdsantiagodiez@gmail.com",
    description="Genetic algorithm implementation of \n"
    "the Travelling Salesman Problem",
    url="https://github.com/sdsantiagodiez/tsp_ga",
    project_urls={
        "Bug Tracker": "https://github.com/sdsantiagodiez/tsp_ga/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requires,
    # package_dir={"pytsp": ""},
    packages=find_packages(exclude=["tests", "app"]),
    python_requires=">=3.8",
)
