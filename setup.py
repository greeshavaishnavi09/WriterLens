import setuptools

with open("README.md", "r", encoding = "utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Handwriting_Verification_Using_Siamese_Neural_Networks"
AUTHOR_USER_NAME = "Greesha Vaishnavi"
SRC_REPO = "Handwriting_Verification_Using_Siamese_Neural_Networks"
AUTHOR_EMAIL = "greeshavaishnavi@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "A small python package for recommending books",
    long_description = long_description,
    long_description_content = "text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Summariser": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where = "src")
)