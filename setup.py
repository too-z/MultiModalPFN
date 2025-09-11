import re
from pathlib import Path
from setuptools import setup, find_packages


DESCRIPTION = ""
HERE = Path(__file__).parent
try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

def get_package_version() -> str:
    with open(HERE / "mmpfn/__init__.py") as f:
        result = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if result:
            return result.group(1)
    raise RuntimeError("Can't get package version")

setup(
	name="mmpfn",
	version=get_package_version(),
 	description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wall Kim",
    url="https://github.com/too-z/MultiModalPFN",
    license="Apache",
    license_files=("LICENSE",),
	packages=find_packages(),
	install_requires=[],
)
