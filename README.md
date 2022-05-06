<img src="./img/infineon_logo.png" alt="Infineon Logo" height="50"/>
<img src="./img/xensiv_logo.png" alt="XENSIV Logo" height="50"/>

# Vital Sensing with Radar and ML (Hackathon)

[![Gitter](https://badges.gitter.im/ifx-eestec-hack/community.svg)](https://gitter.im/ifx-eestec-hack/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Getting Started

**TL;DR:** This quick guide explains how to aquire data from the Infineon radar board with Python.

Please start by installing the most recent version of [Python](https://www.python.org/) and [Pip](https://pypi.org/project/pip/). We also recommend to install [Jupyter](https://jupyter.org/) for running the examples. Afterwards you can clone this git repo:
```
git clone https://github.com/Infineon/hackathon
cd hackathon
```

Now you can install the required Python package for the data acquisition with Pip:
```
pip install ifxdaq-3.0.1-py3-none-xxx.whl
```

:information_source: Make sure to replace _xxx_ by the correct filename according to your operating system and system architecture (see *.whl files in the repo for possible options).

**Now your system is ready to aquire radar data!**

Since some processing steps are recommended for the radar data you should install the following dependencies:

```
pip install numpy
pip install lsq-ellipse circle_fit
```

**For further information please check the ifxdaq documentation.** To access to documentation navigate in the cloned repository to the folder `docs` and open the file `index.html` with a web browser.

## Hackathon Material
* [Challenge Introduction Slides](./challenge_introduction.pdf)

## Documentation Links
* [Infineon Radar Development Kit](https://softwaretools.infineon.com/tools/com.ifx.tb.tool.ifxradarsdk)
* [DEMO BGT60TR13C Radar Demo Board](https://www.infineon.com/cms/en/product/evaluation-boards/demo-bgt60tr13c/)
* [Infineon BGT60TR13C Radar Chip](https://www.infineon.com/cms/en/product/sensor/radar-sensors/radar-sensors-for-iot/60ghz-radar/bgt60tr13c/)

## Infineon Team

<img src="./img/anja.jfif" alt="Anja" height="150"/>

**Anja** (Application & Product Marketing for Radar Sensing)

<img src="./img/souvik.jfif" alt="Souvik" height="150"/>

**Souvik** (Senior Staff Machine Learning Engineer)

<img src="./img/sarfaraz.jfif" alt="Sarfaraz" height="150"/>

**Sarfaraz** (Software Developer for Signal Processing)

<img src="./img/julian.jfif" alt="Julian" height="150"/>

**Julian** (Senior Embedded Systems Engineer)

### How to reach us?
[![Gitter](https://badges.gitter.im/ifx-eestec-hack/community.svg)](https://gitter.im/ifx-eestec-hack/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) or talk to Julian :)