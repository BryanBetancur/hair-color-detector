

# Hair Color Detector [pypi](https://pypi.org/project/hair-color-detector/)

Segmentator based from [nganngants/hair-dye-web-app](https://github.com/nganngants/hair-dye-web-app)

## Overview

This project allows you to get hair color of a person from an image. 

Basic Usage
-----------

```python
from hair_color_detector import HairColorDetector
hcd = HairColorDetector()
hcd.get_color('test.png',  save_result=True, n_clusters=10)
```