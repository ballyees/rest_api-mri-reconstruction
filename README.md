# rest_api-mri-reconstruction
## Dependencies and Installation
```bash
pip install -r reqirements.txt
```
## download models
URL: https://drive.google.com/drive/folders/1FTrf8QZLzghZF1fY3zQZdtHRALiDbqKl?usp=sharing
Move models to models folder
## run
```bash
python run main.py
```
model run on http://localhost:8000

## How to use

send image with json data `keyword` "file" and encoding with base64 encoding see on `test.py` line 13 -> 21 to http://localhost:8000/{fast,slow}/{3x,5x,8x}
and get response image base64 encoding on data with `keyword` "response"
