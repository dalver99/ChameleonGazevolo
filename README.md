### 1. Git clone

### 2. pip install -r requirements.txt

models are
from https://github.com/wildchlamydia/mivolo?tab=readme-ov-file
mivolo_imbd.pth.tar

from https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view
yolov8x_person_face.pt

download models, locate on
/models/(model1)
/models/(model2)

Looks something like
![https://i.imgur.com/ZlGHKvV.png](https://i.imgur.com/ZlGHKvV.png)

Then when image pops, click left down corner and right top corner in order (twice)
You will get output image link on console.

Thx.

### Process for video > DB

0. add \*.mp4 to your gitignore for personal git usage..
1. Get mp4 file on root
2. Set a unique frame name, then run 'splitandsavezip.py' # Splits the timelapse video, and saves as images, and saves a ZIP file for faster access
3. run 'uploadoriginalimages.py' # Uploads all the images on DB.
4. Clear 'output/' folder before next step for DB update
5. run 'gazevolofromzip.py' # Goes through all the images with local GPU, Then saves them on /output. This also uploads gaze data to DB, so 6 MUST follow right after to avoid data tangling.
6. run 'uploadmodeloutputimages.py' #Goes through all model-throughputs, uploads on SQL.

Thank you!
