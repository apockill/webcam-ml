# webcam-ml

![Example](./docs/zoom-screenshot.png)

# Dependencies
```
# Python
poetry install

# Linux
sudo apt-get install v4l2loopback-utils
```

# Usage
Install dependencies, then before running run the following command
with the desired fake webcam device:
```
sudo modprobe v4l2loopback video_nr=1 card_label="ML Webcam"
```

## Debugging
If you get the following error, 
[try recompiling v4l2loopback](https://askubuntu.com/questions/1263554/sudo-modprobe-v4l2loopback-modprobe-error-could-not-insert-v4l2loopback-bad)
```
modprobe: ERROR: could not insert 'v4l2loopback': Bad address
```
