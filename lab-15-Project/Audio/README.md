Create a Docker image on Raspberry Pi that runs an object detection model using LiteRT.
The hardware setup consists of a Webcam connected to the Raspberry Pi board using a USB Hub.
Use the ideas on AI model output tensor, LiteRT runtime, MQTT and Docker are discussed in 
the course. 

**Step 1:**

Create a **src** folder containing the inference code, sound classifier TFLite model file inside a directory
called **RPi-YAMNET**. 

Outside the src folder, create a requirements.txt file with the following Python packages 

- numpy
- ai-edge-litert
- pyalsaaudio
- paho-mqtt==1.6.1

Create a Docker file to use debian:bookworm-slim OS and Python 3.10-slim.  The slim versions are to make 
the image lightweight. 

**Step 2:**

Install Docker on Raspberry Pi using this command

$ curl -sSL https://get.docker.com | sh

Add your user to the `docker` group

$ sudo usermod -aG docker $USER

**Step 3:**

Create Docker image using build.

- Build image locally on the Raspberry Pi -3B and and expected to take about 3-4 minutes.
- Change to the RPi-YAMNET directory and issue the build command.

```bash
$sudo docker build -t YAMNET-mqtt .
```

**Step 4:**

 Check if the image is successfully created

```bash
$docker images

REPOSITORY       TAG       IMAGE ID       CREATED          SIZE
YAMNET-mqtt     latest    c73cb468959d   7 minutes ago    405MB

```

**Step 5:**
Run the Docker container created in step 3 in the background.  USB Camera is connected on the /dev/video0

```bash
$sudo docker run --privileged -v hw:3,0:hw3,0 YAMNET-mqtt &
```

**Step 6:**
Observe the output from the application. 

- The output is a MQTT payload consisting of timestamp and label from class of interest. 
- Sample outputs with audio detecting silence

```bash
[3] 37984
pi@raspberrypi:~/RPi-YAMNET$ INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
TFLite Safety Monitor Active...
[!!!] SILENCE DETECTED (Score: 0.59)
[!!!] SILENCE DETECTED (Score: 0.85)
[!!!] SILENCE DETECTED (Score: 0.80)



```

**Step 7:**

- Change the MQTT topic name in the application code (yamnet-street.py) and subscribe to the topic on another computer to view the output from the acoustic model.
- Log the data of detections with timestamp to a SQLite database. 
- Use the SQLite database content to create a dashboard using Streamlit

