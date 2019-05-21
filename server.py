import paho.mqtt.client as mqtt
import awesome
import numpy as np
import tensorflow as tf
from tensorflow import keras
from python_speech_features import mfcc

sig = []
model = keras.models.load_model('my_model.h5')
# Define event callbacks


def on_connect(client, userdata, flags, rc):
    print("rc: " + str(rc))


def on_message(client, obj, msg):
    global sig
    global model
    if len(sig) == 0:
        print("strart")
    # print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
    for i in msg.payload.split():
        sig = sig + [[int(i)]*2]
    print(len(sig))
    if len(sig) > 17000:
        # print(sig[0])
        new_rate = 11025
        mfcc_feat = mfcc(np.array(sig), new_rate, 200 / new_rate, 80 / new_rate)

        print(len(mfcc_feat))
        a = mfcc_feat[0:400]
        a = a.ravel()
        a = a.tolist()
        if 5200 > len(a):
            l = len(a)
            a = a + [0] * (5200 - l)
        sig = []

        predictions = model.predict_classes(np.array([a]))

        print(predictions[0])


def on_publish(client, obj, mid):
    print("mid: " + str(mid))


def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_log(client, obj, level, string):
    print(string)


mqttc = mqtt.Client()
# Assign event callbacks
mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.on_publish = on_publish
mqttc.on_subscribe = on_subscribe

# Uncomment to enable debug messages
# mqttc.on_log = on_log


topic = "LED1"
# Connect
mqttc.username_pw_set("rjmwwxdb", "uNGUnTiZBV4c")
mqttc.connect("m16.cloudmqtt.com", 10924)

mqttc.subscribe(topic, 0)

# Publish a message
mqttc.publish(topic, "my message")

# Continue the network loop, exit when an error occurs
rc = 0
while rc == 0:
    rc = mqttc.loop()
print("rc: " + str(rc))
