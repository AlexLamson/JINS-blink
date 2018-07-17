# JINS-blink
Detect blinks using the J!NS MEME glasses

### Recording Data
Data is recorded with the [JINS-MEME Data Logger](https://github.com/jins-meme/ES_R-DataLogger-for-Windows/releases) and OpenFace.
1. Connect the JINS data logger to the glasses, but do not start recording. Set the mode to'Full'. It is recommended to set the transmission speed to 100Hz.
2. Set OpenFace to record AUs. You may also set it to record other things, but AUs are all that is necessary.
3. Begin recording the webcam with OpenFace, then begin measuring with JINS. Once both are running, blink repeatedly at the webcam.
4. Begin to blink normally, making sure your eyes are visible to the webcam and the JINS glasses are getting a good signal.
5. Before stopping the recording blink repeatedly again. Stop measuring with JINS and then stop OpenFace.
6. Move the generated OpenFace and JINS files into a folder of your choosing, and modify the `path` variable in `alignment.py` and `merge_data.py`.
7. Run `alignment.py`. Zoom in as necessary and right click a point on the JINS EOG plot, then right click the corresponding point in time on the OpenFace plot. Close the window. A new window will open showing the end of the data. Again, right click on an JINS EOG point and then on the corresponding OpenFace point. Close the window.
8. Run `merge_data.py`.

### Training a Model 
After recording data, a model can be trained to look at the JINS EOG signal and determine if the user is blinking or not.
1. Set the `training_data_fname` in `train.py` to the `combined.csv` file produced by `merge_data.py`.
2. Run `train.py`

### Realtime Blink Evaluation
Once a model has been trained, blinks can be detected in realtime.
1. Open the JINS MEME Data Logger. Make sure 'external output socket' is turned on in settings. Connect to the glasses and begin. Set the mode to 'Full' and the transmission speed to the speed you chose when recording. Start measuring.
2. Run `live_classify.py`. It will ask for the IP address and port of the JINS data logger. You can also run live classify with arguments like so:
```sh
$ python live_classify.py ip_address port
```