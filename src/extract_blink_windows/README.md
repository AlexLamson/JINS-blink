How to use set up
=================

There are 4 paths on lines 33, 34, 40, 41. Change those paths to match what you have on your computer. The script will check those paths for files in those paths.

Blink frames are saved to `JINS-blink\res\blink_frames`. It would be a good idea to copy the blink files you already have into there.

Run the script with:
`python hand_label_blinks_tool.py`

Changing lines 64-69 let's you explicitly skip subjects/labels.



Adding a new subject
====================

Create a new folder in the same directory as all the other subjects, formatted similarly (ex. "301"). Save your jins data in the same subfolders as for the other subjects (ex. "WINCE/106/label0/jins").

Change the `subject_numbers` variable on line 26 to include this number in order to tell the script to look for this folder.

Title the jins data in the same way as the other files, e.g. "109_label2.csv".

If there's an error
-------------------
The script should tell you if it expects to find a file and does/doesn't see it (and if it can't find it, it will skip to the next file to be labeled).


When labeling
=============

Try not to drag the right click. This will zoom the graph.

In the script output in the console, it will tell you what subject and label it is checking for, and if there is an error of any kind.


Additional thoughts
===================

107 label 0 - blinks are quite close together, may want to be careful with this

111 label 0 - blinks are quite close together, may want to be careful with this

may want to stratify samples by subject - some subjects have many blinks, others have only a few
