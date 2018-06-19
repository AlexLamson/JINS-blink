import os
import re


def get_jins_openface_csv(path):
    result_jins = None
    result_openface = None

    jins_re = '^[0-9|A]*_[0-9|A]*\.csv$'
    of_re    = '^webcam.*\.csv$'

    for fname in os.listdir(path):
        print(fname)
        if re.match(jins_re, fname):
            if result_jins:
                print('WARNING: multiple candidates for jins file')
            result_jins = path + fname

        if re.match(of_re, fname):
            if result_openface:
                print('WARNING: multiple candidates for openface file')
            result_openface = path + fname

    return result_jins, result_openface


def guarantee_execution(function, fargs):
    attempting = True

    while attempting:
        try:
            function(*fargs)
            attempting = False
        except Exception as e:
            print(e)

            if input("Try again? Submit 'n' to abort: ") == 'n':
                attempting = False
