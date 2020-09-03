"""
Module to do basic timing of various events in code

create a Timings object, then call start and end at various points in the code
to record timing for the contained code blocks
"""


import time

class Timing(object):
    __totalTime = 0.0
    __startTime = None
    __count = -1 # Skip first frame to avoid dll load overhead
    __name = None

    def __init__(self, name):
        self.__name = name

    def start(self):
        if self.__startTime is not None:
            print ("Error - multiple start calls to " + self.__name)
        self.__startTime = time.time()

    def end(self):
        if self.__startTime is None:
            print ("Error - no start call to " + self.__name)
            return

        self.__count += 1
        if self.__count > 0:
            self.__totalTime += time.time() - self.__startTime
        self.__startTime = None

    def __str__(self):
        if self.__count <= 0:
            return self.__name + " : no events recorded"
        return self.__name + " : " + str(self.__count) + " events. Total time = " + str(self.__totalTime) + ", average time = " + str(self.__totalTime / self.__count)


class Timings(object):
    __timings = {}

    def __del__(self):
        print(str(self))

    def start(self, name):
        if name not in self.__timings:
            self.__timings[name] = Timing(name)
        self.__timings[name].start()

    def end(self, name):
        if name not in self.__timings:
            print("Error - end called before start for " + name)
            return
        self.__timings[name].end()

    def __str__(self):
        s = ""
        for k in self.__timings.keys():
            s += str(self.__timings[k])
            s += '\n'
        return s


