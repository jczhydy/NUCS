import sys

class StdRedirect:
    def __init__(self, filename):
        self.stream = sys.stdout
        self.file = open(filename,'w')

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        pass

    def __del__(self):
        self.file.close()