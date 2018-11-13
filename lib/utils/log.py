import numpy as np
import sys
BEGIN_DEBUG=True


def log(string,filename="log.txt"):
    np.set_printoptions(threshold='nan')
    if BEGIN_DEBUG == True:
        output = sys.stdout
        outputfile = open("E:\\"+filename, "a")
        sys.stdout = outputfile
        print("------------------------");
        print(string)
        print("------------------------")
        sys.stdout=output

