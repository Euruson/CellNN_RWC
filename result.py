import sys
import getopt
import importlib


def main(argv):
    methods = [
        "edge",
        "laplace",
        "half-toning",
        "inverse-halftoning",
        "erosion",
        "laplace-gaussian",
        "connectivity",
    ]
    try:
        opts, args = getopt.getopt(argv, "hm:", ["help", "method="])
    except getopt.GetoptError:
        print("Error: result.py -m <method>")
        print("   or: result.py --method=<method>")
        print("method = %s" % (", ".join(methods)))
        sys.exit(2)
    if opts == []:
        print("Error: result.py -m <method>")
        print("   or: result.py --method=<method>")
        print("method = %s" % (", ".join(methods)))
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("result.py -m <method>")
            print("or: result.py --method=<method>")
            print("method = %s" % (", ".join(methods)))
            sys.exit()
        elif opt in ("-m", "--method"):
            if arg in methods:
                mm = importlib.import_module(arg)
                mm.paper_result()
            else:
                print("Error: method = %s" % (", ".join(methods)))
                sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])
