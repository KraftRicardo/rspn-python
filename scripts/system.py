from platform import python_version
import sys
import platform


if __name__ == '__main__':
    print("Current Python Version-", python_version())
    print("User Current Version:-", sys.version)
    print("VERSION", platform.python_implementation())