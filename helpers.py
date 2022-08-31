'''
    File containing helper functions for general use
'''
import json, os, sys

def OpenJSON(filename):
    '''Open JSON file as dictionary
    params:
	filename (str) = JSON file to open
    returns:
	dict: Python dictionary of JSON
    '''
    if sys.version_info.major == 3:
        return json.load(open(filename,'r'))
    else:
        return json.load(open(filename,'r'), object_hook=AsciiEncodeDict)

def AsciiEncodeDict(data):
    '''Encodes dict to ASCII for python 2.7'''
    ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
    return dict(map(ascii_encode, pair) for pair in data.items())

