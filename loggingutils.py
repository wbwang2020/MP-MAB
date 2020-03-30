# -*- coding: utf-8 -*-
"""
@author: Wenbo Wang

[Wang2020] Wenbo Wang, Amir Leshem, Dusit Niyato and Zhu Han, "Decentralized Learning for Channel 
Allocation inIoT Networks over Unlicensed Bandwidth as aContextual Multi-player Multi-armed Bandit Game"

License:
This program is licensed under the GPLv2 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

"""
This file implement the logging module as the wrapper of the standard logging API provided by python.
Please use the format, e.g., "info_logger().log_info("...")" to record the information of interest in a log file 
stored in the path "$PWD/results"
"""

__author__ = "Wenbo Wang"

import logging
import os
import functools

from datetime import datetime


def __singleton(class_):
    """
    Make <class_> a singleton class with only one single instance.
    Note that it cannot prevent instantiation in multiple processes
    """    
    @functools.wraps(class_)
    def wrapper_singleton(*args, **kwargs):
        if wrapper_singleton.instance is None:
#            print("wrapper_singleton.instance")
            wrapper_singleton.instance = class_(*args, **kwargs)
            
        return wrapper_singleton.instance
    
    wrapper_singleton.instance = None
    
    return wrapper_singleton

@__singleton
class info_logger(object):
    def __init__(self):                       
        log_file_name = 'log'
        # the logging module may be used by different process in the parallel mode
        # for each process we create a single log file
        process_id = os.getpid()
        
        now = datetime.now()
        current_date = now.strftime("(%Y-%m-%d-%H-%M-%S)")
        cwd = os.getcwd() # current directory    
        logFilePath = "{}\{}\{}-{}-{}.log".format(cwd, "results", log_file_name, process_id, current_date)
                 
        # get the instance of logger
        self.logger = logging.getLogger(log_file_name)        
        self.logger.setLevel(logging.DEBUG)

        #define the output format        
        logging_format = logging.Formatter("[%(threadName)s, %(levelname)s] %(message)s")
#        logging_format = logging.Formatter('%(name)s  %(asctime)s  %(levelname)-8s:%(message)s')
 
        # file handler        
        file_handler = logging.FileHandler(logFilePath, mode='w')
        file_handler.setFormatter(logging_format)
        file_handler.setLevel(logging.DEBUG)
 
        self.logger.addHandler(file_handler)
        
        print("logger created @ {}".format(logFilePath))
        self.log_info("logger created")

    # for different levels of messages, we can also call the logger member directly
    def log_info(self, msg):
        self.logger.info(msg)
 
    def log_debug(self, msg):
        self.logger.debug(msg)

    def log_error(self, msg):
        self.logger.error(msg)
    
if __name__ == '__main__':
    print("Warning: this script 'loggingutils.py' is NOT executable..")  # DEBUG
    exit(0)
else:
    # turn it on then we create one log file for each process before it is really needed
#    fileLogger = info_logger()
    pass