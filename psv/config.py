"""Configuration & Logging Setttings

Import C to get configuration values
Import L to get a logger for this application

This is intended for reuse -- with the variable APPNAME set to a mioniker for the project. 

This will ensure that the user has folders:
-  ~/.config/
-  ~/{APPNAME}/logs/ 


"""

import os
import json
import logging, logging.config
from traceback import format_exc
from easydict import EasyDict

APPNAME='psv'
CONFIG_PATH = os.path.expanduser(f'~/.config/{APPNAME}/config.json')

# Make some config folders
os.makedirs(os.path.expanduser(f'~/.config/{APPNAME}'), exist_ok=True)
os.makedirs(os.path.expanduser(f'~/.{APPNAME}/logs'), exist_ok=True)

# The C object has all of our configuration settings. 
C = EasyDict()

C.LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': "[%(asctime)s] %(levelname)s [%(pathname)s:%(lineno)d]  %(process)d %(thread)d %(message)s"
        },
        'simple': {
            'format': "%(levelname)s [%(pathname)s:%(lineno)d] %(message)s"
        },
    },
    'filters': {
        # 'special': {
        #     '()': 'project.logging.SpecialFilter',
        #     'foo': 'bar',
        # }
    },
    'handlers': {
       'console':{
            'level':'INFO',
            'class':'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'verbose',
            'filename': os.path.expanduser(f'~/.{APPNAME}/logs/{APPNAME}.log'),
            'mode': 'a',
            'maxBytes': 10485760,
            'backupCount': 5,
        },
    },
    'loggers': {
        APPNAME: {
            'handlers': ['console', 'file'],
            'level': 'INFO',
 #           'filters': ['special']
        }
    }
}

logging.config.dictConfig(C.LOGGING)
L = logging.getLogger(APPNAME)



# TODO: We should put this in a different repo and download it on demand
C.DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../psvdata'))


# Replace our default config with data from a config file
def load(path = None):
    global L
    if path is None:
        path =  os.path.expanduser(CONFIG_PATH)

    try:
        with open(path) as f:
            cfg = json.load(f)
            C.update(cfg)

        # Update the logging config based on the new settings
        logging.config.dictConfig(C.LOGGING)
        L = logging.getLogger(APPNAME)

    except IOError as e:
        L.error(f"Unable to process config file {path}", exc_info=True)

# Save our settings -- in case they are changed programattically or missing on the system.
def save(path = None):
    try:
        if path is None:
         path =  os.path.expanduser(CONFIG_PATH)

        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(path, exist_ok = True)

        with open(path, 'w') as f:    
            json.dump(C, f, indent=True)
    except IOError as e:
        L.error(f"Unable to save config to {path}", exc_info=True)


# Whenever this is loaded, we try to get settings from the system and then save them back
if os.path.isfile(CONFIG_PATH):
    load(CONFIG_PATH)
    L.debug(f'Loaded config from {CONFIG_PATH}')
else:
    L.info(f"Creating configuration file in {CONFIG_PATH}")
    save(CONFIG_PATH)
