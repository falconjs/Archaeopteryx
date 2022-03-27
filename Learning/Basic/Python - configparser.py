import configparser

"""
configparser writer
"""
config = configparser.ConfigParser()
config['DEFAULT'] = {'ServerAliveInterval': '45',
                     'Compression': 'yes',
                     'CompressionLevel': '9'}

config['bitbucket.org'] = {}
config['bitbucket.org']['User'] = 'hg'

config['topsecret.server.com'] = {}
topsecret = config['topsecret.server.com']
topsecret['Port'] = '50022'  # mutates the parser
topsecret['ForwardX11'] = 'no'  # same here

config['DEFAULT']['ForwardX11'] = 'yes'

with open('example.ini', 'w') as configfile:
    config.write(configfile)

"""
configparser reader
"""
config = configparser.ConfigParser(
    delimiters=('=',':'),
    comment_prefixes=('#') # remove ;
)
config.sections()
# []

config.read('example.ini')
# ['example.ini']

config.read('./Learning/Basic/example.ini')
config.sections()
# ['bitbucket.org', 'topsecret.server.com']

'bitbucket.org' in config
# True

'bytebong.com' in config
# False

'user' in config
# False

config['bitbucket.org']['User']
# 'hg'

config['DEFAULT']['Compression']
# 'yes'

config['bitbucket.org']['ForwardX11']
# 'yes'

config['b1_00_cn_mcm_account']['load_args']
config['b1_00_cn_mcm_account']['sep']
config['b1_00_cn_mcm_account']['dtype']

import json
table = json.loads(config['b1_00_cn_mcm_account']['dtype'])

config['DEFAULT']['compressionlevel']
config.getint('DEFAULT', 'compressionlevel')

topsecret = config['topsecret.server.com']

'Port' in topsecret
# True

topsecret['ForwardX11']
# 'no'

topsecret['Port']
# '50022'

for key in config['bitbucket.org']:
    print(key)
# user
# compressionlevel
# serveraliveinterval
# compression
# forwardx11

# Read from Different config for the same variable
# Keep the last one
conf_paths = ["./conf/base", "./conf/local"]
conf_loader = configparser.ConfigParser()
file_list = [i + '/example.ini' for i in conf_paths]
conf_loader.read(file_list)
conf_loader['bitbucket.org']['user']
conf_loader.get('bitbucket.org', 'user')
