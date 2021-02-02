import os
from jupyter_server.auth import passwd

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = int(os.getenv('PORT', 8888))
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_root = True
c.ServerApp.iopub_data_rate_limit = 10000000000
if 'PASSWORD' in os.environ:
  password = os.environ['PASSWORD']
  if password:
    c.ServerApp.password  = passwd(password)
  else:
    c.ServerApp.password = ''
    c.ServerApp.token = ''
  del os.environ['PASSWORD']
