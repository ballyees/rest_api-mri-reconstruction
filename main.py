from sanic import Sanic
from sanic_cors import CORS
from sanic.response import json
from blueprints.fast import bp_fast
from blueprints.slow import bp_slow

from helper import *

app = Sanic(__name__)
app.blueprint(bp_fast)
app.blueprint(bp_slow)
CORS(app)

# define model here
@app.route('/')
async def test(request):
    return json({})

host = '0.0.0.0'
port = 8000
if __name__ == '__main__':
    try:
        app.run(host=host, port=port, auto_reload=False)
    except KeyboardInterrupt:
        exit(1)
        # cmd command for kill all python process
        # "taskkill /f /im "python.exe" /t"