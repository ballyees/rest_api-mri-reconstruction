from sanic import Sanic
from sanic_cors import CORS
from sanic.response import json
from blueprints.fast import bp_fast
from blueprints.slow import bp_slow
app = Sanic(__name__)
app.blueprint(bp_fast)
app.blueprint(bp_slow)
CORS(app)

# define model here
@app.route('/')
async def test(request):
    return json({})

# host = '127.0.0.1'
# host = '0.0.0.0'
if __name__ == '__main__':
    try:
        app.run(auto_reload=False)
    except KeyboardInterrupt:
        exit(1)
        # cmd command for kill all python process
        # "taskkill /f /im "python.exe" /t"