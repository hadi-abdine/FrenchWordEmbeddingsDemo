#!/usr/bin/python3
import sys
import os
import cherrypy
from cherrypy.lib.static import serve_file
from explorer import Model
from run import app as myflask
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--ip', type=str, default='127.0.0.1')
args = parse.parse_args()
STATIC_DIR = os.path.dirname(os.path.realpath(__file__)) + '/static'
CACHE = {}


class ApiController(object):

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.config(**{'tools.encode.encoding': 'utf-8'})
    def explore(self,
                query='paris',
                limit='100',
                enable_clustering='',
                num_clusters='4'):
        cache_key = '-'.join([query, limit, enable_clustering, num_clusters])
        result = CACHE.get(cache_key, {})
        if len(result) > 0:
            return {'result': CACHE[cache_key], 'cached': False}
        try:
            exploration = self.model.explore(query, limit=int(limit))
            exploration.reduce()
            if len(enable_clustering):
                if (len(num_clusters)):
                    num_clusters = int(num_clusters)
                exploration.cluster(num_clusters=num_clusters)
            result = exploration.serialize()
            CACHE[cache_key] = result
            return {'result': result, 'cached': False}
        except KeyError:
            return {'error': {'message': 'No vector found for ' + query}}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.config(**{'tools.encode.encoding': 'utf-8'})
    def compare(self, **kw):
        limit = kw.pop('limit', '100')
        queries = kw.pop('queries[]', [])
        try:
            result = self.model.compare(queries, limit=int(limit))
            return {'result': result}
        except KeyError:
            return {'error':
                    {'message': 'No vector found for {}'.format(queries)}}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.config(**{'tools.encode.encoding': 'utf-8'})
    def autocomplete(self, query=None, limit='20'):
        result = self.model.autocomplete(query, limit=int(limit))
        return {'result': result}


class AppController(object):

    @cherrypy.expose
    def index(self, **kw):
        return serve_file(STATIC_DIR + '/indexss.html', "text/html")

# def secureheaders():
#     headers = cherrypy.response.headers
#     headers['X-Frame-Options'] = 'DENY'
#     headers['X-XSS-Protection'] = '1; mode=block'
#     headers['Content-Security-Policy'] = "default-src='self'"

if __name__ == '__main__':
    
    cherrypy.config.update({
                            'server.socket_host': args.ip,
                            'server.socket_port': 8080,
                            'engine.autoreload.on': True,
                            })

    api_controller = ApiController()
    api_controller.model = Model('../word2vec/dascim2.bin')
    app_controller = AppController()
    dispatch = cherrypy.dispatch.RoutesDispatcher()
    dispatch.connect('api', '/api/:action', controller=api_controller)
    dispatch.connect('app', '/:id', controller=app_controller, action="index")
    dispatch.connect('app', '/', controller=app_controller, action="index")

    cherrypy.tree.graft(myflask.wsgi_app, '/FrenchLinguisticResources')


    app = cherrypy.tree.mount(None, config={
        '/': {
            'request.dispatch': dispatch,
            'tools.staticdir.on': True,
            'tools.staticdir.dir': STATIC_DIR
        }

    })
    
    cherrypy.engine.start()
    
    cherrypy.engine.block()

    # })
    # cherrypy.quickstart(app2, '/explore')
