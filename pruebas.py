import urllib3
from urllib.parse import urlencode

import time

http = urllib3.PoolManager()


def current_time_millis():
    return int(round(time.time() * 1000))


def prueba_get_request(cantidad_requests):
    t_inicial = current_time_millis()
    for i in range(0, cantidad_requests):
        http.request('GET', 'http://localhost:8080/clasificar', fields={
            'mensaje': 'hola como andas'
        })

    t_final = current_time_millis() - t_inicial

    print('(cant_requests: %s, t_transcurrido: %s)' % (cantidad_requests, t_final))


def prueba_post_request(cantidad_requests, cantidad_epochs):
    t_inicial = current_time_millis()
    for i in range(0, cantidad_requests):
        http.request('POST', 'http://localhost:8080/reclasificar?' + urlencode({
            'mensaje': 'estas haciendo cualquier cosa',
            'conducta': '10',
            'epochs': str(cantidad_epochs)
        }))

    t_final = current_time_millis() - t_inicial

    print('(cant_requests: %s, cant_epochs: %s, t_transcurrido: %s)' % (cantidad_requests, cantidad_epochs, t_final))


prueba_get_request(100)
prueba_get_request(500)
prueba_get_request(1000)
prueba_post_request(100, 1)
prueba_post_request(100, 5)
prueba_post_request(500, 1)
prueba_post_request(500, 5)
prueba_post_request(1000, 1)
prueba_post_request(1000, 5)
