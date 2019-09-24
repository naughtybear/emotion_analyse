# -*- coding: utf-8 -*-
import socket
from to_model import Communicate

#==== Create to_python communication ====
to_python = Communicate('python ../test_model.py')
subprocess = to_python.create_subprocess()

if subprocess is not None:
    print('Subprocess create successfully')

    HOST = '127.0.0.1'
    PORT = 8001

    #===== Create socket server =====

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)


    print('Server start at {} {}', format(HOST, str(PORT)))
    print('Wait for connect...')

else:
    print('No create subprocess!')

while True:

    #==== Connect successfully ====
    connect, address = server.accept()

    print('Connect by ', str(address))

    # Receive Data from node server
    data = connect.recv(1024).decode('utf-8')

    if not data:
        pass

    else:
        if subprocess is not None:   # subprocess create successfully.

            subprocess_status = to_python.check_subprocess_status(
                subprocess)

            if subprocess_status is None:   # subprocess doesn't finish
                print("Receive: ", data)

                # Send Data to mode
                msg = to_python.communicate_model(subprocess, data)

                if msg is None:
                    msg = 'Output data Error!'
                    connect.send(msg.encode('utf-8'))

                else:
                    # Send result to node server
                    connect.send(msg.encode('utf-8'))

            else:
                msg = 'subprocess already finished'
                connect.send(msg.encode('utf-8'))
                break

        else:
            msg = 'subprocess create error'
            connect.send(msg.encode('utf-8'))
            break

    # connect.close()

if to_python.check_subprocess_status(subprocess) is None:
    to_python.kill_subprocess(subprocess)


server.close()
