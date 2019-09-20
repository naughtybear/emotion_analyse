# -*- coding: utf-8 -*-
import subprocess
from subprocess import Popen
import os


class Communicate():
    def __init__(self, CMD):
        self.CMD = CMD

    def create_subprocess(self):

        if self._is_file_exist(self.CMD):

            try:
                sp = Popen(self.CMD, shell=True, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE)
                # print(sp)
            except Exception as E:
                print(E)
                return None
            else:
                return sp

        else:
            print('path not exist!')
            return None

    def _is_file_exist(self, CMD):

        file_path = CMD.split(' ')[1]
        return os.path.isfile(file_path)

    def check_subprocess_status(self, sp):

        return sp.poll()

    def communicate_model(self, sp, input_data):

        status = self.check_subprocess_status(sp)

        if status is None:
            print('input data', input_data.encode())
            input_data = ((input_data + '\n').encode())
            sp.stdin.write(input_data)
            sp.stdin.flush()

            output_data = sp.stdout.readline()
            print(output_data.decode('utf8'))
            while True:
                tmp_data = sp.stdout.readline()
                #print(tmp_data.decode('utf8'))
                if tmp_data.decode('utf8') == 'end\n':
                    break
                output_data = output_data + tmp_data
                #print('yaya')
            print(output_data)
            if output_data:
                return output_data.decode('utf8')

            else:
                print('data error')
                print(output_data)
                return None

        else:
            print('subprocess status ', status)
            print('Subprocess End!!!')
            return None

    def kill_subprocess(self, sp):
        sp.terminate()
        sp.wait()


if __name__ == '__main__':

    to_python = Communicate('python model.py')
    subp = to_python.create_subprocess()

    while True:

        if subp is not None:
            data = input('input')
            if data == '0':
                break

            subp_status = to_python.check_subprocess_status(subp)

            if subp_status is None:
                print('subprocess not end !!')
                msg = to_python.communicate_model(subp, data)

                if msg is None:
                    print('Output data Error!')

                else:
                    print('Output Data is ', msg)

            else:

                break

        else:
            print('subprocess create error')

            break

    if to_python.check_subprocess_status(subp) is None:
        to_python.kill_subprocess(subp)
