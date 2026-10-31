# -*- coding: utf-8 -*-
import os
import argparse
import logging

from label_studio_ml.api import init_app
from _sam_backend import SAMBackend

app = init_app(
    model_class=SAMBackend,
    model_dir=os.environ.get('MODEL_DIR', os.path.dirname(__file__))
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label Studio SAM ML Backend')
    parser.add_argument('-p', '--port', dest='port', type=int, default=9090, help='Server port')
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0', help='Server host')
    args, _ = parser.parse_known_args()
    app.run(host=args.host, port=args.port)
