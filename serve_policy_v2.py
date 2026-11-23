#!/usr/bin/env python3
"""Loopback development endpoint for an action policy; never executes tools."""
import argparse, json
from http.server import BaseHTTPRequestHandler, HTTPServer


def build_server(model, adapter, representation='raw', host='127.0.0.1', port=8089):
    import agent_v2 as core
    if representation == 'facts':
        from agent_facts_v2 import LMPolicy
    else:
        LMPolicy = core.LMPolicy
    policy = LMPolicy(model, adapter, 'cpu', threads=2)
    required = {'area', 'user', 'artifacts', 'checks', 'documents', 'used',
                'missing_exports', 'check_fingerprints'}

    class Handler(BaseHTTPRequestHandler):
        def emit(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                self.emit(404, {'error': 'unknown_path'})
            else:
                self.emit(200, {'status': 'loaded', 'scope': 'local action selection only'})

        def do_POST(self):
            if self.path != '/choose':
                self.emit(404, {'error': 'unknown_path'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 1_000_000:
                    raise ValueError('invalid_body_size')
                payload = json.loads(self.rfile.read(length))
                state = payload['state']
                if not isinstance(state, dict) or set(state) != required:
                    raise ValueError('send only the complete public state; no fixture or gold')
                if len(json.dumps(state, ensure_ascii=False).encode()) > 12_000:
                    raise ValueError('public state exceeds this development endpoint size limit')
                if state['area'] not in {'Seek', 'mAIstro'} or not isinstance(state['user'], str):
                    raise ValueError('invalid area or user')
                for key in ['artifacts', 'checks', 'documents', 'check_fingerprints']:
                    if not isinstance(state[key], dict):
                        raise ValueError('invalid public object ' + key)
                for key in ['used', 'missing_exports']:
                    if not isinstance(state[key], list):
                        raise ValueError('invalid public list ' + key)
                letter, usage = policy.choose(state, 'standard')
                action, argument, diagnosis = core.ACTIONS[letter]
                self.emit(200, {'proposal_only': True, 'requires_controller_validation': True, 'letter': letter, 'action': action, 'argument': argument,
                                'diagnosis': diagnosis, 'usage': usage,
                                'scope': 'model selected an action; no tool or platform call was executed'})
            except (ValueError, KeyError, TypeError) as error:
                self.emit(400, {'error': str(error)})
            except RuntimeError:
                self.emit(500, {'error': 'inference_failure; no diagnostic conclusion was produced'})

        def log_message(self, *_):
            pass

    return HTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--representation', choices=['raw', 'facts'], default='raw')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8089)
    args = parser.parse_args()
    server = build_server(args.model, args.adapter, args.representation, args.host, args.port)
    print(json.dumps({'address': server.server_address, 'scope': 'development only; no tools executed'}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
