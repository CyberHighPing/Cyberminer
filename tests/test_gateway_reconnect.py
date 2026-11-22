"""Transport regression on a real private Unix socket; no wallet, GPU or network node."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socketserver
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway_miner
import host_miner

@contextmanager
def server_for(respond):
    seen=[]
    class Peer(socketserver.StreamRequestHandler):
        def handle(self):
            while line:=self.rfile.readline():
                message=json.loads(line)
                seen.append(message["method"])
                result=respond(message,seen)
                if result is None:
                    return
                self.wfile.write(json.dumps({"jsonrpc":"2.0","id":message["id"],**result}).encode()+b"\n")
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)
        with socketserver.UnixStreamServer(str(p/"pool.sock"),Peer) as pool:
            t=threading.Thread(target=pool.serve_forever,kwargs={"poll_interval":.015},daemon=True)
            t.start()
            try:
                yield p,seen
            finally:
                pool.shutdown()
                t.join(timeout=2)
                assert not t.is_alive()

def use(path, action):
    args=types.SimpleNamespace(socket=path/"pool.sock",work_dir=path/"receipts",rpc_timeout=1,native_dir=None,
                               tile="16x32",m=131072,n=131072,k=16384)
    mock_api=types.SimpleNamespace(verify_plain_proof_for_cert_version=True,PlainProofV4=True)
    with patch.object(host_miner,"import_verifier",return_value=mock_api),patch.object(gateway_miner,"run",side_effect=action):
        return gateway_miner.connect(args)

class ReconnectTest(unittest.TestCase):
    def test_read_disconnect_recovers_on_fresh_stream(self):
        def respond(msg,seen):
            return None if len(seen)==1 else {"result":{"fresh":True}}
        with server_for(respond) as (d,seen):
            value=use(d,lambda args,rpc,miner:rpc.call("getMiningInfo",{}))
            self.assertEqual(value,{"fresh":True})
            self.assertEqual(seen,["getMiningInfo","getMiningInfo"])

    def test_no_replay_after_submit_ack_lost(self):
        def action(args,rpc,miner):
            rpc.call("getMiningInfo",{})
            return rpc.call("submitPlainProof",{"plain_proof":"transport test"})
        with server_for(lambda msg,seen:{"result":1} if len(seen)==1 else None) as (d,seen):
            with self.assertRaises(ConnectionError):
                use(d,action)
            self.assertEqual(seen,["getMiningInfo","submitPlainProof"])

    def test_valid_application_rejection_is_not_retried(self):
        with server_for(lambda msg,seen:{"error":{"code":-32001,"message":"paused"}}) as (d,seen):
            with self.assertRaises(gateway_miner.RpcError):
                use(d,lambda args,rpc,miner:rpc.call("getMiningInfo",{}))
            self.assertEqual(seen,["getMiningInfo"])

    def test_persistent_disconnect_is_bounded(self):
        with server_for(lambda msg,seen:None) as (d,seen):
            with self.assertRaises(ConnectionError):
                use(d,lambda args,rpc,miner:rpc.call("getMiningInfo",{}))
            self.assertEqual(seen,["getMiningInfo","getMiningInfo"])

if __name__=="__main__":
    unittest.main()
