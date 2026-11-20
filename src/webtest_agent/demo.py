"""Synthetic smoke fixture, not a course benchmark or autonomous policy."""
from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


DEFAULT_DEMO_PORT = 18080
UNIT_PRICE_CENTS = 2500
DEMO_SCENARIO = (
    "Add Notebook (unit price CNY 25.00) to the cart. Change Quantity to 2 and click Update cart. "
    "Verify that the cart contains Notebook, the saved cart quantity is 2, and the cart total is CNY 50.00."
)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Paper Shop · Local Demo</title>
<style>
*{box-sizing:border-box}body{font:17px/1.6 system-ui,sans-serif;margin:0;background:#f4f3ed;color:#20332c}
.page{max-width:1000px;margin:auto;padding:36px 24px}header{display:flex;justify-content:space-between;align-items:center;gap:20px;border-bottom:1px solid #d6ddd5;padding-bottom:24px}
.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#607769}h1{font-size:32px;line-height:1.2;margin:8px 0}h2{font-size:24px;margin:0 0 16px}p{margin:10px 0}.muted{color:#637368}
main{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:28px}.panel{padding:28px;background:#fff;border:1px solid #dce2da;border-radius:16px}
.cover{height:180px;display:grid;place-items:center;background:#eef0e5;border-radius:10px;margin-bottom:24px}
.book{width:110px;height:145px;background:#315b49;border-left:9px solid #264536;box-shadow:7px 7px 0 #c5cdbd;border-radius:3px;color:#f7f4df;display:grid;place-items:center;font:20px Georgia,serif;letter-spacing:2px}
button,input{font:inherit;border-radius:8px;padding:10px 14px}button{border:1px solid #315b49;background:#315b49;color:white;cursor:pointer}button.secondary{background:transparent;color:#315b49}button:disabled{opacity:.5;cursor:wait}
input{width:90px;border:1px solid #b6c4ba}form{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin:24px 0}label{display:grid;gap:6px}button:focus-visible,input:focus-visible{outline:3px solid #d39a3c;outline-offset:3px}
#cart-total{border-top:1px solid #dce2da;padding-top:16px;font-size:22px;font-weight:650}#status{padding:14px 0;min-height:52px}footer{font-size:13px;color:#637368}[hidden]{display:none!important}
@media(max-width:680px){main{grid-template-columns:1fr}header{align-items:start}h1{font-size:26px}.panel{padding:22px}}
</style></head>
<body><div class="page">
<header><div><div class="eyebrow">Local shopping demo</div><h1>The Paper Shop</h1><p class="muted">A small cart. A clear result.</p></div><button id="reset" class="secondary" disabled>Reset cart</button></header>
<main><section class="panel" aria-labelledby="product-title"><div class="cover" aria-hidden="true"><div class="book">NOTES</div></div>
<h2 id="product-title">Notebook</h2><p class="muted">A simple notebook for everyday ideas.</p><p>Unit price: CNY 25.00</p><button id="add" disabled>Add to cart</button></section>
<section class="panel" aria-labelledby="cart-title"><h2 id="cart-title">Shopping cart</h2><p id="cart-empty" class="muted">Your cart is empty.</p>
<div id="cart-details" hidden><p id="cart-item"></p>
<form id="quantity-form"><label for="quantity">Quantity<input id="quantity" type="number" min="1" max="20" step="1" value="1" required disabled></label><button id="update" type="submit" disabled>Update cart</button></form>
<p class="muted">Edit the quantity, then save it with Update cart.</p><p id="saved-quantity"></p></div>
<p id="cart-total">Cart total: CNY 0.00</p></section></main>
<p id="status" role="status" aria-live="polite">Loading cart...</p>
<footer>Synthetic test fixture · No payments are processed · Not a teacher-provided scenario.</footer>
</div>
<script>
const fault=new URLSearchParams(location.search).get('fault')||'none';
const byId=id=>document.getElementById(id), statusNode=byId('status');
function busy(value){for(const id of ['add','reset','quantity','update'])byId(id).disabled=value;}
function render(cart){
 byId('cart-empty').hidden=cart.quantity>0;
 byId('cart-details').hidden=cart.quantity===0;
 byId('cart-item').textContent='Cart item: '+cart.name;
 byId('quantity').value=cart.quantity||1;
 byId('saved-quantity').textContent='Cart quantity: '+cart.quantity;
 const total=fault==='wrong_total'&&cart.quantity>1?cart.unit_price_cents:cart.total_cents;
 byId('cart-total').textContent='Cart total: CNY '+(total/100).toFixed(2);
}
async function request(path,body){
 const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
 const response=await fetch(path,options), data=await response.json();
 if(!response.ok)throw new Error(data.error||'Cart request failed');
 return data;
}
async function update(path,body,pending,complete){
 busy(true);statusNode.textContent=pending;
 try{render(await request(path,body));statusNode.textContent=complete;}
 catch(error){statusNode.textContent=error.message;}
 finally{busy(false);}
}
byId('add').onclick=()=>update('/cart/add',{},'Adding item...','Item added');
byId('quantity-form').onsubmit=event=>{
 event.preventDefault();
 update('/cart/update',{quantity:Number(byId('quantity').value)},'Updating cart...','Cart updated');
};
byId('reset').onclick=()=>update('/reset',{},'Resetting cart...','Cart reset');
update('/cart',undefined,'Loading cart...','Ready');
</script></body></html>'''


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, DemoHandler)
        self.quantity = 0
        self.state_lock = threading.Lock()

    def cart_state(self):
        """Call while holding state_lock; monetary amounts are integer cents."""
        return {"sku": "notebook", "name": "Notebook", "unit_price_cents": UNIT_PRICE_CENTS,
                "quantity": self.quantity, "total_cents": self.quantity * UNIT_PRICE_CENTS}


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, status, payload, content_type="application/json"):
        data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            fault = parse_qs(urlsplit(self.path).query).get("fault", ["none"])[0]
            if fault not in {"none", "wrong_total"}:
                self._send(400, {"error": "Unknown fixture fault"})
            else:
                self._send(200, HTML, "text/html")
        elif path == "/health":
            self._send(200, {"status": "ok", "synthetic_fixture": True})
        elif path == "/cart":
            with self.server.state_lock:
                cart = self.server.cart_state()
            self._send(200, cart)
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/reset":
            with self.server.state_lock:
                self.server.quantity = 0
                cart = self.server.cart_state()
            self._send(200, cart)
            return
        if path not in {"/cart/add", "/cart/update"}:
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            quantity = payload.get("quantity")
            if path == "/cart/update" and (type(quantity) is not int or not 1 <= quantity <= 20):
                raise ValueError("Quantity must be an integer from 1 to 20")
        except (ValueError, AttributeError, UnicodeError):
            self._send(400, {"error": "Invalid cart request; quantity must be an integer from 1 to 20"})
            return
        with self.server.state_lock:
            if path == "/cart/add":
                if self.server.quantity >= 20:
                    self._send(409, {"error": "The cart quantity limit is 20"})
                    return
                self.server.quantity += 1
            else:
                if self.server.quantity == 0:
                    self._send(409, {"error": "Add an item before updating its quantity"})
                    return
                self.server.quantity = quantity
            cart = self.server.cart_state()
        self._send(200, cart)


@contextmanager
def demo_server(host="127.0.0.1", port=0):
    """Start an isolated server; yields its URL and shuts it down on exit."""
    server = DemoServer((host, port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def serve_demo(host="127.0.0.1", port=DEFAULT_DEMO_PORT):
    """Serve the fixture and report its actual URL, including for port zero."""
    try:
        server = DemoServer((host, port))
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"Cannot bind demo server to {host}:{port}: {exc}. "
            "The port may be in use or reserved by the system. "
            "Try 'webtest demo-server --port 0' and use the printed URL.",
        ) from exc
    print(f"Synthetic demo fixture: http://{host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_DEMO_PORT)
    args = parser.parse_args()
    serve_demo(args.host, args.port)


if __name__ == "__main__":
    main()
