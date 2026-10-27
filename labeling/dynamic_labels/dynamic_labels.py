import json
import re
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests


LS_URL = "http://127.0.0.1:19090"
PORT = 19091
DEFAULT_PROJECT_ID = 3

BASE_DIR = Path(__file__).resolve().parent
TOKEN_FILE = BASE_DIR / "token.txt"

RECT_NAME = "label"
TEXT_NAME = "dynamic_class"
SENTINEL = "__dynamic__"

lock = threading.Lock()


def clean_class_name(value):
    if not isinstance(value, str):
        return None

    value = re.sub(r"\s+", " ", value.strip())

    if not value:
        return None

    if value == SENTINEL:
        return None

    return value


def read_token():
    token = TOKEN_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not token:
        raise RuntimeError("token.txt is empty")

    return token


def get_headers():
    token = read_token()

    # Personal Access Token
    try:
        r = requests.post(
            f"{LS_URL}/api/token/refresh",
            json={"refresh": token},
            timeout=10,
        )

        if r.ok:
            access = r.json().get("access")

            if access:
                return {
                    "Authorization": f"Bearer {access}",
                    "Content-Type": "application/json",
                }

    except requests.RequestException:
        pass

    # Legacy token fallback
    return {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }


def api(method, path, **kwargs):
    response = requests.request(
        method,
        f"{LS_URL}{path}",
        headers=get_headers(),
        timeout=20,
        **kwargs,
    )

    response.raise_for_status()
    return response


def project_id_from_payload(payload, annotation):
    value = annotation.get("project")

    if isinstance(value, int):
        return value

    if isinstance(value, dict):
        value = value.get("id")

        if value:
            return int(value)

    value = payload.get("project")

    if isinstance(value, int):
        return value

    if isinstance(value, dict):
        value = value.get("id")

        if value:
            return int(value)

    return DEFAULT_PROJECT_ID


def add_class_to_config(project_id, class_name):
    with lock:
        project = api(
            "GET",
            f"/api/projects/{project_id}/",
        ).json()

        config = project.get("label_config")

        if not config:
            raise RuntimeError(
                "Project has no label_config"
            )

        root = ET.fromstring(config)

        parent = root.find(
            f".//RectangleLabels[@name='{RECT_NAME}']"
        )

        if parent is None:
            raise RuntimeError(
                "RectangleLabels name='label' not found"
            )

        existing = {
            node.attrib.get("value", "")
            .strip()
            .casefold()
            for node in parent.findall("Label")
        }

        if class_name.casefold() in existing:
            return False

        ET.SubElement(
            parent,
            "Label",
            {
                "value": class_name,
                "size": "small",
            },
        )

        ET.indent(root, space="  ")

        new_config = ET.tostring(
            root,
            encoding="unicode",
        )

        api(
            "PATCH",
            f"/api/projects/{project_id}/",
            json={
                "label_config": new_config,
            },
        )

        print(
            f"[CLASS ADDED] {class_name!r}",
            flush=True,
        )

        return True


def process_annotation(
    payload,
    annotation,
):
    annotation_id = annotation.get("id")

    if not annotation_id:
        return

    results = annotation.get("result", [])

    region_classes = {}

    # Find class text for each bbox region
    for result in results:
        if (
            result.get("type") == "textarea"
            and result.get("from_name") == TEXT_NAME
        ):
            region_id = result.get("id")

            texts = (
                result.get("value", {})
                .get("text", [])
            )

            if isinstance(texts, str):
                texts = [texts]

            if not texts:
                continue

            class_name = clean_class_name(
                texts[0]
            )

            if class_name and region_id:
                region_classes[
                    region_id
                ] = class_name

    if not region_classes:
        return

    project_id = project_id_from_payload(
        payload,
        annotation,
    )

    for class_name in set(
        region_classes.values()
    ):
        add_class_to_config(
            project_id,
            class_name,
        )

    converted = []

    for result in results:

        # Remove helper textarea result
        if (
            result.get("type") == "textarea"
            and result.get("from_name") == TEXT_NAME
            and result.get("id") in region_classes
        ):
            continue

        if (
            result.get("type")
            == "rectanglelabels"
            and result.get("from_name")
            == RECT_NAME
        ):
            region_id = result.get("id")

            if region_id in region_classes:
                value = result.setdefault(
                    "value",
                    {},
                )

                labels = value.get(
                    "rectanglelabels",
                    [],
                )

                if SENTINEL in labels:
                    value[
                        "rectanglelabels"
                    ] = [
                        region_classes[
                            region_id
                        ]
                    ]

        converted.append(result)

    api(
        "PATCH",
        f"/api/annotations/{annotation_id}/",
        json={
            "result": converted,
        },
    )

    print(
        f"[ANNOTATION CONVERTED] "
        f"id={annotation_id}",
        flush=True,
    )


def process_webhook(payload):
    action = payload.get("action")

    if action not in {
        "ANNOTATION_CREATED",
        "ANNOTATION_UPDATED",
    }:
        return

    annotation = (
        payload.get("annotation")
        or {}
    )

    try:
        process_annotation(
            payload,
            annotation,
        )

    except Exception as exc:
        print(
            f"[ERROR] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path in {
            "/",
            "/health",
        }:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            body = self.rfile.read(
                length
            )

            payload = json.loads(
                body.decode("utf-8")
            )

            threading.Thread(
                target=process_webhook,
                args=(payload,),
                daemon=True,
            ).start()

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        except Exception as exc:
            print(
                f"[WEBHOOK ERROR] {exc}",
                flush=True,
            )

            self.send_response(400)
            self.end_headers()

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    print(
        f"Dynamic labels listening "
        f"on 127.0.0.1:{PORT}",
        flush=True,
    )

    ThreadingHTTPServer(
        ("127.0.0.1", PORT),
        Handler,
    ).serve_forever()
