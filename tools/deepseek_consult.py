#!/usr/bin/env python3
"""Read-only consultant bridge for the locally hosted DeepSeek Harness UI."""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import time
import urllib.parse
import urllib.request
import uuid

import yaml


READ_ONLY_PREAMBLE = """You are a read-only technical consultant. Do not call tools, run
Shell commands, read or write files, create subagents, or modify the workspace. Work only
from the public facts in this prompt. Codex is the sole workspace writer and independently
tests every accepted suggestion. Do not reveal hidden chain-of-thought; return only concise,
public technical analysis.\n\n"""


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    parsed = urllib.parse.urlparse(config["base_url"])
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("DeepSeek Harness base_url must be loopback HTTP")
    if config.get("transport") != "dsh_harness":
        raise ValueError("only dsh_harness transport is supported")
    return config


class HarnessClient:
    def __init__(self, base_url, timeout_s):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        # Do not route loopback traffic through HTTP(S)_PROXY.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, method, payload, timeout_s=None):
        rpc_id = str(uuid.uuid4())
        body = {
            "type": "client-request",
            "rpcId": rpc_id,
            "method": method,
            "payload": payload,
        }
        request = urllib.request.Request(
            self.base_url + "/api/" + method,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=timeout_s or self.timeout_s) as response:
            output = json.load(response)
        if output.get("rpcId") != rpc_id:
            raise RuntimeError("DeepSeek Harness RPC id mismatch")
        result = output.get("result", {})
        if not result.get("ok"):
            raise RuntimeError(json.dumps(result.get("error", result), ensure_ascii=False))
        return result.get("value")

    def last_sequence(self, session_id):
        history = self.call("session.history", {"sessionId": session_id, "maxMessages": 1})
        return max((entry["event"]["seq"] for entry in history["events"]), default=-1)

    def running(self, session_id):
        sessions = self.call("session.list", {})["items"]
        for session in sessions:
            if session["sessionId"] == session_id:
                return bool(session["running"])
        raise RuntimeError("DeepSeek Harness session not found: " + session_id)


def response_after(client, session_id, after_seq):
    history = client.call(
        "session.history", {"sessionId": session_id, "maxMessages": 20}
    )
    events = [
        item["event"] for item in history["events"] if item["event"]["seq"] > after_seq
    ]
    tool_events = [event for event in events if event["type"].startswith("tool/")]
    messages = [event for event in events if event["type"] == "assistant/message"]
    if not messages:
        raise RuntimeError("DeepSeek Harness completed without an assistant message")
    parts = messages[-1].get("data", {}).get("message", {}).get("content", [])
    text_parts = [part.get("text", "") for part in parts if part.get("type") == "text"]
    answer = "\n\n".join(part for part in text_parts if part)
    if not answer:
        raise RuntimeError("DeepSeek Harness assistant message has no public text")
    return answer, events, tool_events


def safe_stem(value):
    allowed = []
    for character in value.lower():
        allowed.append(character if character.isalnum() else "_")
    return "_".join(filter(None, "".join(allowed).split("_")))[:60] or "consult"


def save_dialogue(root, config, task, session_id, prompt, answer, events, tool_events):
    now = datetime.datetime.now(datetime.timezone.utc)
    dialogue_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + safe_stem(task)
    docs_dir = root / config["docs_dir"]
    results_dir = root / config["results_dir"]
    docs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "dialogue_id": dialogue_id,
        "timestamp_utc": now.isoformat(),
        "task": task,
        "session_id": session_id,
        "provider": config["provider"],
        "model": config["model"],
        "reasoning_effort": config.get("reasoning_effort"),
        "codex_prompt": prompt,
        "deepseek_public_answer": answer,
        "deepseek_patch": None,
        "tool_events_detected": [event["type"] for event in tool_events],
        "codex_review": "PENDING",
        "codex_decision": "PENDING",
        "applied_files": [],
        "applied_diff_sha256": None,
        "event_sequence_range": [events[0]["seq"], events[-1]["seq"]] if events else [],
    }
    json_path = results_dir / (dialogue_id + ".json")
    markdown_path = docs_dir / (dialogue_id + ".md")
    with open(json_path, "x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, sort_keys=True)
    markdown = """# {task}

- Dialogue ID: `{dialogue_id}`
- Session: `{session_id}`
- Model: `{provider}/{model}`
- Tool events: `{tool_events}`

## Codex → DeepSeek

{prompt}

## DeepSeek → Codex

{answer}

## Codex review

PENDING

## Decision and applied diff

PENDING
""".format(
        task=task,
        dialogue_id=dialogue_id,
        session_id=session_id,
        provider=config["provider"],
        model=config["model"],
        tool_events=", ".join(record["tool_events_detected"]) or "NONE",
        prompt=prompt,
        answer=answer,
    )
    with open(markdown_path, "x", encoding="utf-8") as stream:
        stream.write(markdown)
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    return dialogue_id, markdown_path, json_path, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=".codex/deepseek.local.yaml")
    parser.add_argument("--task", required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as stream:
            user_prompt = stream.read()
    else:
        user_prompt = args.prompt
    prompt = READ_ONLY_PREAMBLE + user_prompt.strip()
    session_id = config["session_id"]
    client = HarnessClient(config["base_url"], config["timeout_s"])

    models = client.call("session.models", {"sessionId": session_id})
    target = {
        "provider": config["provider"],
        "model": config["model"],
        **(
            {"reasoningEffort": config["reasoning_effort"]}
            if config.get("reasoning_effort")
            else {}
        ),
    }
    if models["current"] != target:
        client.call("session.selectModel", {"sessionId": session_id, **target})
    if client.running(session_id):
        raise RuntimeError("DeepSeek Harness session is already running")
    after_seq = client.last_sequence(session_id)
    print("\n=== Codex -> DeepSeek ({}) ===\n{}\n".format(args.task, prompt), flush=True)
    client.call(
        "session.prompt",
        {
            "sessionId": session_id,
            "mode": "queue",
            "content": [{"type": "text", "text": prompt}],
            "clientTimeZone": config["client_time_zone"],
        },
    )
    deadline = time.monotonic() + float(config["timeout_s"])
    saw_running = False
    while time.monotonic() < deadline:
        running = client.running(session_id)
        saw_running = saw_running or running
        if saw_running and not running:
            break
        time.sleep(float(config["poll_interval_s"]))
    else:
        raise TimeoutError("DeepSeek Harness consultation timed out")
    answer, events, tool_events = response_after(client, session_id, after_seq)
    print("\n=== DeepSeek -> Codex ===\n{}\n".format(answer), flush=True)
    dialogue_id, markdown_path, json_path, digest = save_dialogue(
        root, config, args.task, session_id, prompt, answer, events, tool_events
    )
    print(
        json.dumps(
            {
                "dialogue_id": dialogue_id,
                "markdown": str(markdown_path),
                "json": str(json_path),
                "json_sha256": digest,
                "tool_events": [event["type"] for event in tool_events],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if tool_events:
        raise SystemExit("DeepSeek violated read-only no-tools consultation boundary")


if __name__ == "__main__":
    main()
