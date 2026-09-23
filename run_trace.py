import json
import sys
from pathlib import Path
from science_env import ScienceSandbox

def run_trace(path):
    env = ScienceSandbox()
    trace = json.loads(Path(path).read_text())
    transcript = []
    for i, step in enumerate(trace, 1):
        result = env.call(step["tool"], step.get("args", {}))
        transcript.append({"step": i, "call": step, "result": result})
    return transcript

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "traces/good_trace.json"
    transcript = run_trace(path)
    print(json.dumps(transcript, indent=2))
