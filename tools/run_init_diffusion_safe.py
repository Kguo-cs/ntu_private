"""Native src/run.py CLI with the CPU-verified cached scene loader and GC off.

Example: python tools/run_init_diffusion_safe.py experiment=init_diffusion_overnight_best
Hydra arguments are forwarded unchanged. This launcher does not alter the model,
loss, optimizer, checkpoint path, data ordering, seed or evaluation definitions.
Runs one owned child with default limits of18GiB RSS and1.5GiB host MemAvailable.
Hydra CLI is unchanged. SIM_NATIVE_RUNTIME_AUDIT sets the parent audit JSON path;
by default it is created under logs/native_runtime_guard/. The child policy is in
the adjacent .child.json. SIM_NATIVE_MAX_RSS_GIB/SIM_NATIVE_MIN_AVAILABLE_GIB can
override limits explicitly. Intended native configurations use workers=0. Memory
stops and interruptions do not promise a saved checkpoint or exact continuation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHILD_ENV = 'SIM_NATIVE_RUNTIME_GUARD_CHILD'
GIB_KIB = 1024 ** 2


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT/'src/run.py',
             ROOT/'tools/runtime_cached_torch_loader_sweep.py', ROOT/'tools/torch_pickle_class_cache.py']
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_native_child():
    gc.disable()
    gc.collect = lambda *unused_args, **unused_kwargs: 0
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader, runtime_policy
    install_cached_loader(torch)
    policy = runtime_policy()
    policy.update(source_hashes=source_hashes(), entrypoint='native src/run.py via runpy',
                  scope='current process; intended native configs have num_workers=0',
                  argv=sys.argv[1:], status='running')
    print('NATIVE_RUNTIME_POLICY '+json.dumps(policy, sort_keys=True), flush=True)
    audit_path = os.environ.get('SIM_NATIVE_RUNTIME_AUDIT')
    if audit_path:
        path = Path(audit_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(policy, indent=2)+'\n')
    sys.argv[0] = str(ROOT/'src/run.py')
    try:
        runpy.run_path(str(ROOT/'src/run.py'), run_name='__main__')
    except SystemExit as error:
        # Hydra --help/--cfg exits0 normally; failures keep the actual code.
        if error.code not in (None, 0):
            raise
    policy.update(runtime_policy())
    policy.update(source_hashes=source_hashes(), status='success')
    if audit_path:
        path.write_text(json.dumps(policy, indent=2)+'\n')
    print('NATIVE_RUNTIME_SUCCESS '+json.dumps(policy, sort_keys=True), flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def memory_limits():
    limits = {'max_rss_gib': float(os.environ.get('SIM_NATIVE_MAX_RSS_GIB', '18')),
              'min_available_gib': float(os.environ.get('SIM_NATIVE_MIN_AVAILABLE_GIB', '1.5')),
              'poll_seconds': float(os.environ.get('SIM_NATIVE_MEMORY_POLL_SECONDS', '2'))}
    import math
    if (not all(math.isfinite(value) for value in limits.values()) or limits['max_rss_gib'] <= 0
            or limits['min_available_gib'] < 0 or not 0.1 <= limits['poll_seconds'] <= 60):
        raise ValueError('Native memory limits must be finite; RSS>0, MemAvailable>=0, poll0.1..60s')
    return limits


def save_audit(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def read_memory(pid):
    status = Path(f'/proc/{pid}/status').read_text()
    meminfo = Path('/proc/meminfo').read_text()
    rss = int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))
    available = int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith('MemAvailable:')))
    return {'child_rss_kib': rss, 'host_mem_available_kib': available}


def memory_stop_reason(memory, limits):
    if memory['child_rss_kib'] > limits['max_rss_gib']*GIB_KIB:
        return 'child_RSS_exceeded_limit'
    if memory['host_mem_available_kib'] < limits['min_available_gib']*GIB_KIB:
        return 'host_MemAvailable_below_limit'
    return None


def stop_owned_child(process):
    """Stop only our new session; retain actual return code, with bounded wait."""
    sent = []
    if process.poll() is not None:
        return sent
    try:
        os.killpg(process.pid, signal.SIGINT)
        sent.append('SIGINT')
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        sent.append('SIGKILL')
        process.wait(timeout=10)
    except ProcessLookupError:
        process.wait(timeout=10)
    return sent


class GuardInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__(f'Native supervisor received signal{signum}')
        self.signum = signum


def supervisor_signal(signum, unused_frame):
    raise GuardInterrupted(signum)


def run_guarded(argv, audit_path, limits=None, *, memory_reader=read_memory):
    """Stdlib parent: no Torch import, fit, checkpoint load or objective change."""
    limits = memory_limits() if limits is None else limits
    audit_path = Path(audit_path).resolve()
    child_path = audit_path.with_name(audit_path.stem+'.child.json')
    if audit_path.exists() or child_path.exists():
        raise FileExistsError('Native runtime audit already exists; choose a fresh SIM_NATIVE_RUNTIME_AUDIT path')
    command = [sys.executable, str(Path(__file__).resolve()), *argv]
    record = {'status': 'starting', 'command': command, 'Hydra_cli': list(argv), 'parent_pid': os.getpid(),
              'source_hashes': source_hashes(), 'limits': limits, 'audit_path': str(audit_path),
              'child_policy_path': str(child_path), 'started_unix': time.time(), 'memory_polls': 0,
              'peak_observed_child_rss_kib': 0, 'minimum_observed_host_mem_available_kib': None,
              'stop_reason': None, 'stop_signals': [], 'training_objective_changed': False,
              'checkpoint_auto_save_or_exact_resume_promised': False,
              'memory_scope': 'owned child RSS plus global host MemAvailable; RSS is not sum of worker descendants',
              'configuration_only': '--cfg' in argv or '--help' in argv}
    save_audit(audit_path, record)
    process = None
    try:
        process = subprocess.Popen(command, env=dict(os.environ, **{CHILD_ENV: '1',
                                   'SIM_NATIVE_RUNTIME_AUDIT': str(child_path)}), start_new_session=True)
        record.update(status='running', child_pid=process.pid)
        save_audit(audit_path, record)
        print('NATIVE_GUARD_AUDIT '+str(audit_path), flush=True)
        while process.poll() is None:
            try:
                memory = memory_reader(process.pid)
            except (OSError, StopIteration, ValueError):
                if process.poll() is not None:
                    break
                raise
            record.update(last_memory=memory, memory_polls=record['memory_polls']+1)
            record['peak_observed_child_rss_kib'] = max(record['peak_observed_child_rss_kib'], memory['child_rss_kib'])
            previous = record['minimum_observed_host_mem_available_kib']
            record['minimum_observed_host_mem_available_kib'] = (memory['host_mem_available_kib'] if previous is None
                                                               else min(previous, memory['host_mem_available_kib']))
            reason = memory_stop_reason(memory, limits)
            if reason:
                record.update(status='memory_limit_stopped', stop_reason=reason, stop_requested_unix=time.time())
                save_audit(audit_path, record)
                record['stop_signals'] = stop_owned_child(process)
                break
            save_audit(audit_path, record)
            time.sleep(limits['poll_seconds'])
        record.update(child_returncode=process.wait(), ended_unix=time.time())
        if record['status'] == 'memory_limit_stopped':
            code = 1
        elif record['child_returncode'] != 0:
            record.update(status='child_failed', stop_reason='native_child_nonzero_exit')
            code = record['child_returncode'] if record['child_returncode'] > 0 else 128-record['child_returncode']
        else:
            policy = json.loads(child_path.read_text())
            if (policy.get('status') != 'success' or policy.get('automatic_gc_enabled') is not False
                    or policy.get('source_hashes') != record['source_hashes']):
                raise RuntimeError('Native child returned0 without the actual successful unchanged runtime policy')
            record.update(status='success', child_policy_sha256=hashlib.sha256(child_path.read_bytes()).hexdigest())
            code = 0
    except (KeyboardInterrupt, InterruptedError) as error:
        signum = getattr(error, 'signum', signal.SIGINT)
        record.update(status='interrupted', stop_reason='supervisor_interrupted', interrupt_signal=int(signum),
                      error=repr(error), ended_unix=time.time())
        if process is not None:
            record['stop_signals'] = stop_owned_child(process)
            record['child_returncode'] = process.returncode
        code = 128+int(signum)
    except BaseException as error:
        record.update(status='supervisor_failed', stop_reason='supervisor_or_memory_monitor_error',
                      error=repr(error), ended_unix=time.time())
        if process is not None:
            record['stop_signals'] = stop_owned_child(process)
            record['child_returncode'] = process.returncode
        code = 1
    save_audit(audit_path, record)
    print('NATIVE_GUARD_FINISHED '+json.dumps(record, sort_keys=True), flush=True)
    return code


def main():
    if os.environ.get(CHILD_ENV) == '1':
        return run_native_child()
    signal.signal(signal.SIGINT, supervisor_signal)
    signal.signal(signal.SIGTERM, supervisor_signal)
    stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    audit = os.environ.get('SIM_NATIVE_RUNTIME_AUDIT') or str(ROOT/'logs/native_runtime_guard'/f'{stamp}_{os.getpid()}.json')
    raise SystemExit(run_guarded(sys.argv[1:], audit))


if __name__ == '__main__':
    main()
