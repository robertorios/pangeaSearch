"""Run: python -m app.worker

Requires Redis (same instance as Sidekiq is fine; uses DB index 1 by default).

On macOS, RQ's default fork worker crashes with Objective-C / torch / Whisper
(objc_initializeAfterForkError → "Python quit unexpectedly"). We use
SimpleWorker there (no fork). Linux keeps the normal Worker.
"""

import logging
import platform
import sys

from redis import Redis
from rq import SimpleWorker, Worker

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def main() -> None:
    settings = get_settings()
    redis_conn = Redis.from_url(settings.redis_url)
    # macOS + ML stacks (torch/whisper) cannot safely fork after objc init.
    worker_cls = SimpleWorker if platform.system() == "Darwin" else Worker
    print(
        "pangeaSearch RQ worker listening on queue=%r redis=%r whisper=%r embed=%r mode=%s"
        % (
            settings.rq_queue_name,
            settings.redis_url,
            settings.whisper_model,
            settings.embedding_model,
            worker_cls.__name__,
        ),
        flush=True,
    )
    worker = worker_cls([settings.rq_queue_name], connection=redis_conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
