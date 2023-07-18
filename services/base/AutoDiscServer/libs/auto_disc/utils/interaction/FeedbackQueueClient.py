#!/usr/bin/env python3
import os
import pickle
import stat
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, Optional
from uuid import uuid4 as generate_uuid

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


@dataclass
class Feedback:
    content: Dict
    id: Optional[int] = field(default_factory=lambda: generate_uuid().int)


class FeedbackQueueClient:
    """Class whose instances are connection clients to a file-based message
    queue allowing two-way communication about feedbacks.

    NOTE: this class does not inherit from `Leaf` because the persistence of its
    state is already guaranteed by the existence of the file-based message queue.
    """

    def __init__(self, persist_path: str = "/tmp/messageq") -> None:
        # setup directories for the queue
        self.question_path = os.path.join(persist_path, "questions")
        self.response_path = os.path.join(persist_path, "responses")
        os.makedirs(self.question_path, exist_ok=True)
        os.makedirs(self.response_path, exist_ok=True)

        # create in-memory queues
        self.questions = Queue()
        self.responses = Queue()

    def get_question(self, timeout: int = 5, block: bool = True):
        """Get a question from the in-memory cache."""

        return self.questions.get(block=block, timeout=timeout)

    def put_question(self, question: Feedback):
        """Put a question to disk and synchronize with in-memory cache."""
        # persist feedback to disk
        file_to_write = os.path.join(self.question_path, str(question.id))
        with open(file_to_write, "wb") as f:
            pickle.dump(question, f)

        # set read-only
        os.chmod(file_to_write, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        # put into the in-memory queue
        self.questions.put(question)

        return

    def get_response(self, timeout: int = 5, block: bool = True):
        """Get a response from the in-memory cache."""
        return self.responses.get(block=block, timeout=timeout)

    def put_response(self, response: Feedback):
        """Put a response to disk and synchronize with in-memory cache."""
        # persist feedback to disk
        file_to_write = os.path.join(self.response_path, str(response.id))
        with open(file_to_write, "wb") as f:
            pickle.dump(response, f)

        # set read-only
        os.chmod(file_to_write, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        # put into the in-memory queue
        self.responses.put(response)

        return

    def listen_for_questions(self):
        """Asynchronously watch a directory and add new file paths to in-memory queue.

        Return
            A handle which can be used to terminate the watch by calling its
                shutdown() method
        Raises
            FileNotFoundError: Could not find directory dir.
        """
        return self._watch_directory(dir=self.question_path, queue=self.questions)

    def listen_for_responses(self):
        """Asynchronously watch a directory and add new file paths to in-memory queue."""
        return self._watch_directory(dir=self.response_path, queue=self.responses)

    @staticmethod
    def _watch_directory(dir: str, queue: Queue):
        """Asynchronously watch a directory and add new file paths to queue.

        For performance reasons, the opening of files are to be handled by the
        caller, however the `Feedback` queues are such that each feedback is
        read-only on disk.

        Args
            dir: Path to the directory to watch.
            queue: In-memory queue to which file paths will be added.

        Return
            A handle which can be used to terminate the watch by calling its
                shutdown() method
        Raises
            FileNotFoundError: Could not find directory dir.
        """
        if not os.path.isdir(dir):
            raise FileNotFoundError(
                f"Could not open directory with name {dir}, it may not exist."
            )

        class QueueHandler(FileSystemEventHandler):
            def __init__(self, q: Queue):
                super().__init__()
                self.queue = q

            def on_created(self, event):
                # only act on file creations
                if event.is_directory == False:
                    self.queue.put(event.src_path)

        observer = Observer()
        queue_handler = QueueHandler(queue)
        observer.schedule(queue_handler, dir, recursive=False)
        observer.start()

        return observer


def main():
    pass


if __name__ == "__main__":
    main()
