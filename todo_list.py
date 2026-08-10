class TodoList:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def add_task(self, task: str) -> None:
        self.tasks.append(task)

    def remove_task(self, task: str) -> None:
        if task in self.tasks:
            self.tasks.remove(task)

    def list_tasks(self) -> list[str]:
        return self.tasks
