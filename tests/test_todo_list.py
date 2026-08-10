import pytest
from todo_list import TodoList

class TestTodoList:
    def test_add_task(self) -> None:
        todo_list = TodoList()
        todo_list.add_task('Buy groceries')
        assert 'Buy groceries' in todo_list.list_tasks()

def test_remove_task() -> None:
    todo_list = TodoList()
    todo_list.add_task('Walk the dog')
    todo_list.remove_task('Walk the dog')
    assert 'Walk the dog' not in todo_list.list_tasks()

@pytest.mark.parametrize("task", ['Read a book', 'Write code'])
def test_list_tasks(task: str) -> None:
    todo_list = TodoList()
    todo_list.add_task(task)
    tasks = todo_list.list_tasks()
    assert task in tasks
    assert len(tasks) == 1