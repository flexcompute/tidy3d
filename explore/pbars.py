# from time import sleep
# from rich.console import Console

# console = Console()
# console.print()

# tasks = [f"task {n}" for n in range(1, 11)]

# with console.status("[bold green]Working on tasks...") as status:
#     while tasks:
#         task = tasks.pop(0)
#         for i in range(100):
#             sleep(.01)
#         console.log(f"{task} complete")


from time import sleep

from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table


job_progress = Progress(
    "{task.description}",
    SpinnerColumn(),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
)
job1 = job_progress.add_task("[green]Cooking")
job2 = job_progress.add_task("[magenta]Baking", total=200)
job3 = job_progress.add_task("[cyan]Mixing", total=400)

total = sum(task.total for task in job_progress.tasks)
overall_progress = Progress()
overall_task = overall_progress.add_task("All Jobs", total=int(total))

progress_table = Table.grid()
progress_table.add_row(
    Panel.fit(overall_progress, title="Overall Progress", border_style="green", padding=(2, 2)),
    Panel.fit(job_progress, title="[b]Jobs", border_style="red", padding=(1, 2)),
)

with Live(progress_table, refresh_per_second=10):
    while not overall_progress.finished:
        sleep(0.1)
        for job in job_progress.tasks:
            if not job.finished:
                job_progress.advance(job.id)

        completed = sum(task.completed for task in job_progress.tasks)
        overall_progress.update(overall_task, completed=completed)
