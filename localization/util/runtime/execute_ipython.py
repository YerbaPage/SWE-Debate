from IPython import get_ipython
from plugins.location_tools.repo_ops.repo_ops import (
    search_code_snippets,
    get_entity_contents,
    explore_graph_structure,
    explore_tree_structure,
)

from IPython.utils.capture import capture_output
from IPython.terminal.interactiveshell import TerminalInteractiveShell


def execute_ipython(code_to_execute):
    # Manually initialize an IPython shell
    ipython_shell = TerminalInteractiveShell.instance()

    # Inject the function into the IPython environment
    ipython_shell.user_ns['search_code_snippets'] = search_code_snippets
    ipython_shell.user_ns['get_entity_contents'] = get_entity_contents
    ipython_shell.user_ns['explore_graph_structure'] = explore_graph_structure
    ipython_shell.user_ns['explore_tree_structure'] = explore_tree_structure
    # ipython_shell.user_ns['explore_repo_structure'] = explore_repo_structure
    # ipython_shell.user_ns['search_interactions_among_modules'] = search_interactions_among_modules

    # Execute the code in the IPython shell
    with capture_output() as captured:
        ipython_shell.run_cell(code_to_execute)

    output = ''
    if captured.stdout:
        output += captured.stdout
    if captured.stderr:
        output += captured.stderr

    return output if output else None