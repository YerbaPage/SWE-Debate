import libcst as cst
import libcst.matchers as m
import ast
import tokenize
from io import StringIO
import logging
# from libcst.display import dump


def parse_class_docstrings(target_file: str) -> list:
    with open(target_file, 'r') as f:
        source_code = f.read()
        
    # Parse the code string
    parsed_code = ast.parse(source_code)
    docstring_nodes = []
    # Iterate through nodes to find the class definition
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.ClassDef):
            # Retrieve the class docstring
            docstring = ast.get_docstring(node)
            if docstring:
                # Find the start and end lines of the docstring
                start_line = node.body[0].lineno  # First node under the class is usually the docstring
                end_line = start_line + len(docstring.splitlines()) - 1
                docstring_nodes.append({
                    'start_line': start_line,
                    'end_line': end_line,
                    'content': docstring
                })
    return docstring_nodes


# class ClassDocstringVisitor(cst.CSTVisitor):
#     def __init__(self):
#         self.class_docstrings = []

#     def visit_ClassDef(self, node: cst.ClassDef):
#         # Extract the docstring if it's the first statement in the body
#         if node.body.body and isinstance(node.body.body[0], cst.SimpleStatementLine):
#             first_stmt = node.body.body[0].body[0]
#             if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
#                 # This is the docstring
#                 self.class_docstrings.append(first_stmt.value.value)


# def parse_class_docstrings(file_content: str) -> list:
#     """Parse the docstrings of classes in the given code."""
#     try:
#         tree = cst.parse_module(file_content)
#     except:
#         return []

#     visitor = ClassDocstringVisitor()
#     tree.visit(visitor)
#     return visitor.class_docstrings


def parse_import_nodes(target_file):
    with open(target_file, 'r') as f:
        source_code = f.read()

    # Parse the source code
    tree = ast.parse(source_code)
    class ImportCollector(ast.NodeVisitor):
        def __init__(self):
            self.imports = []

        def visit_Import(self, node):
            self.imports.append({
                "type": "import",
                "module": None,  # Regular imports don't specify a module
                "names": [alias.name for alias in node.names],
                "start_line": node.lineno,
                "end_line": getattr(node, 'end_lineno', node.lineno)  # Use node.lineno if end_lineno is not available
            })
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            self.imports.append({
                "type": "from import",
                "module": node.module,
                "names": [alias.name for alias in node.names],
                "start_line": node.lineno,
                "end_line": getattr(node, 'end_lineno', node.lineno)  # Use node.lineno if end_lineno is not available
            })
            self.generic_visit(node)

    import_collector = ImportCollector()
    import_collector.visit(tree)

    # return the collected imports
    return import_collector.imports


def parse_comment_nodes(target_file):
    comment_nodes = []
    with open(target_file, 'r') as f:
        source_code = f.read()
    # Tokenize the source code to find comments and their locations
    source = StringIO(source_code)
    tokens = tokenize.generate_tokens(source.readline)

    for token_type, token_string, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            # For comments, this will usually be the same as start_line
            comment_nodes.append({
                "start_line": start[0],
                "end_line": end[0],
                "content": token_string
            })
            logging.debug(f"Found comment: {token_string} starting at line {start[0]} and ending at line {end[0]}")
    return comment_nodes


def is_import_statement(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False


def is_comment(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False 


def is_docstring(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False


class GlobalVariableVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        self.global_assigns = []

    def leave_Module(self, original_node: cst.Module) -> list:
        assigns = []
        for stmt in original_node.body:
            # Match simple assignments
            if m.matches(stmt, m.SimpleStatementLine()) and m.matches(stmt.body[0], m.Assign()):
                start_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).start
                end_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).end
                assigns.append([stmt, start_pos, end_pos])

            # Match annotated assignments (AnnAssign)
            elif m.matches(stmt, m.SimpleStatementLine()) and m.matches(stmt.body[0], m.AnnAssign()):
                start_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).start
                end_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).end
                assigns.append([stmt, start_pos, end_pos])

        self.global_assigns.extend(assigns)


def parse_global_var_from_code(file_content: str) -> dict[str, dict]:
    """Parse global variables."""
    try:
        tree = cst.parse_module(file_content)
    except:
        return file_content

    wrapper = cst.metadata.MetadataWrapper(tree)
    visitor = GlobalVariableVisitor()
    wrapper.visit(visitor)

    global_assigns = {}
    for assign_stmt, start_pos, end_pos in visitor.global_assigns:
        # Handle both Assign and AnnAssign cases
        if isinstance(assign_stmt.body[0], cst.Assign):
            for t in assign_stmt.body:
                try:
                    targets = [t.targets[0].target.value]
                except:
                    try:
                        targets = t.targets[0].target.elements
                        targets = [x.value.value for x in targets]
                    except:
                        targets = []
                for target_var in targets:
                    global_assigns[target_var] = {
                        "start_line": start_pos.line,
                        "end_line": end_pos.line,
                    }
        elif isinstance(assign_stmt.body[0], cst.AnnAssign):
            targets = [assign_stmt.body[0].target.value]
        else:
            targets = []

        for target_var in targets:
            global_assigns[target_var] = {
                "start_line": start_pos.line,
                "end_line": end_pos.line,
            }
    return global_assigns


def parse_global_var_from_file(file_path):
    with open(file_path, 'r') as f:
        file_content = f.read()
    global_vars = parse_global_var_from_code(file_content)
    return global_vars


def is_global_var(line, global_vars):
    for gvar, lrange in global_vars.items():
        if line >= lrange['start_line'] and line <= lrange['end_line']:
            return gvar
    return None


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef) or isinstance(
                    n, ast.AsyncFunctionDef
                ):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) or isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()