"""Python Insertion Strategy

Handles code insertion for indentation-based blocks in Python.

This strategy:
* **Calculates Indentation**: Derives the required prefix based on the column 
    offset of the block's start point.
* **Validates Insertion Points**: Permits placement at the very start of a 
    block or immediately following any newline node.
* **Defines Terminal Conditions**: Identifies control-flow exits and the 
    'pass' statement as boundaries to prevent redundant or dead-code insertion.
"""

from typing import Optional

from .insertion_strategy import InsertionStrategy
from .....node import Node


class PythonInsertionStrategy(InsertionStrategy):
    """
    Structural strategy for Python-specific code insertion.

    This strategy allows for insertions at the start of blocks (after the colon)
    and after any statement followed by a newline. It specifically treats the
    'pass' statement as a block terminator to avoid mutating abstract or
    placeholder methods.
    """

    def get_indent_prefix(self, node: Node) -> str:
        """
        Calculates the indentation prefix.

        Args:
            node (Node): The 'block_scope' node being analyzed.

        Returns:
            str: The whitespace string to be used as a prefix for inserted code,
                 or empty string if the column information is unavailable.
        """
        siblings = node.parent.children if node.parent else []
        idx = siblings.index(node)
        if idx == 0:
            return ""  # no preceding sibling

        preceding = siblings[idx - 1]
        if preceding.text and preceding.type == "whitespace":
            return preceding.text

        return ""  # No whitespace used

    def is_valid_container(self, node: Node) -> bool:
        """
        Validates that the node is a block scope suitable for insertion.

        Checks if the node does not contain a 'pass' statement, which indicates an abstract or placeholder block.
        Also ensures that the block is not a single-line block by verifying that its starting column is different
        from its parent's starting column.

        Args:
            node (Node): The node to validate.

        Returns:
            bool: True if the node is a valid block scope, False otherwise.
        """
        if any(child.type == "pass_statement" for child in node.children):
            return False

        if node.parent and node.parent.start_point[0] == node.start_point[0]:
            return False

        return True

    def is_valid_gap(self, current: Node, preceding: Optional[Node]) -> bool:
        """
        Determines if the gap before the current node is a valid Python insertion point.

        In Python, a gap is valid if it is at the very beginning of a block
        (preceding is None) or if it follows a whitespace node  .

        Args:
            current (Node): The node immediately following the potential insertion.
            preceding (Node | None): The node immediately preceding the insertion.

        Returns:
            bool: True if insertion is syntactically safe, False otherwise.
        """
        # At the very start of a block is always a valid gap
        if preceding is None:
            return True

        # After any whitespace node, ensuring the new code starts on its own line
        return preceding.type == "whitespace"

    def is_terminal(self, node: Node) -> bool:
        """
        Identifies statements that should halt further insertions within a block.

        Args:
            node (Node): The node to evaluate.

        Returns:
            bool: True if the node is a return, break, continue, raise, or pass.
        """
        terminal_types = {
            "return_statement",
            "break_statement",
            "continue_statement",
            "pass_statement",
        }
        return node.type in terminal_types
