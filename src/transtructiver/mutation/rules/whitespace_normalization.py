"""whitespace_normalization.py

Defines the WhitespaceNormalizationRule, which standardizes indentation and
spacing across a Concrete Syntax Tree (CST). This rule enforces a consistent
indentation style and cleans up inline and trailing whitespace to ensure 
uniform code formatting.
"""

from typing import List

from ...mutation.mutation_context import MutationContext
from .mutation_rule import MutationRule, MutationRecord
from ...node import Node


# Default number of spaces per indentation level
DEFAULT_BASE_UNIT = 4
NUMERIC_TYPES = (
    "float",
    "integer",
    "number_literal",
    "decimal_integer_literal",
    "decimal_floating_point",
)


class WhitespaceNormalizationRule(MutationRule):
    """
    Concrete mutation rule that standardizes all whitespace and structural spacing within a CST.

    Normalization logic:
    - Indentation: Snaps to the nearest multiple of base_unit.
    - Trailing space: Removed entirely.
    - Inline space: Collapsed to exactly one space.
    - Structural Spacing: Injects missing whitespace after commas and around operators.
    - Padding: Strips unnecessary whitespace inside brackets and parentheses.

    Mutation Actions:
    - REFORMAT: Applied when modifying existing whitespace node text.
    - INSERT: Applied when injecting new synthetic whitespace nodes (with sentinel coordinates).
    - DELETE: Applied when removing empty lines.

    Attributes:
        level (int): The mutation level.
        base_unit (int): The number of spaces per indentation level.
    """

    # CLI rule name (used by the auto-discovery in cli.py).
    rule_name = "whitespace-normalization"

    def __init__(self, level: int = 0, base_unit: int = DEFAULT_BASE_UNIT):
        """
        Initializes the rule with a specific indentation base unit.

        Args:
            level (int): The mutation level.
            base_unit (int): Number of spaces per indentation level.
        """
        super().__init__()
        self._level = level
        self.base_unit = base_unit

    def is_numeric(self, node: Node) -> bool:
        """Checks if a node represents a numeric literal."""
        return any(t in node.type for t in NUMERIC_TYPES)

    def _is_indentation(self, node: Node) -> bool:
        """
        Checks if the node is at the start of a line (column 0).

        Args:
            node (Node): The node to inspect.

        Returns:
            bool: True if the node is at column 0, False otherwise.
        """
        return node.start_point[1] == 0

    def _is_trailing_whitespace(self, node: Node) -> bool:
        """
        Checks if the node is followed immediately by a newline node.

        Args:
            node (Node): The whitespace node to inspect.

        Returns:
            bool: True if the next sibling is a "newline" node.
        """
        if not node.parent:
            return False

        children = node.parent.children
        idx = children.index(node)
        # Check if there is a next sibling and if it is a newline
        return (idx + 1 < len(children)) and (children[idx + 1].type == "newline")

    def _snap_to_grid(self, indent_length: int, base_unit: int) -> int:
        """
        Rounds the indentation length to the nearest multiple of the base unit.

        Args:
            indent_length (int): The current length of the whitespace text.
            base_unit (int): The target indentation step.

        Returns:
            int: The normalized indentation length.
        """
        remainder = indent_length % base_unit

        # If it's already a multiple, no change needed
        if remainder == 0:
            return indent_length

        # Determine the distance to the "grid lines"
        distance_up = base_unit - remainder
        distance_down = remainder

        # Snap to the closest multiple
        if distance_up <= distance_down:
            return indent_length + distance_up
        else:
            return indent_length - distance_down

    def _is_padding_to_strip(self, node: Node) -> bool:
        """
        Identifies if a whitespace node is unwanted padding inside brackets.

        Args:
            node (Node): The whitespace node to inspect.

        Returns:
            bool: True if the node is adjacent to '(' or '[' and should be removed.
        """
        if not node.parent:
            return False

        children = node.parent.children
        idx = children.index(node)

        # Check if neighbor to the left is an opener or neighbor to the right is a closer
        if idx > 0 and children[idx - 1].type in ("(", "["):
            return True
        if idx + 1 < len(children) and children[idx + 1].type in (")", "]"):
            return True

        return False

    def _handle_structural_spacing(
        self, root: Node, child: Node, idx: int, context: MutationContext
    ) -> List[MutationRecord]:
        """
        Handles missing spaces after commas and around operators by injection.

        Args:
            root (Node): The parent node.
            child (Node): The current child node being inspected.
            idx (int): The current index of the child in the live children list.
            context (MutationContext): The mutation context for tracking state across rules.

        Returns:
            List[MutationRecord]: Records of any injected whitespace nodes.
        """
        records = []
        if idx + 1 >= len(root.children):
            return records

        next_node = root.children[idx + 1]

        if next_node.type in ("++", "--"):
            return records

        # Space after comma/operator, or before an operator
        is_trigger_before = getattr(next_node, "field", None) == "operator"
        is_trigger_after = (child.type == ",") or (getattr(child, "field", None) == "operator")

        # Skip inserting a space if the previous node is '-' and next node is numeric
        if child.type == "-" and self.is_numeric(next_node):
            return records

        # Insert a space if needed and not already present
        if (is_trigger_before or is_trigger_after) and next_node.type != "whitespace":
            new_ws = Node(
                start_point=(context.next_id(), -1),
                end_point=child.end_point,
                type="whitespace",
                text=" ",
            )
            new_ws.parent = root
            root.children.insert(idx + 1, new_ws)

            records.append(
                self.record_insert(
                    new_ws.start_point,
                    insertion_point=child.end_point,
                    new_text=" ",
                    new_type="whitespace",
                )
            )
        return records

    def _normalize_whitespace(self, node: Node) -> List[MutationRecord]:
        """
        Normalizes a single whitespace node.

        Handles indentation, trailing, and inline whitespace.

        Args:
            node (Node): The whitespace node to normalize.

        Returns:
            List[MutationRecord]: A list containing the mutation record if any.
        """
        records: List[MutationRecord] = []

        original_text = node.text if node.text else ""
        new_text = original_text

        if self._is_indentation(node):
            # Handle Indentation: Expand tabs and snap to grid
            expanded_len = len(original_text.expandtabs(self.base_unit))
            new_text = " " * self._snap_to_grid(expanded_len, self.base_unit)

        elif self._is_trailing_whitespace(node) or self._is_padding_to_strip(node):
            # Handle Trailing/Padding: Remove
            new_text = ""
        else:
            # Handle Inline: Collapse to single space
            new_text = " "

        # Only update if there is a change to avoid unnecessary mutations
        if new_text != original_text:
            records.append(self.record_reformat(node, new_text))
        return records

    def _handle_newline_node(self, node: Node, idx: int, siblings: List[Node]) -> List[Node]:
        """
        Handles newline nodes, particularly for removing empty lines.

        Args:
            node (Node): The current newline node being inspected.
            idx (int): The index of the current child node being inspected.
            siblings (List[Node]): The list of sibling nodes.

        Returns:
            List[Node]: A list of nodes to delete.
        """
        to_delete: List[Node] = []

        if not node.parent:
            return []

        if not self._is_empty_line(node, siblings, idx):
            return []
        to_delete.append(node)

        # Skip whitespace after newline
        i = idx + 1
        while i < len(siblings) and siblings[i].type == "whitespace":
            to_delete.append(siblings[i])
            i += 1

        return to_delete

    def _is_empty_line(self, node: Node, siblings: List[Node], idx: int) -> bool:
        """
        Detects empty lines in CST, which is when this node pattern occurs:
        - newline + newline
        - newline + (any number of whitespace) + newline

        Args:
            node (Node): The current newline node being inspected.
            siblings (List[Node]): The list of sibling nodes.
            idx (int): The index of the current child node being inspected.

        Returns:
            bool: True if the pattern matches an empty line, False otherwise.
        """

        i = idx + 1

        # Skip over any whitespace nodes immediately following this newline
        while i < len(siblings) and siblings[i].type == "whitespace":
            i += 1

        return i < len(siblings) and siblings[i].type == "newline"

    def apply(self, root: Node, context: MutationContext) -> List[MutationRecord]:
        """
        Traverses the CST and collects all mutation operations without modifying the tree.

        All structural changes (deletions) are deferred until traversal completes.

        Args:
            root (Node): The root node of the CST to mutate.
            context (MutationContext): The mutation context for tracking changes.

        Returns:
            List[MutationRecord]: A list of all mutation records generated during traversal.
        """
        records, to_delete = self._apply_collect(root, context)

        for node in to_delete:
            if node.parent:
                records.append(self.record_delete(node.parent, node))

        return records

    def _apply_collect(
        self, root: Node, context: MutationContext
    ) -> tuple[List[MutationRecord], List[Node]]:
        """
        Collects mutation records and nodes to delete without mutating the tree during traversal.

        Returns:
            (records, to_delete)
        """
        records: List[MutationRecord] = []
        to_delete: List[Node] = []

        children = list(root.children)
        for idx, child in enumerate(children):

            if child.type == "whitespace":
                records.extend(self._normalize_whitespace(child))

            elif child.type == "newline" and self._level >= 1:
                to_delete.extend(self._handle_newline_node(child, idx, root.children))
            else:
                records.extend(self._handle_structural_spacing(root, child, idx, context))

            records_child, delete_child = self._apply_collect(child, context)
            records.extend(records_child)
            to_delete.extend(delete_child)

        return records, to_delete
