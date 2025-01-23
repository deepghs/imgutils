"""
Exception class used to indicate that a transform object cannot be parsed by a specific parser.
"""


class NotParseTarget(Exception):
    """
    Exception raised when a transform object cannot be parsed by a specific parser.

    This exception is used internally by transform parsers to indicate that they are not
    able to handle the given transform object, allowing the parsing system to try the
    next available parser.
    """
    pass
